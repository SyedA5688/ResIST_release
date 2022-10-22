import os
os.environ["CUDA_VISIBLE_DEVICES"] = "3,2,1,0"

import time
import torch.nn.functional as functional
import torch.distributed as dist
import argparse
import torch
from pthflops import count_ops
from random import seed

from data import (
    get_cifar10_loaders,
    get_cifar100_loaders,
    get_svhn_loaders,
)


def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    """3x3 convolution with padding"""
    return torch.nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                           padding=dilation, groups=groups, bias=False, dilation=dilation)


class BasicBlock(torch.nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None):
        super(BasicBlock, self).__init__()
        if groups != 1 or base_width != 64:
            raise ValueError('BasicBlock only supports groups=1 and base_width=64')
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = norm_layer(planes)
        self.relu = torch.nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = norm_layer(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        if self.downsample is not None:
            identity = self.downsample(x)
        out += identity
        out = self.relu(out)
        return out


class PreActBlock(torch.nn.Module):
    """
    Pre-activation version of the BasicBlock.
    """
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(PreActBlock, self).__init__()
        self.bn1 = torch.nn.BatchNorm2d(in_planes, affine=True, track_running_stats=False)
        self.conv1 = torch.nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = torch.nn.BatchNorm2d(planes, affine=True, track_running_stats=False)
        self.conv2 = torch.nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.downsample = None
        if stride != 1 or in_planes != self.expansion*planes:
            self.downsample = torch.nn.Sequential(
                torch.nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False)
            )

    def forward(self, x):
        out = functional.relu(self.bn1(x))
        shortcut = self.downsample(out) if self.downsample is not None else x
        out = self.conv1(out)
        out = self.conv2(functional.relu(self.bn2(out)))
        out += shortcut
        return out


class PreActResNet(torch.nn.Module):
    # taken from https://github.com/kuangliu/pytorch-cifar

    def __init__(self, block, num_blocks, out_size=512, num_classes=10):
        super(PreActResNet, self).__init__()
        self.in_planes = 64
        self.num_blocks = num_blocks
        self.conv1 = torch.nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.fc = torch.nn.Linear(out_size*block.expansion, num_classes)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return torch.nn.Sequential(*layers)

    def forward(self, x):
        out = self.conv1(x)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = functional.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        return out


def PreActResNet101(blocks=[3, 4, 23, 3], out_size=512, num_classes=10):
    return PreActResNet(PreActBlock, blocks, out_size=out_size, num_classes=num_classes)

def PreActResNet152(blocks=[3, 4, 36, 3], out_size=512, num_classes=10):
    return PreActResNet(PreActBlock, blocks, out_size=out_size, num_classes=num_classes)

def PreActResNet200(blocks=[3, 4, 50, 3], out_size=512, num_classes=10):
    return PreActResNet(PreActBlock, blocks, out_size=out_size, num_classes=num_classes)


test_rank = 0
test_total_time = 0

def broadcast_module(specs, module:torch.nn.Module):
    group = dist.new_group(list(range(specs['world_size'])))
    for para in module.parameters():
        dist.broadcast(para.data, src=0, group=group, async_op=False)
    dist.destroy_process_group(group)

def all_reduce_module(specs, args, module):
    group = dist.group.WORLD
    for para in module.parameters():
        dist.all_reduce(para.data, op=dist.ReduceOp.SUM, group=group)
        para.data = para.data.div_(specs['world_size'])
    

def reduce_module(specs, args, module:torch.nn.Module):
    group = dist.new_group(list(range(specs['world_size'])))
    for para in module.parameters():
        dist.reduce(para.data, dst=0, op=dist.ReduceOp.SUM, group=group)
        if args.rank == 0: # compute average
            para.data = para.data.div_(specs['world_size'])
    dist.destroy_process_group(group)


class DistributedResNetModel:
    def __init__(self, model:PreActResNet):
        self.base_model = model

    def prepare_eval(self):
        return 
 
    def prepare_train(self, args):
        return

    def dispatch_model(self, specs, args):
        for idx in range(1, len(self.base_model.layer3)):
            broadcast_module(specs, self.base_model.layer3[idx])
    
    def sync_model(self, specs, args):
        # print('running all reduce')
        all_reduce_module(specs, args, self.base_model.conv1)
        all_reduce_module(specs, args, self.base_model.layer1)
        all_reduce_module(specs, args, self.base_model.layer2)
        all_reduce_module(specs, args, self.base_model.layer4)
        all_reduce_module(specs, args, self.base_model.fc)
        all_reduce_module(specs, args, self.base_model.layer3[0])
        # print('finish all reduce')

        # print('running reduce')
        for idx in range(1, len(self.base_model.layer3)):
            reduce_module(specs, args, self.base_model.layer3[idx])
        # print('finish reduce')

    def ini_sync_dispatch_model(self, specs, args):
        broadcast_module(specs, self.base_model.conv1)
        broadcast_module(specs, self.base_model.layer1)
        broadcast_module(specs, self.base_model.layer2)
        broadcast_module(specs, self.base_model.layer4)
        broadcast_module(specs, self.base_model.fc)
        broadcast_module(specs, self.base_model.layer3[0])
        for idx in range(1, len(self.base_model.layer3)):
            broadcast_module(specs, self.base_model.layer3[idx])

    def prepare_whole_model(self, specs, args):
        return


def train(specs, args, model_name, dist_model: DistributedResNetModel, optimizer, device, train_loader, num_iter):
    # employ a step schedule for the sub nets
    lr = specs.get('lr', 1e-2)
    epoch = 0

    # training loop
    flops_counter = 0

    # training loop
    for i, (data, target) in enumerate(train_loader):
        data = data.to(device)
        target = target.to(device)
        if num_iter % specs['repartition_iter'] == 0:
            if num_iter > 0:
                # print('dispatching model')
                dist_model.dispatch_model(specs, args)
                # print('finish dispatch')
            optimizer = torch.optim.SGD(
                    dist_model.base_model.parameters(), lr=lr,
                    momentum=specs.get('momentum', 0.9), weight_decay=specs.get('wd', 5e-4))

        optimizer.zero_grad()
        output = dist_model.base_model(data)

        # Count FLOPs
        ops, returned_data = count_ops(dist_model.base_model, data)
        flops_counter += ops

        loss = functional.cross_entropy(output, target)
        loss.backward()
        optimizer.step()
        if (
                ((num_iter + 1) % specs['repartition_iter'] == 0) or
                (i == len(train_loader) - 1 and epoch == specs['epochs'])):
            # print('syncing the model')
            dist_model.sync_model(specs, args)
            # print('finished sync')
            # num_sync = num_sync+1
            # end_time = time.time()
            # elapsed_time = end_time - start_time
            # print('Epoch {}, Iter {}, Rank {}: Train Num sync {} total time {:3.2f}s'.format(epoch, i, args.rank, num_sync, elapsed_time))
            # if args.rank == 0:
            #     if num_sync == 1:
            #         train_time_log[num_sync - 1] = elapsed_time
            #     else:
            #         train_time_log[num_sync - 1] = train_time_log[num_sync - 2] + elapsed_time
            #
            #     # Update train loss and accuracy lists
            #     temp_train_loss = agg_train_loss if i == 0 else agg_train_loss / i
            #     train_acc = train_num_correct / total_ex
            #     train_loss_log[num_sync - 1] = temp_train_loss
            #     train_acc_log[num_sync - 1] = train_acc

                # print('total time {:3.2f}s'.format(train_time_log[num_sync - 1]))
                # print('total broadcast time', test_total_time)
            
            # print('preparing and testing')
            dist_model.prepare_whole_model(specs, args)
        num_iter = num_iter + 1

    return flops_counter


def main():
    specs = {
        'test_type': 'ist_resnet',      # should be either ist or baseline
        'model_type': 'preact_resnet',  # use to specify type of resnet to use in baseline
        'use_valid_set': False,
        'model_version': 'v1',          # only used for the mobilenet tests
        'dataset': 'cifar100',
        'repartition_iter': 50,         # number of iterations to perform before re-sampling subnets
        'epochs': 40,
        'world_size': 4,                # number of subnets to use during training
        'layer_sizes': [3, 4, 23, 3],   # used for resnet baseline, number of blocks in each section
        'expansion': 1.,
        'lr': 0.01,
        'momentum': 0.9,
        'wd': 5e-4,
        'log_interval': 5,
    }

    parser = argparse.ArgumentParser(description='PyTorch ResNet (IST distributed)')
    # parser.add_argument('--dataset', type=str, default='cifar10')
    parser.add_argument('--dist-backend', type=str, default='nccl', metavar='S',
                        help='backend type for distributed PyTorch')
    parser.add_argument('--dist-url', type=str, default='tcp://127.0.0.1:9914', metavar='S',
                        help='master ip for distributed PyTorch')
    parser.add_argument('--rank', type=int, default=0, metavar='R',
                        help='rank for distributed PyTorch')
    parser.add_argument('--repartition_iter', type=int, default=50, metavar='N',
                         help='keep model in local update mode for how many iteration (default: 5)')
    parser.add_argument('--lr', type=float, default=0.1, metavar='LR',
                         help='learning rate (default: 1.0 for BN)')
    parser.add_argument('--pytorch-seed', type=int, default=3, metavar='S',
                        help='random seed (default: -1)')
    parser.add_argument('--use-cuda', default=True, type=lambda x: (str(x).lower() == 'true'),
                        help='if this is set to True, will use cuda to train')
    parser.add_argument('--cuda-id', type=int, default=0, metavar='N',
                        help='cuda index, if the instance has multiple GPUs.')
    parser.add_argument('--model_name', type=str, default='cifar10_local_iter')
    parser.add_argument('--save-dir', type=str, default='./runs/LocalSGD/', metavar='D',
                        help='directory where experiment will be saved')
    args = parser.parse_args()

    specs['repartition_iter'] = args.repartition_iter
    specs['lr'] = args.lr

    if args.pytorch_seed == -1:
        torch.manual_seed(args.rank)
        seed(0)
    else:
        torch.manual_seed(args.pytorch_seed*args.rank)
        seed(args.pytorch_seed)  # This makes sure, node use the same random key so that they does not need to sync partition info.
    print(args.cuda_id, torch.cuda.device_count())
    if args.use_cuda:
        assert args.cuda_id < torch.cuda.device_count()
        device = torch.device('cuda', args.cuda_id)
    else:
        device = torch.device('cpu')
    dist.init_process_group(backend=args.dist_backend, init_method=args.dist_url,
                            rank=args.rank, world_size=specs['world_size'])
    global test_rank
    test_rank=args.rank
    if specs['dataset'] == 'cifar10':
        out_size = 512
        for_cifar = True
        num_classes = 10
        input_size = 32
        trn_dl, test_dl = get_cifar10_loaders(specs.get('use_valid_set', False))
        criterion = torch.nn.CrossEntropyLoss()
    elif specs['dataset'] == 'cifar100':
        out_size = 512
        for_cifar = True
        num_classes = 100
        input_size = 32
        trn_dl, test_dl = get_cifar100_loaders(specs.get('use_valid_set', False))
        criterion = torch.nn.CrossEntropyLoss()
    elif specs['dataset'] == 'svhn':
        for_cifar = True # size of data is the same
        num_classes = 10
        input_size = 32
        out_size = 512
        trn_dl, test_dl = get_svhn_loaders()
        criterion = torch.nn.CrossEntropyLoss()
    else:
        raise NotImplementedError(f'{specs["dataset"]} dataset not supported')

    model_name = args.model_name

    print("Training from scratch!")
    dist_model = DistributedResNetModel(
            model=PreActResNet101(out_size=out_size, num_classes=num_classes).to(device))
    num_iter = 0

    print("Running initial sync")
    dist_model.ini_sync_dispatch_model(specs, args)
    optimizer = torch.optim.SGD(dist_model.base_model.parameters(), lr=specs.get('lr', 1e-2),
                    momentum=specs.get('momentum', 0.9), weight_decay=specs.get('wd', 5e-4))

    flops = train(specs, args, model_name, dist_model, optimizer, device, trn_dl, num_iter)
    time.sleep(5)
    print("Rank {} worker, {} operations".format(args.rank, flops))


if __name__ == '__main__':
    main()
