import os
import time
import argparse
from datetime import datetime
from random import seed

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import torch.nn.functional as functional

from data import (
    get_cifar100_loaders,
)

os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"


def plot_loss_curves(train_losses, test_losses, iter_count, save_path):
    assert len(train_losses) == len(test_losses) == iter_count, "Unequal sizes in loss curve plotting."
    time = list(range(iter_count))
    visual_df = pd.DataFrame({
        "Train Loss": train_losses,
        "Test Loss": test_losses,
        "Iterations": time
    })

    plt.rcParams.update({'font.size': 16})
    sns.lineplot(x='Iterations', y='Loss Value', hue='Loss Type',
                 data=pd.melt(visual_df, ['Iterations'], value_name="Loss Value", var_name="Loss Type"))
    plt.title("ResNet Training Loss Curves")
    plt.yscale("log")
    plt.savefig(save_path, bbox_inches='tight', facecolor="white")
    plt.close()


def plot_acc_curves(train_accs, test_accs, iter_count, save_path):
    assert len(test_accs) == iter_count, "Unequal sizes in accuracy curve plotting."
    time = list(range(iter_count))
    visual_df = pd.DataFrame({
        "Train Accuracy": train_accs,
        "Test Accuracy": test_accs,
        "Iterations": time
    })

    plt.rcParams.update({'font.size': 16})
    sns.lineplot(x='Iterations', y='Accuracy Value', hue='Accuracy Type',
                 data=pd.melt(visual_df, ['Iterations'], value_name="Accuracy Value", var_name="Accuracy Type"))
    plt.title("ResNet Training Acc Curves")
    plt.savefig(save_path, bbox_inches='tight', facecolor="white")
    plt.close()


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
    '''Pre-activation version of the BasicBlock.'''
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(PreActBlock, self).__init__()
        self.active_flag = True
        self.scale_constant = 1.0
        self.bn1 = torch.nn.BatchNorm2d(in_planes, affine=True, track_running_stats=False)
        self.conv1 = torch.nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = torch.nn.BatchNorm2d(planes, affine=True, track_running_stats=False)
        self.conv2 = torch.nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.downsample = None
        if stride != 1 or in_planes != self.expansion * planes:
            self.downsample = torch.nn.Sequential(
                torch.nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False)
            )

    def forward(self, x):
        if self.active_flag:  # Instead of zero out weights, this can also avoid computation.
            out = functional.relu(self.bn1(x))
            shortcut = self.downsample(out) if self.downsample is not None else x
            out = self.conv1(out)
            out = self.conv2(functional.relu(self.bn2(out))) * self.scale_constant
            out += shortcut
        else:
            out = x
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
        self.fc = torch.nn.Linear(out_size * block.expansion, num_classes)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
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


def train(specs, args, start_time, model_name, ddp_model, optimizer, device, train_loader, test_loader,
          epoch, num_iter, num_log, train_time_log, train_loss_log, train_acc_log, test_loss_log, test_acc_log,
          expt_save_path):
    # employ a step schedule for the subnets
    lr = specs.get('lr', 1e-2)
    if epoch > int(specs['epochs'] * 0.5):
        lr /= 10
    if epoch > int(specs['epochs'] * 0.75):
        lr /= 10
    if optimizer is not None:
        for pg in optimizer.param_groups:
            pg['lr'] = lr
    print(f'Learning Rate: {lr}')

    # training loop
    agg_train_loss = 0.
    train_num_correct = 0.
    total_ex = 0.

    for i, (data, target) in enumerate(train_loader):
        print("Iteration {}".format(num_iter))
        data = data.to(device)
        target = target.to(device)

        optimizer.zero_grad()
        output = ddp_model(data)
        loss = functional.cross_entropy(output, target)
        agg_train_loss += loss.item()
        loss.backward()
        optimizer.step()
        train_pred = output.max(1, keepdim=True)[1]  # get the index of the max log-probability
        total_ex += target.size(0)
        train_num_correct += train_pred.eq(target.view_as(train_pred)).sum().item()
        if (num_iter + 1) % specs['log_interval'] == 0:
            num_log = num_log + 1
            end_time = time.time()
            elapsed_time = end_time - start_time
            print('Node {}: Train Num sync {}, total time {:3.2f}s'.format(args.rank, num_log, elapsed_time))

            if args.rank == 0:
                if num_log == 1:
                    train_time_log[num_log - 1] = elapsed_time
                else:
                    train_time_log[num_log - 1] = train_time_log[num_log - 2] + elapsed_time

                # Update train loss and accuracy lists
                temp_train_loss = agg_train_loss if i == 0 else agg_train_loss / i
                train_acc = train_num_correct / total_ex
                train_loss_log[num_log - 1] = temp_train_loss
                train_acc_log[num_log - 1] = train_acc

                print('total time {:3.2f}s'.format(train_time_log[num_log - 1]))
                test(ddp_model, args, device, test_loader, epoch, num_log, test_loss_log, test_acc_log)
                print('done testing')
            start_time = time.time()
        num_iter = num_iter + 1

    if args.rank == 0:
        # save model checkpoint at the end of each epoch
        np.savetxt(os.path.join(expt_save_path, '{}_train_time.log'.format(model_name)),
                   train_time_log, fmt='%1.4f', newline=' ')
        np.savetxt(os.path.join(expt_save_path, '{}_test_loss.log'.format(model_name)),
                   test_loss_log, fmt='%1.4f', newline=' ')
        np.savetxt(os.path.join(expt_save_path, '{}_test_acc.log'.format(model_name)),
                   test_acc_log, fmt='%1.4f', newline=' ')
        np.savetxt(os.path.join(expt_save_path, '{}_train_loss.log'.format(model_name)),
                   train_loss_log, fmt='%1.4f', newline=' ')
        np.savetxt(os.path.join(expt_save_path, '{}_train_acc.log'.format(model_name)),
                   train_acc_log, fmt='%1.4f', newline=' ')

        checkpoint = {
            'model': ddp_model.state_dict(),
            'epoch': epoch,
            'num_sync': num_log,
            'num_iter': num_iter,
        }
        torch.save(checkpoint, os.path.join(expt_save_path, '{}_model.pth'.format(model_name)))
    return num_log, num_iter, start_time, optimizer


def test(ddp_model, args, device, test_loader, epoch, num_log, test_loss_log, test_acc_log):
    if args.rank == 0:
        ddp_model.eval()
        agg_val_loss = 0.
        num_correct = 0.
        total_ex = 0.
        criterion = torch.nn.CrossEntropyLoss()
        for model_in, labels in test_loader:
            model_in = model_in.to(device)
            labels = labels.to(device)
            with torch.no_grad():
                model_output = ddp_model(model_in)
                val_loss = criterion(model_output, labels)
            agg_val_loss += val_loss.item()
            _, preds = model_output.max(1)
            total_ex += labels.size(0)
            num_correct += preds.eq(labels).sum().item()
        agg_val_loss /= len(test_loader)
        val_acc = num_correct / total_ex
        print("Epoch {}, Test Loss: {:.6f}; Test Accuracy: {:.4f}.\n"
              .format(epoch, agg_val_loss, val_acc))
        test_loss_log[num_log - 1] = agg_val_loss
        test_acc_log[num_log - 1] = val_acc
        ddp_model.train()


def main():
    specs = {
        'test_type': 'ist_resnet',  # should be either ist or baseline
        'model_type': 'preact_resnet',  # use to specify type of resnet to use in baseline
        'use_valid_set': False,
        'model_version': 'v1',  # only used for the mobilenet tests
        'dataset': 'cifar100',
        'epochs': 200,
        'world_size': 4,  # number of subnets to use during training
        'layer_sizes': [3, 4, 23, 3],  # used for resnet baseline, number of blocks in each section
        'log_interval': 50,  # used for resnet baseline, number of blocks in each section
        'expansion': 1.,
        'momentum': 0.9,
        'wd': 5e-4,
        'min_blocks_per_site': 0,  # used for the resnet ist, allow overlapping block partitions to occur
    }

    parser = argparse.ArgumentParser(description='PyTorch ResNet')
    # parser.add_argument('--dataset', type=str, default='cifar10')
    parser.add_argument('--dist-backend', type=str, default='nccl', metavar='S',
                        help='backend type for distributed PyTorch')
    parser.add_argument('--dist-url', type=str, default='tcp://127.0.0.1:9915', metavar='S',
                        help='master ip for distributed PyTorch')
    parser.add_argument('--rank', type=int, default=0, metavar='R',
                        help='rank for distributed PyTorch')
    parser.add_argument('--lr', type=float, default=0.1, metavar='LR',
                        help='learning rate (default: 1.0 for BN)')
    parser.add_argument('--pytorch-seed', type=int, default=1, metavar='S',
                        help='random seed (default: -1)')
    parser.add_argument('--use-cuda', default=True, type=lambda x: (str(x).lower() == 'true'),
                        help='if this is set to True, will use cuda to train')
    parser.add_argument('--cuda-id', type=int, default=0, metavar='N',
                        help='cuda index, if the instance has multiple GPUs.')
    parser.add_argument('--model_name', type=str, default='cifar10_local_iter')
    parser.add_argument('--save-dir', type=str, default='./runs/ResNet_baseline/', metavar='D',
                        help='directory where experiment will be saved')
    args = parser.parse_args()
    specs['lr'] = args.lr

    if args.pytorch_seed == -1:
        torch.manual_seed(0)
        seed(0)
    else:
        torch.manual_seed(args.pytorch_seed)
        seed(args.pytorch_seed)
    # seed(0)  # This makes sure, node use the same random key so that they does not need to sync partition info.
    if args.use_cuda:
        assert args.cuda_id < torch.cuda.device_count()
        device = torch.device('cuda', args.cuda_id)
    else:
        device = torch.device('cpu')

    if specs['dataset'] == 'cifar100':
        out_size = 512
        num_classes = 100
        trn_dl, test_dl = get_cifar100_loaders(specs.get('use_valid_set', False))
        # for_cifar = True
        # input_size = 32
        # criterion = torch.nn.CrossEntropyLoss()
    else:
        raise NotImplementedError(f'{specs["dataset"]} dataset not supported')

    model_name = args.model_name

    # Create save directories
    if not os.path.exists(args.save_dir):
        os.mkdir(args.save_dir)

    expt_save_path = os.path.join(args.save_dir, datetime.now().strftime('%Y-%m-%d-%H_%M_%S'))
    if not os.path.exists(expt_save_path):
        os.mkdir(expt_save_path)

    print('Training model from scratch!')
    model = PreActResNet101(out_size=out_size, num_classes=num_classes).to(device)
    ddp_model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.rank], output_device=args.rank)

    train_time_log = np.zeros(2000)
    test_loss_log = np.zeros(2000)
    test_acc_log = np.zeros(2000)
    train_loss_log = np.zeros(2000)
    train_acc_log = np.zeros(2000)
    start_epoch = 0
    num_log = 0
    num_iter = 0

    epochs = specs['epochs']
    optimizer = torch.optim.SGD(ddp_model.parameters(), lr=args.lr,
                                momentum=specs.get('momentum', 0.9), weight_decay=specs.get('wd', 5e-4))
    start_time = time.time()
    for epoch in range(start_epoch + 1, epochs + 1):
        num_log, num_iter, start_time, optimizer = train(
            specs, args, start_time, model_name, ddp_model, optimizer, device,
            trn_dl, test_dl, epoch, num_iter, num_log, train_time_log, train_loss_log,
            train_acc_log, test_loss_log, test_acc_log, expt_save_path)

    # Plot loss and accuracy curves
    test_loss_log = test_loss_log[:num_log]
    test_acc_log = test_acc_log[:num_log]
    train_loss_log = train_loss_log[:num_log]
    train_acc_log = train_acc_log[:num_log]
    plot_loss_curves(train_loss_log, test_loss_log, num_log,
                     save_path=os.path.join(expt_save_path, '{}_loss_curves.png'.format(model_name)))
    plot_acc_curves(train_acc_log, test_acc_log, num_log,
                    save_path=os.path.join(expt_save_path, '{}_accuracy_curves.png'.format(model_name)))


if __name__ == '__main__':
    main()
