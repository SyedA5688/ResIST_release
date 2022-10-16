import os

os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"
os.environ["NCCL_DEBUG"] = "INFO"

import torch.nn.functional as functional
import torch.distributed as dist
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import argparse
import torch
import time
from datetime import datetime
# import torchvision.transforms as transforms
# from torchvision.datasets.cifar import CIFAR10
from random import shuffle, choice, seed

from data import (
    get_cifar10_loaders,
    get_cifar100_loaders,
    get_tin_data,
    get_svhn_loaders,
)
# from utils import get_demon_momentum, aggregate_resnet_optimizer_statistics


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
    plt.title("Periodic Central Training Loss Curves")
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
    plt.title("Periodic Central Training Acc Curves")
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
        if stride != 1 or in_planes != self.expansion*planes:
            self.downsample = torch.nn.Sequential(
                torch.nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False)
            )

    def forward(self, x):
        if self.active_flag: # Instead of zero out weights, this can also avoid computation.
            out = functional.relu(self.bn1(x))
            shortcut = self.downsample(out) if self.downsample is not None else x
            out = self.conv1(out)
            out = self.conv2(functional.relu(self.bn2(out)))*self.scale_constant
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


def sample_block_indices_with_overlap(num_sites, num_blocks, min_blocks_per_site):
    assert min_blocks_per_site < num_blocks
    total_blocks = max(num_sites * min_blocks_per_site, num_blocks - 1)
    full_perm = []
    while total_blocks > 0.:
        temp = [i for i in range(num_blocks - 1)]
        shuffle(temp)
        full_perm = full_perm + temp
        total_blocks -= (num_blocks - 1)
    blocks_per_site = max(int((num_blocks - 1) / num_sites), min_blocks_per_site)
    subnet_sizes = [blocks_per_site for x in range(num_sites)]
    remaining_blocks = num_blocks - sum(subnet_sizes) - 1
    if remaining_blocks > 0:
        for i in range(remaining_blocks):
            subnet_sizes[i] += 1
    start_idx = 0
    site_indices = []
    for i in range(num_sites):
        curr_size = subnet_sizes[i]
        curr_block = full_perm[start_idx: start_idx + curr_size]
        curr_block = [val + 1 for val in curr_block]
        curr_block = list(set(curr_block))
        assert not 0 in curr_block
        site_indices.append(curr_block)
        start_idx += curr_size
    for site_idx in range(len(site_indices)):
        while len(site_indices[site_idx]) < min_blocks_per_site:
            next_idx = choice([i + 1 for i in range(num_blocks - 1)])
            if not next_idx in site_indices[site_idx]:
                site_indices[site_idx].append(next_idx)
    return site_indices


test_rank = 0
test_total_time = 0
def broadcast_module(module: torch.nn.Module, rank_list=None,source=0):
    """
    This function broadcasts all parameters of the module passed in to all workers.
    """
    # print("Calling broadcast_module()")
    if rank_list is None:
        group = dist.group.WORLD
    else:
        group = dist.new_group(rank_list)

    for para in module.parameters():
        dist.broadcast(para.data, src=source, group=group, async_op=False)

    if rank_list is not None:
        dist.destroy_process_group(group)

def reduce_module(specs, args, module: torch.nn.Module, rank_list=None):
    """
    This function performs reduce on the parameters of the module passed in, meaning that
    the sum of the parameter copies across workers is passed into the rank 0 process, and
    then on the rank 0 process it is divided by the number of workers to average it.
    """
    # print("Calling reduce_module()")
    if rank_list is None:
        raise 'error'
    else:
        group = dist.new_group(rank_list)
    for para in module.parameters():
        dist.reduce(para.data, dst=min(rank_list), op=dist.ReduceOp.SUM, group=group)
        if args.rank == min(rank_list): # compute average
            if rank_list is None:
                para.data = para.data.div_(specs['world_size'])
            else:
                para.data = para.data.div_(len(rank_list))
    if rank_list is not None:
        dist.destroy_process_group(group)

def all_reduce_module(specs, args, module: torch.nn.Module, rank_list=None):
    """
    This function performs all-reduce on the parameters in the module passed in, meaning that
    all workers receive the sum of the parameter copies across all workers. Then, the parameters
    are divided based on the number of workers present. This is the function doing the federated
    averaging.
    """
    # print("Calling all_reduce_module()")
    group = dist.group.WORLD
    for para in module.parameters():
        dist.all_reduce(para.data, op=dist.ReduceOp.SUM, group=group)
        if rank_list is None:
            para.data = para.data.div_(specs['world_size'])
        else:
            para.data = para.data.div_(len(rank_list))
    if rank_list is not None:
        dist.destroy_process_group(group)

class ISTResNetModel():
    def __init__(self, model: PreActResNet, num_sites=4, min_blocks_per_site=0):
        self.base_model = model
        self.min_blocks_per_site = min_blocks_per_site
        self.num_sites = num_sites
        self.site_indices = None
        if min_blocks_per_site == 0:
            self.scale_constant = 1.0/num_sites
        else:
            # dropout prob becomes total blocks per site / total blocks in layer3
            self.scale_constant = max(1.0/num_sites, min_blocks_per_site/22)
        self.layer_server_list=[]

    def prepare_eval(self):
        """
        This function is called by rank 0 worker, is used to turn on all residual blocks in
        order to run eval on test set.
        """
        print("Calling prepare_eval()!")
        for i in range(1, self.base_model.num_blocks[2]):
            self.base_model.layer3[i].active_flag = True
            self.base_model.layer3[i].scale_constant = self.scale_constant

    def prepare_train(self, args):
        """
        This function is called by rank 0 worker, is used to turn active flags of residual
        blocks back to True/False depending on partitioning. It is used by rank 0 worker
        after test set evaluation.
        """
        print("Calling prepare_train()!")
        for i in range(1, self.base_model.num_blocks[2]):
            self.base_model.layer3[i].active_flag = i in self.site_indices[args.rank]
            self.base_model.layer3[i].scale_constant = 1.0

    def dispatch_model(self, specs, args):
        """
        This function broadcasts the ResNet layer 3 residual blocks to different workers. It is the
        dispatch function used during ResIST training to repartition ResNet.
        """
        # print("Calling dispatch_model()!")
        self.site_indices = sample_block_indices_with_overlap(num_sites=self.num_sites,
                                                         num_blocks=self.base_model.num_blocks[2],
                                                         min_blocks_per_site=self.min_blocks_per_site)
        for i in range(1, self.base_model.num_blocks[2]):
            current_group = []
            for site_i in range(self.num_sites):
                if i in self.site_indices[site_i]:
                    current_group.append(site_i)
            if not (self.layer_server_list[i] in current_group):
                current_group.append(self.layer_server_list[i])
            broadcast_module(self.base_model.layer3[i], rank_list=current_group,source=self.layer_server_list[i])
            self.base_model.layer3[i].active_flag = i in self.site_indices[args.rank]

    def sync_model(self, specs, args):
        """
        This function performs all reduce on the entire ResNet (layers 1,2,3,4) to sync up
        parameters.
        """
        # print("Calling sync_model()!")
        # aggregate conv1
        all_reduce_module(specs, args, self.base_model.conv1)
        # aggregate layer 1 & 2 & 4
        all_reduce_module(specs, args, self.base_model.layer1)
        all_reduce_module(specs, args, self.base_model.layer2)
        all_reduce_module(specs, args, self.base_model.layer4)
        # aggregate FC layer
        all_reduce_module(specs, args, self.base_model.fc)
        # apply IST aggregation here
        all_reduce_module(specs, args, self.base_model.layer3[0])
        self.layer_server_list=[-1]
        for i in range(1,self.base_model.num_blocks[2]):

            current_group = []
            for site_i in range(self.num_sites):
                if i in self.site_indices[site_i]:
                    current_group.append(site_i)
            self.layer_server_list.append(min(current_group))
            reduce_module(specs, args, self.base_model.layer3[i], rank_list=current_group)

    def ini_sync_dispatch_model(self, specs, args):
        """
        This function broadcasts the entire ResNet to all workers, including partitioning
        layer3 residual blocks
        """
        print("Calling ini_sync_dispatch_model()!")
        # broadcast conv1 
        broadcast_module(self.base_model.conv1, source=0)

        # # broadcast layer 1 & 2 & 4
        broadcast_module(self.base_model.layer1, source=0)

        broadcast_module(self.base_model.layer2, source=0)

        broadcast_module(self.base_model.layer4, source=0)

        # # broadcast FC layer
        broadcast_module(self.base_model.fc, source=0)

        broadcast_module(self.base_model.layer3[0], source=0)

        self.site_indices = sample_block_indices_with_overlap(num_sites=self.num_sites,
                                                              num_blocks=self.base_model.num_blocks[2],
                                                              min_blocks_per_site=self.min_blocks_per_site)

        # # apply IST here
        for i in range(1, self.base_model.num_blocks[2]):
            broadcast_module(self.base_model.layer3[i], source=0)
            self.base_model.layer3[i].active_flag = i in self.site_indices[args.rank]

    def prepare_whole_model(self, specs, args):
        """
        This function broadcasts the layer3 residual blocks of the ResNet to all workers
        """
        # print("Calling prepare_whole_model()!")
        for i in range(1, self.base_model.num_blocks[2]):

            current_group = []
            for site_i in range(self.num_sites):
                if i in self.site_indices[site_i]:
                    current_group.append(site_i)

            if not (current_group[0] == 0):
                broadcast_module(self.base_model.layer3[i], rank_list=[0, min(current_group)],
                                 source=min(current_group))


def train(specs, args, start_time, model_name, ist_model: ISTResNetModel, optimizer, device, train_loader, test_loader, epoch, num_sync, num_iter, train_time_log, train_loss_log, train_acc_log, test_loss_log, test_acc_log, expt_save_path):

    # employ a step schedule for the sub nets
    lr = specs.get('lr', 1e-2)
    if epoch > int(specs['epochs']*0.5):
        lr /= 10
    if epoch > int(specs['epochs']*0.75):
        lr /= 10
    if optimizer is not None:
        for pg in optimizer.param_groups:
            pg['lr'] = lr
    print(f'Learning Rate: {lr}')

    # training loop
    agg_train_loss = 0.
    train_num_correct = 0.
    total_ex = 0.

    centralized_training = False
    decentralized_train_counter = args.central_train_freq  # Count down iterations of decentralized training
    centralized_train_counter = 0

    for i, (data, target) in enumerate(train_loader):
        print("Epoch {}, Iteration {}. Batch {}/{}".format(epoch, num_iter, i, len(train_loader)))
        ##### Update centralized training flag: do central training every X epochs #####
        if not centralized_training and decentralized_train_counter <= 0:
            print("Doing centralized training for next {} iterations".format(args.central_train_iter))
            centralized_training = True
            centralized_train_counter = args.central_train_iter
            ist_model.sync_model(specs, args)  # All-reduce entire ResNet
            ist_model.prepare_whole_model(specs, args)  # broadcast layer3 residual blocks to all workers
            ist_model.prepare_eval()  # Turn on all residual blocks in workers

        ##### Training logic, if doing central training then only rank 0 node should do training #####
        if centralized_training:
            if args.rank == 0:
                print("Rank 0 doing centralized training...")
                data = data.to(device)
                target = target.to(device)
                optimizer.zero_grad()
                output = ist_model.base_model(data)
                loss = functional.cross_entropy(output, target)
                agg_train_loss += loss.item()
                loss.backward()
                optimizer.step()
                train_pred = output.max(1, keepdim=True)[1]  # get the index of the max log-probability
                total_ex += target.size(0)
                train_num_correct += train_pred.eq(target.view_as(train_pred)).sum().item()

                if (num_iter + 1) % specs['repartition_iter'] == 0 \
                        and (num_iter + 1) % (args.central_train_freq + args.central_train_iter) != 0:
                    # Will occur 1 iter before repartition
                    # (i == len(train_loader) - 1 and epoch == specs['epochs']))\
                    # and i != len(train_loader) - 1:
                    num_sync = num_sync + 1
                    end_time = time.time()
                    elapsed_time = end_time - start_time
                    print('Node {}: Train Num sync {}, total time {:3.2f}s'.format(args.rank, num_sync, elapsed_time))

                    if num_sync == 1:
                        train_time_log[num_sync - 1] = elapsed_time
                    else:
                        train_time_log[num_sync - 1] = train_time_log[num_sync - 2] + elapsed_time

                    # Update train loss and accuracy lists
                    temp_train_loss = agg_train_loss if i == 0 else agg_train_loss / i
                    train_acc = train_num_correct / total_ex
                    train_loss_log[num_sync - 1] = temp_train_loss
                    train_acc_log[num_sync - 1] = train_acc

                    print('total time {:3.2f}s'.format(train_time_log[num_sync - 1]))
                    # print(f'preparing and testing')
                    # ist_model.prepare_whole_model(specs, args)  # broadcast layer3 residual blocks to all workers
                    test(specs, args, ist_model, device, test_loader, epoch, num_sync, test_loss_log, test_acc_log)
                    # print('done testing')
                    ist_model.prepare_eval()  # Turn on all residual blocks in workers, need to do here after test
                    start_time = time.time()
            else:
                print("Rank {} waiting 1 iteration during centralized training period...".format(args.rank))
                time.sleep(0.2)
        else:
            print("Rank {} doing distributed training...".format(args.rank))
            # Repartition if not in a centralized training period
            if num_iter % specs['repartition_iter'] == 0 or i == 0:
                if num_iter > 0 and not specs["resume_training_first_epoch"]:
                    # print('repartitioning, running model dispatch')
                    ist_model.dispatch_model(specs, args)  # Repartition layer 3 residual blocks
                    # print('model dispatch finished')
                else:
                    specs["resume_training_first_epoch"] = False
                optimizer = torch.optim.SGD(
                        ist_model.base_model.parameters(), lr=lr,
                        momentum=specs.get('momentum', 0.9), weight_decay=specs.get('wd', 5e-4))

            data = data.to(device)
            target = target.to(device)
            optimizer.zero_grad()
            output = ist_model.base_model(data)
            loss = functional.cross_entropy(output, target)
            agg_train_loss += loss.item()
            loss.backward()
            optimizer.step()
            train_pred = output.max(1, keepdim=True)[1]  # get the index of the max log-probability
            total_ex += target.size(0)
            train_num_correct += train_pred.eq(target.view_as(train_pred)).sum().item()

            if (num_iter + 1) % specs['repartition_iter'] == 0:  # Will occur 1 iter before repartition
                # (i == len(train_loader) - 1 and epoch == specs['epochs'])):
                # print('running model sync')
                ist_model.sync_model(specs, args)  # All-reduce entire ResNet
                # print('model sync finished')
                num_sync = num_sync + 1
                end_time = time.time()
                elapsed_time = end_time - start_time
                print('Node {}: Train Num sync {}, total time {:3.2f}s'.format(args.rank, num_sync, elapsed_time))
                if args.rank == 0:
                    if num_sync == 1:
                        train_time_log[num_sync - 1] = elapsed_time
                    else:
                        train_time_log[num_sync - 1] = train_time_log[num_sync - 2] + elapsed_time

                    # Update train loss and accuracy lists
                    temp_train_loss = agg_train_loss / i if i != 0 else agg_train_loss
                    train_acc = train_num_correct / total_ex
                    train_loss_log[num_sync - 1] = temp_train_loss
                    train_acc_log[num_sync - 1] = train_acc

                    print('total time {:3.2f}s'.format(train_time_log[num_sync - 1]))

                # print(f'preparing and testing')
                ist_model.prepare_whole_model(specs, args)  # broadcast layer3 residual blocks to all workers
                test(specs, args, ist_model, device, test_loader, epoch, num_sync, test_loss_log, test_acc_log)
                # print('done testing')
                start_time = time.time()
        num_iter = num_iter + 1

        ##### If in a centralized training period, update flags as necessary #####
        if centralized_training:
            centralized_train_counter -= 1
            if centralized_train_counter <= 0:
                print("Centralized training period ended.")
                centralized_training = False
                decentralized_train_counter = args.central_train_freq
                ist_model.prepare_whole_model(specs, args)  # broadcast layer3 residual blocks to all workers
                ist_model.prepare_train(args)
                # print("Done preparing end of centralized training, rank {}".format(args.rank))
        else:
            decentralized_train_counter -= 1

    print("Epoch finished.")
    # save model checkpoint at the end of each epoch
    if args.rank == 0:
        np.savetxt(os.path.join(expt_save_path, '{}_train_time.log'.format(model_name)), train_time_log, fmt='%1.4f', newline=' ')
        np.savetxt(os.path.join(expt_save_path, '{}_test_loss.log'.format(model_name)), test_loss_log, fmt='%1.4f', newline=' ')
        np.savetxt(os.path.join(expt_save_path, '{}_test_acc.log'.format(model_name)), test_acc_log, fmt='%1.4f', newline=' ')
        np.savetxt(os.path.join(expt_save_path, '{}_train_loss.log'.format(model_name)), train_loss_log, fmt='%1.4f', newline=' ')
        np.savetxt(os.path.join(expt_save_path, '{}_train_acc.log'.format(model_name)), train_acc_log, fmt='%1.4f', newline=' ')

        checkpoint = {
                'model': ist_model.base_model.state_dict(),
                'epoch': epoch,
                'num_sync': num_sync,
                'num_iter': num_iter,
        }
        torch.save(checkpoint, os.path.join(expt_save_path, '{}_model.pth'.format(model_name)))
    else:
        time.sleep(3)
    return num_sync, num_iter, start_time, optimizer

def test(specs, args, ist_model: ISTResNetModel, device, test_loader, epoch, num_sync, test_loss_log, test_acc_log):
    # Do validation only on prime node in cluster.
    if args.rank == 0:
        ist_model.prepare_eval()
        ist_model.base_model.eval()
        agg_val_loss = 0.
        num_correct = 0.
        total_ex = 0.
        criterion = torch.nn.CrossEntropyLoss()
        for model_in, labels in test_loader:
            model_in = model_in.to(device)
            labels = labels.to(device)
            with torch.no_grad():
                model_output = ist_model.base_model(model_in)
                val_loss = criterion(model_output, labels)
            agg_val_loss += val_loss.item()
            _, preds = model_output.max(1)
            total_ex += labels.size(0)
            num_correct += preds.eq(labels).sum().item()
        agg_val_loss /= len(test_loader)
        val_acc = num_correct/total_ex
        print("Epoch {} Number of Sync {} Local Test Loss: {:.6f}; Test Accuracy: {:.4f}.\n".format(epoch, num_sync, agg_val_loss, val_acc))
        test_loss_log[num_sync - 1] = agg_val_loss
        test_acc_log[num_sync - 1] = val_acc
        ist_model.prepare_train(args)  # reset all scale constants
        ist_model.base_model.train()


def main():
    specs = {
        'test_type': 'ist_resnet',  # should be either ist or baseline
        'model_type': 'preact_resnet',  # use to specify type of resnet to use in baseline
        'use_valid_set': False,
        'model_version': 'v1',  # only used for the mobilenet tests
        'dataset': 'cifar100',
        'repartition_iter': 50,  # number of iterations to perform before re-sampling subnets
        'epochs': 40,  # 160 epochs in ResIST paper
        'world_size': 4,  # number of subnets to use during training
        'layer_sizes': [3, 4, 23, 3],  # used for resnet baseline, number of blocks in each section
        'expansion': 1.,
        'lr': 0.1,  # .01
        'momentum': 0.9,
        'wd': 5e-4,
        'log_interval': 5,
        'min_blocks_per_site': 0,  # used for the resnet ist, allow overlapping block partitions to occur
    }

    parser = argparse.ArgumentParser(description='PyTorch ResNet (IST distributed)')
    # parser.add_argument('--dataset', type=str, default='cifar10')
    parser.add_argument('--dist-backend', type=str, default='gloo', metavar='S',
                        help='backend type for distributed PyTorch (default: nccl)')
    parser.add_argument('--dist-url', type=str, default='tcp://127.0.0.1:9002', metavar='S',
                        help='master ip for distributed PyTorch')
    parser.add_argument('--rank', type=int, default=0, metavar='R',
                        help='rank for distributed PyTorch')
    parser.add_argument('--repartition_iter', type=int, default=50, metavar='N',
                        help='keep model in local update mode for how many iteration (default: 5)')
    parser.add_argument('--lr', type=float, default=0.1, metavar='LR',
                        help='learning rate (default: 1.0 for BN)')
    parser.add_argument('--pytorch-seed', type=int, default=10, metavar='S',
                        help='random seed (default: -1)')
    parser.add_argument('--use-cuda', default=True, type=lambda x: (str(x).lower() == 'true'),
                        help='if this is set to True, will use cuda to train')
    parser.add_argument('--cuda-id', type=int, default=0, metavar='N',
                        help='cuda index, if the instance has multiple GPUs.')
    parser.add_argument('--model_name', type=str, default='cifar100_local_iter')
    parser.add_argument('--save-dir', type=str, default='./runs/ResIST_centralized_training/', metavar='D',
                        help='directory where experiment will be saved')
    parser.add_argument('--central-train-freq', type=int, default=95, metavar='N',
                        help='perform centralized training every X iterations (default: 4)')
    parser.add_argument('--central-train-iter', type=int, default=5, metavar='N',
                        help='perform centralized training for Y iterations (default: 20)')
    args = parser.parse_args()

    specs['repartition_iter'] = args.repartition_iter
    specs['lr'] = args.lr

    if args.pytorch_seed == -1:
        torch.manual_seed(args.rank)
        seed(0)
    else:
        torch.manual_seed(args.rank*args.pytorch_seed)
        seed(args.pytorch_seed)
    #seed(0)  # This makes sure, node use the same random key so that they does not need to sync partition info.
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

    # check if model has been checkpointed already
    model_name = args.model_name

    # Create save directories
    if args.rank == 0 and not os.path.exists(args.save_dir):
        os.mkdir(args.save_dir)

    expt_save_path = os.path.join(args.save_dir, datetime.now().strftime('%Y-%m-%d-%H_%M_%S'))
    if args.rank == 0 and not os.path.exists(expt_save_path):
        os.mkdir(expt_save_path)
    # expt_save_path = os.path.join(args.save_dir, "2022-09-18-11_59_32")
    specs['resume_training_first_epoch'] = False

    if os.path.exists(expt_save_path) and specs['resume_training_first_epoch']:
        print('Loading model from a checkpoint!')
        ist_model = ISTResNetModel(
                model=PreActResNet101(out_size=out_size, num_classes=num_classes).to(device),
                num_sites=specs['world_size'], min_blocks_per_site=specs['min_blocks_per_site'])
        checkpoint = torch.load(os.path.join(expt_save_path, '{}_model.pth'.format(model_name)))
        ist_model.base_model.load_state_dict(checkpoint['model'])
        start_epoch = checkpoint['epoch']
        num_sync = checkpoint['num_sync']
        num_iter = checkpoint['num_iter']
        train_time_log = np.loadtxt(os.path.join(expt_save_path, '{}_train_time.log'.format(model_name))) if args.rank == 0 else None
        test_loss_log = np.loadtxt(os.path.join(expt_save_path, '{}_test_loss.log'.format(model_name))) if args.rank == 0 else None
        test_acc_log = np.loadtxt(os.path.join(expt_save_path, '{}_test_acc.log'.format(model_name))) if args.rank == 0 else None
        train_loss_log = np.loadtxt(os.path.join(expt_save_path, '{}_train_loss.log'.format(model_name))) if args.rank == 0 else None
        train_acc_log = np.loadtxt(os.path.join(expt_save_path, '{}_train_acc.log'.format(model_name))) if args.rank == 0 else None
        print(f'Starting at epoch {start_epoch}')
    else:
        print('Training model from scratch!')
        ist_model = ISTResNetModel(
                model=PreActResNet101(out_size=out_size, num_classes=num_classes).to(device),
                num_sites=specs['world_size'], min_blocks_per_site=specs['min_blocks_per_site'])
        train_time_log = np.zeros(2000) if args.rank == 0 else None
        test_loss_log = np.zeros(2000) if args.rank == 0 else None
        test_acc_log = np.zeros(2000) if args.rank == 0 else None
        train_loss_log = np.zeros(2000) if args.rank == 0 else None
        train_acc_log = np.zeros(2000) if args.rank == 0 else None
        start_epoch = 0
        num_sync = 0
        num_iter = 0

    print('running initial sync')
    ist_model.ini_sync_dispatch_model(specs, args)  # Broadcast entire ResNet to all workers, partition layer 3
    print('initial sync finished')
    epochs = specs['epochs']
    optimizer = None 
    start_time = time.time()
    for epoch in range(start_epoch + 1, epochs + 1):
        num_sync, num_iter, start_time, optimizer = train(
                specs, args, start_time, model_name, ist_model, optimizer, device,
                trn_dl, test_dl, epoch, num_sync, num_iter, train_time_log, train_loss_log,
                train_acc_log, test_loss_log, test_acc_log, expt_save_path)

    if args.rank == 0:
        # Plot loss and accuracy curves
        test_loss_log = test_loss_log[:num_sync]
        test_acc_log = test_acc_log[:num_sync]
        train_loss_log = train_loss_log[:num_sync]
        train_acc_log = train_acc_log[:num_sync]
        plot_loss_curves(train_loss_log, test_loss_log, num_sync,
                         save_path=os.path.join(expt_save_path, '{}_loss_curves.png'.format(model_name)))
        plot_acc_curves(train_acc_log, test_acc_log, num_sync,
                        save_path=os.path.join(expt_save_path, '{}_accuracy_curves.png'.format(model_name)))


if __name__ == '__main__':
    main()
