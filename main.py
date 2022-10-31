# Copyright 2021 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================
"""train"""
import os
import argparse
import ast
import os
import sys
import json
import yaml
import time
import numpy as np 

import mindspore
from mindspore import Model
from mindspore import context
from mindspore import nn
from mindspore import dtype as mstype
from mindspore.common import set_seed
from mindspore.train.callback import ModelCheckpoint, CheckpointConfig, LossMonitor, TimeMonitor
from mindspore.context import ParallelMode

from mindspore.communication.management import get_group_size
from mindspore.communication.management import get_rank
from mindspore.communication.management import init
from mindspore.parallel._utils import (_get_device_num, _get_gradients_mean,
                                       _get_parallel_mode, _get_enable_parallel_optimizer)
import moxing as mox
from src.nlrinit import NLRNet
from src.loss import do_Loss
from src.tools import set_device
from src.data import get_dataset,_get_rank_info


print('workroot:' + os.getcwd())
workroot = os.getcwd()
parser = argparse.ArgumentParser(description='MindSpore Lenet Example')

# define 2 parameters for running on modelArts
# data_url,train_url是固定用于在modelarts上训练的参数，表示数据集的路径和输出模型的路径
parser.add_argument('--data_path',
                    help='path to training/inference dataset folder',
                    default= workroot + '/data/')

parser.add_argument('--save_path',
                    help='model folder to save/load',
                    default= workroot + '/model/')

parser.add_argument("--batch_size", default=8, type=int, metavar="N",
                    help="mini-batch size (default: 256), this is the total "
                            "batch size of all GPUs on the current node when "
                            "using Data Parallel or Distributed Data Parallel")


parser.add_argument("--device_id", default=0, type=int, help="Device Id")
parser.add_argument("--device_num", default=1, type=int, help="device num")
parser.add_argument("--epochs", default=80, type=int, metavar="N", help="number of total epochs to run")

parser.add_argument("--seed", default=1996, type=int, help="seed for initializing training. ")
parser.add_argument("--save_every", default=20, type=int, help="Save every ___ epochs(default:2)")

random_seed = 1996
np.random.seed(random_seed)
mindspore.set_seed(random_seed)


class CustomWithLossCell(mindspore.nn.Cell):
    """连接前向网络和损失函数"""

    def __init__(self, backbone, loss_fn):
        """输入有两个，前向网络backbone和损失函数loss_fn"""
        super(CustomWithLossCell, self).__init__(auto_prefix=False)
        self._backbone = backbone
        self._loss_fn = loss_fn

    def construct(self, data,data1,data2,data3,label):
        output = self._backbone(data,data1,data2,data3)                 # 前向计算得到网络输出
        return self._loss_fn(output,label)  # 得到多标签损失值
    
def main():
    
    print(os.listdir(workroot))
    args = parser.parse_args()

    if "DEVICE_NUM" not in os.environ.keys():
        os.environ["DEVICE_NUM"] = str(args.device_num)
        os.environ["RANK_SIZE"] = str(args.device_num)   

    #初始化数据和模型存放目录
    data_dir = args.data_path  #先在训练镜像中定义数据集路径
    train_dir = args.save_path  #先在训练镜像中定义输出路径
    reduce_dir = workroot + 'data/reduce/'
    full_dir = workroot + 'data/full/'
    print(os.getcwd())
    
   if not os.path.exists(data_dir):
        os.makedirs(data_dir)
    if not os.path.exists(train_dir):
        os.makedirs(train_dir)


    set_seed(args.seed)
    
    mode = {
        0: context.GRAPH_MODE,
        1: context.PYNATIVE_MODE
    }
    device_num, rank_id = _get_rank_info()

    if device_num == 1 :
        context.set_context(mode=mode[0], device_target=args.device_target)
    else:
        context.set_context(mode=mode[0], device_target=args.device_target)
        
    net = NLRNet()

    data = get_dataset(data_dir)

    batch_num = data.train_dataset.get_dataset_size()

    import mindspore.nn as nn
    milestone = [40, 80]
    learning_rates = [0.001, 0.0005]
    lr = nn.piecewise_constant_lr(milestone,learning_rates)

    criterion = do_Loss()
    optimizer = mindspore.nn.Adam(net.trainable_params(),learning_rate=0.0005)
    trainwtihloss = CustomWithLossCell(net,criterion)
    model = Model(network=trainwtihloss, optimizer=optimizer)
    config_ck = CheckpointConfig(save_checkpoint_steps=data.train_dataset.get_dataset_size(),
                                 keep_checkpoint_max=40)
    time_cb = TimeMonitor(data_size=data.train_dataset.get_dataset_size())
    
    ckpt_save_dir = train_dir

    ckpoint_cb = ModelCheckpoint(prefix='nlr', directory=ckpt_save_dir,
                                 config=config_ck)
    loss_cb = LossMonitor(100)

    
    print("begin train")
    model.train(int(args.epochs), data.train_dataset,
                callbacks=[time_cb, ckpoint_cb, loss_cb],
                dataset_sink_mode=False)
    print("train success")


if __name__ == '__main__':
    main()