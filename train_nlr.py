import os 
os.system('pip install opencv-contrib-python-headless')
import mindspore
import  sys
import  time
import  glob
import  numpy as np
import  logging
import  argparse

from mindspore.train.callback import ModelCheckpoint, CheckpointConfig, LossMonitor,TimeMonitor
from mindspore import Tensor, Model

from mindspore import context, DatasetHelper, connect_network_with_dataset
from mindspore import dtype as mstype
from mindspore import nn,DynamicLossScaleManager
from mindspore.parallel._utils import _get_device_num

from mindspore.common import set_seed
from mindspore.context import ParallelMode
from mindspore.communication.management import init, get_rank, get_group_size
from nlr import NLRNet
from mindspore import dtype as mstype
from loss import do_Loss

from data import get_dataset
context.set_context(mode=context.PYNATIVE_MODE, device_target="GPU")
random_seed = 1996
np.random.seed(random_seed)

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

from mindspore.train.callback import Callback

                             
# Training settings
parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
#The dataset location is placed under /dataset
parser.add_argument('--traindata', default="/dataset/" ,help='path to train dataset')
parser.add_argument('--testdata', default="/dataset/test" ,help='path to test dataset')
parser.add_argument('--epoch_size', type=int, default=40, help='how much epoch to train')
parser.add_argument('--batch_size', type=int, default=16, help='how much batch_size in epoch')
if __name__ == '__main__':

    net = NLRNet()
    args, unknown = parser.parse_known_args()
    data = get_dataset()
    batch_num = data.train_dataset.get_dataset_size()
    #rank = set_device(args)

    criterion = do_Loss()
    optimizer = mindspore.nn.Adam(net.trainable_params(),learning_rate=0.0005)
    trainwtihloss = CustomWithLossCell(net,criterion)
    model = Model(network=trainwtihloss, optimizer=optimizer)
    config_ck = CheckpointConfig(save_checkpoint_steps=data.train_dataset.get_dataset_size(),
                                 keep_checkpoint_max=6)
    time_cb = TimeMonitor(data_size=data.train_dataset.get_dataset_size())
    
    ckpt_save_dir = '/tmp/model'

    ckpoint_cb = ModelCheckpoint(prefix='uldr', directory=ckpt_save_dir,
                                 config=config_ck)
    loss_cb = LossMonitor(100)

    print(_get_device_num())
    print("begin train")
    model.train(int(args.epoch_size), data.train_dataset,
                callbacks=[time_cb, ckpoint_cb, loss_cb],
                dataset_sink_mode=False)
    print("train success")