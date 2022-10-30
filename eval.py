import mindspore
import  sys
import  time
import  glob
import  numpy as np
import  logging
import  argparse
import os
from functools import reduce
from mindspore.train.callback import ModelCheckpoint, CheckpointConfig, LossMonitor,TimeMonitor
from mindspore import Tensor, Model

from mindspore import context, DatasetHelper, connect_network_with_dataset
from mindspore import dtype as mstype
from mindspore import nn,DynamicLossScaleManager
from mindspore.parallel._utils import _get_device_num

from mindspore.common import set_seed
from mindspore.context import ParallelMode
from mindspore.communication.management import init, get_rank, get_group_size
from src.nlrinit import NLRNet
from mindspore import dtype as mstype
from src.loss import do_Loss
from src.tools import sam, ergas, scc, D_lambda, D_s_2, qindex
from src.data import get_dataset

context.set_context(mode=context.PYNATIVE_MODE, device_target="Ascend")

random_seed = 1996
np.random.seed(random_seed)




def process_output(collect_output, collect_label, vol_nums):
    # remove rest output
    if vol_nums != 0:

        collect_output[-1] = collect_output[-1][-vol_nums:, :, :, :]
        collect_label[-1] = collect_label[-1][-vol_nums:, :, :, :]

    # condat all batchsize
    collect_output = np.row_stack(np.array(collect_output))
    collect_label = np.row_stack(np.array(collect_label))

    # slice net output
    collect_output[collect_output < 0] = 0
    collect_output[collect_output > 1] = 1

    return collect_output, collect_label

def ref_assement(collect_output, collect_label):
    """down_resolution index test"""
    h, w, c = collect_output[0].shape
    element_nums = h * w * c
    _sam, _erags, _scc, _qn = [], [], [], []
    for item in range(collect_output.shape[0]):
        label_block = collect_label[item, :, :, :]
        output_block = collect_output[item, :, :, :]

        if len(label_block[label_block == 0]) / element_nums > 0.5:
            continue

        _sam.append(sam(label_block, output_block))
        _erags.append(ergas(output_block, label_block))
        _scc.append(scc(output_block, label_block))
        _qn.append(qindex(output_block, label_block, block_size=128))
    return _sam,_erags,_scc,_qn

def test_img_restore( output, fr=False):
    # _output = reduce(lambda x, y: np.r_[x, y], output)
    _output = output[-1] if len(output)==2 else output

    rows, row, = [], []

    col_blocks = 5120 // 128 
    for ix in range(_output.shape[0]):
        if ix % col_blocks == 0 and ix != 0:
            rows.append(np.transpose(reduce(lambda x, y: np.r_[x, y], row), [1, 0, 2]))  # WxHxC -> HxWxC
            row = []
        row.append(np.transpose(_output[ix, :, :, :], [1, 0, 2]))  # HxWxC -> WxHxC
        # if ix == _output.shape[0]-1:
        #     rows.append(np.transpose(reduce(lambda x, y: np.r_[x, y], row), [1, 0, 2]))  # WxHxC -> HxWxC
        #     row = []
    rows.append(np.transpose(reduce(lambda x, y: np.r_[x, y], row), [1, 0, 2]))
    test_img = reduce(lambda x, y: np.r_[x, y], rows)
    print(test_img.shape)
    return test_img

workroot = '/home/work/user-job-dir/'
parser = argparse.ArgumentParser(description='MindSpore')

# define 2 parameters for running on modelArts
# data_url,train_url是固定用于在modelarts上训练的参数，表示数据集的路径和输出模型的路径
parser.add_argument('--ckpt',
                    help='path to ckpt',
                    default= './80/init-uldr-69_700.ckpt')

parser.add_argument('--eval_type',
                    help='test or fr_test',
                    default='test')
parser.add_argument('--data_path',
                    help='full data path',
                    default=workroot + 'data/{}')
def main():
    
    
    args = parser.parse_args()
    net = NLRNet()
    mindspore.load_checkpoint(args.ckpt,net)
    net.set_train(False)
    if args.eval_type == 'test':
        print('load data')
        data = get_dataset(args.data_path,train=args.eval_type)
        print('start test')
        batch_num = data.train_dataset.get_dataset_size()

        collect_output, collect_label = [], []
        for data_ in data.train_dataset.create_dict_iterator():
            keys = list(data_.keys())
            out = net(data_[keys[0]],data_[keys[1]],data_[keys[2]],data_[keys[3]])
            out_numpy = out.transpose(0,2,3,1).asnumpy()
            label_numpy = data_[keys[4]].transpose(0,2,3,1).asnumpy()
            collect_output.append(out_numpy)
            collect_label.append(label_numpy)
        output, label = process_output(collect_output, collect_label, vol_nums = 16)
        sam,erags,scc,qn = ref_assement(output, label)

        print('erags :',np.array(erags).mean())
    else:
        print('load data')
        data = get_dataset(args.data_path,train=args.eval_type)
        print('start test')
        collect_output, collect_label = [], []
        for data_ in data.train_dataset.create_dict_iterator():
            keys = list(data_.keys())
            out = net(data_[keys[0]],data_[keys[1]],data_[keys[2]],data_[keys[3]])
            out_numpy = out.transpose(0,2,3,1).asnumpy()
            label_numpy = data_[keys[4]].transpose(0,2,3,1).asnumpy()
            collect_output.append(out_numpy)
            collect_label.append(label_numpy)
        output, label = process_output(collect_output, collect_label, vol_nums = 16)
        from resource_manager import ResourceManager
        rm = ResourceManager(resource=r'/home/ma-user/work/{}',fr_test=True)
        fr_fake = test_img_restore(output, fr=True)
        D_s_idx = D_s_2(fr_fake, rm.fr_test_lrms,rm.fr_pan,satellite='IKONOS')
        D_lambda_idx = D_lambda(rm.fr_test_lrms, fr_fake,block_size=64)
        QNR_idx = (1 - D_lambda_idx) ** 1. * (1 - D_s_idx) ** 1.
        print('D_lambda : %s, D_s : %s,:QNR : %s' %(D_lambda_idx,D_s_idx,QNR_idx))
main()
        