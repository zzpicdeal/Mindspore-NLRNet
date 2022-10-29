import pickle
import cv2 as cv
#from cv2.ximgproc import guidedFilter
import numpy as np
from os import walk
from os.path import join
from scipy.io import loadmat
from itertools import product
from functools import reduce
import random
# import gdal

class ResourceManager(object):
    def __init__(self, block_size=128, block_num = 1600, network='nlrnet', test_img='record_6.mat', dataset='/reduce', resource=r'resource/{}',
                 warm_start=False,fr_test=False, seed=1024):
        np.random.seed(seed)
        random.seed(seed)

        self.network = network
        self.block_size = block_size
        self.block_num = block_num

        self.train_blocks = {}
        self.test_blocks = {}  # down resolution test

        self.fr_test_blocks = []  # full resolution test
        self.test_max_c = 0
        # self.train_max_c = 0

        # the test img
        self.test_img = test_img
        # the dataset type
        self.dataset = dataset
        # the source path
        self.resource = resource

        if not warm_start:
             # res resolution dataset
            if not fr_test:
                #print(resource.format(dataset))
                test_data, train_data = self.get_train_data_ls(resource.format(dataset))

                #self.train_max_c, self.train_blocks = self.sequential_split(train_data_ls, self.block_size)
                self.train_blocks = self.random_split(train_data)
                self.test_max_c, self.test_blocks["small"] = self.sequential_split(test_data["small"], self.block_size//4)
                # print("small", self.test_max_c)
                self.test_max_c, self.test_blocks["big"] = self.sequential_split(test_data["big"], self.block_size)
                # print("big", self.test_max_c)


                # there are not qb-real
            
            record = loadmat(join(self.resource.format('/full'),'full6.mat')) # self.dataset+'_real'

                
            lrms = record['ms'].astype(np.float32)
            pan = record['pan'].astype(np.float32)
            # print(lrms.shape,pan.shape)

            dsize = pan.shape
            self.fr_test_lrms = lrms
            self.fr_pan = pan[:,:,np.newaxis]
            # handle the full resolution dataset
            self.process_fr_test_data(dsize, self.dataset)
            # handle the res dataset
            if not fr_test:
                self.preprocess()
                print('self.fr_test_max_c:',self.fr_test_max_c,'self.block_size;',self.block_size)
            '''
            with open('cache/ResourceManager.model', 'wb') as fb:
                pickle.dump({'train': self.train_blocks,
                             'test': self.test_blocks,
                             'fr_test': self.fr_test_blocks,
                             'fr_test_lrms': self.fr_test_lrms,
                             'fr_pan': self.fr_pan,
                             'block_size': self.block_size,
                            #  'train_max_c': self.train_max_c,
                             'test_max_c': self.test_max_c,
                             'fr_test_max_c': self.fr_test_max_c}, fb)
            '''

        else:
            with open('cache/ResourceManager.model', 'rb') as fb:
                parameter_dict = pickle.load(fb)
                self.block_size = parameter_dict['block_size']
                # self.train_max_c = parameter_dict['train_max_c']
                self.train_blocks = parameter_dict['train']
                self.test_max_c = parameter_dict['test_max_c']
                self.test_blocks = parameter_dict['test']
                self.fr_test_blocks = parameter_dict['fr_test']
                self.fr_test_lrms = parameter_dict['fr_test_lrms']
                self.fr_test_max_c = parameter_dict['fr_test_max_c']
                self.fr_pan = parameter_dict['fr_pan']

    def preprocess(self):
        offset = len(self.train_blocks["big"])
        stack_input = {}
        stack_input["record"] = self.train_blocks["big"] + self.test_blocks["big"]
        stack_input["lrms"] = self.train_blocks["small"] + self.test_blocks["small"]
        dsize = int(self.block_size * 0.25)

        samples = []
        for ix, record in enumerate(zip(stack_input["record"], stack_input["lrms"])):
            if self.network == 'psgan':
                samples.append(self._preprocess_psgan(record, dsize, self.dataset))
            if self.network == 'mddl':
                samples.append(self._preprocess_mddl(record, dsize, self.dataset))
            if self.network == 'msdcnn':
                samples.append(self._preprocess_msdcnn(record, dsize, self.dataset))
            if self.network == 'pcnn':
                samples.append(self._preprocess_pcnn(record, dsize, self.dataset))
            if self.network == 'gfrnet':
                samples.append(self._preprocess_gfrnet(record, dsize, self.dataset))
            if self.network == 'pannet':
                samples.append(self._preprocess_pannet(record, dsize, self.dataset))
            if self.network == 'gppnn':
                samples.append(self._preprocess_gppnn(record, dsize, self.dataset))
            if self.network == 'nlrnet':
                samples.append(self._preprocess_nlrnet(record, dsize, self.dataset))
            if self.network == 'lsgan':
                samples.append(self._preprocess_lsgan(record, dsize, self.dataset))
            if self.network == 'mhfnet':
                samples.append(self._preprocess_mhfnet(record, dsize, self.dataset))
            if self.network == 'pmfnet':
                samples.append(self._preprocess_pmfnet(record, dsize, self.dataset))
            if self.network == 'nlunet':
                samples.append(self._preprocess_nlunet(record, dsize, self.dataset))
            if self.network == 'vpnet':
                samples.append(self._preprocess_vpnet(record, dsize, self.dataset))
            if self.network == 'fusionnet':
                samples.append(self._preprocess_fusionnet(record, dsize, self.dataset))
            if self.network == 'dircnn':
                samples.append(self._preprocess_dircnn(record, dsize, self.dataset))
            if self.network == 'tcrnet':
                samples.append(self._preprocess_tcrnet(record, dsize, self.dataset))
            if self.network == 'sescgan':
                samples.append(self._preprocess_sescgan(record, dsize, self.dataset))
            if self.network == 'hrnet':
                samples.append(self._preprocess_hrnet(record, dsize, self.dataset))
            if self.network in ['resdense', 'resdense_wsc', 'resdense_w12', 'resdense_w24']:
                samples.append(self._preprocess_resdense(record, dsize, self.dataset))
        self.train_blocks = samples[:offset]
        self.test_blocks = samples[offset:]

    def process_fr_test_data(self, dsize=5120,dataset='gf2'):
        samples = []

        if self.network == 'psgan':
            channels = 4 if dataset == 'gf2' else 8
            ms = self._upsample(self.fr_test_lrms, (dsize[1], dsize[0]))
            input = np.c_[ms, self.fr_pan]
            self.fr_test_max_c, self.fr_test_blocks = self.sequential_split(input, self.block_size)
            for record in self.fr_test_blocks:
                ms = record[:,:,:-1]
                pan = record[:, :, -1:]
                samples.append(([ms, pan], ms))
        elif self.network == 'fusionnet':
            channels = 4 if dataset == 'gf2' else 8
            ms = self._upsample(self.fr_test_lrms, (dsize[1], dsize[0]))
            input = np.c_[ms, self.fr_pan]
            self.fr_test_max_c, self.fr_test_blocks = self.sequential_split(input, self.block_size)
            for record in self.fr_test_blocks:
                ms = record[:,:,:-1]
                pan = record[:, :, -1:]
                samples.append(([ms, pan], ms))
        elif self.network == 'mddl':
            channels = 4 if dataset == 'gf2' else 8
            ms = self._upsample(self.fr_test_lrms, (dsize[1], dsize[0]))
            ms_hp = self._high_pass_filter(ms)
            input = np.c_[ms, ms_hp, self.fr_pan]
            self.fr_test_max_c, self.fr_test_blocks = self.sequential_split(input, self.block_size)
            for record in self.fr_test_blocks:
                ms = record[:, :, :channels]
                ms_hp = record[:, :, channels:-1]
                pan = record[:,:,-1:]
                samples.append(([ms, ms_hp, pan], ms))
        elif self.network == 'mhfnet':
            ms = self.fr_test_lrms
            upms = self._upsample(self.fr_test_lrms, (dsize[1], dsize[0]))
            input = np.c_[upms, self.fr_pan]
            self.fr_test_max_c_ms, self.fr_test_ms_blocks = self.sequential_split(ms, int(self.block_size // 4))
            self.fr_test_max_c, self.fr_test_blocks = self.sequential_split(input, self.block_size)
            for ms, record in zip(self.fr_test_ms_blocks, self.fr_test_blocks):
                # ms = ms[:,:,:]
                upms = record[:,:,:-1]
                pan = record[:, :, -1:]
                samples.append(([ms, pan, upms], upms))  
        elif self.network == 'vpnet':
            ms = self.fr_test_lrms
            upms = self._upsample(self.fr_test_lrms, (dsize[1], dsize[0]))
            input = np.c_[upms, self.fr_pan]
            self.fr_test_max_c_ms, self.fr_test_ms_blocks = self.sequential_split(ms, int(self.block_size // 4))
            self.fr_test_max_c, self.fr_test_blocks = self.sequential_split(input, self.block_size)
            for ms, record in zip(self.fr_test_ms_blocks, self.fr_test_blocks):
                # ms = ms[:,:,:]
                upms = record[:,:,:-1]
                pan = record[:, :, -1:]
                samples.append(([ms, pan, upms], upms))                              
        elif self.network == 'msdcnn':
            channels = 4 if dataset == 'gf2' else 8
            ms = self._upsample(self.fr_test_lrms, (dsize[1], dsize[0]))
            input = np.c_[ms, self.fr_pan]
            self.fr_test_max_c, self.fr_test_blocks = self.sequential_split(input, self.block_size)
            for record in self.fr_test_blocks:
                ms = record[:, :, :channels]
                samples.append(([record], ms))
        elif self.network in ['pcnn', 'resdense']:
            channels = 4 if dataset == 'gf2' else 8
            ms = self._upsample(self.fr_test_lrms, (dsize[1], dsize[0]))
            ms = cv.blur(ms, (5, 5))
            input = np.c_[ms, self.fr_pan]
            self.fr_test_max_c, self.fr_test_blocks = self.sequential_split(input, self.block_size)
            for record in self.fr_test_blocks:
                ms = record[:,:,:channels]
                pan = record[:,:,channels][:,:,np.newaxis]
                samples.append(([ms, pan],ms))
        elif self.network == 'pannet':
            channels = 4 if dataset == 'gf2' else 8
            ms = self._upsample(self.fr_test_lrms, (dsize[1], dsize[0]))
            up_hp_lrms = self._upsample(self._high_pass_filter(self.fr_test_lrms), (dsize[1], dsize[0]))
            hp_pan = self._high_pass_filter(self.fr_pan.reshape(dsize[0], dsize[1]))[:,:,np.newaxis]
            input = np.c_[up_hp_lrms, hp_pan, ms]
            #input = [ms]
            self.fr_test_max_c, self.fr_test_blocks = self.sequential_split(input, self.block_size)

            for record in self.fr_test_blocks:
                stack_input = record[:, :, :(channels+1)]
                ms = record[:, :, (channels+1):(2*channels+1)]
                samples.append(([stack_input, ms], ms))
        elif self.network == 'gppnn':
            ms = self.fr_test_lrms
            upms = self._upsample(self.fr_test_lrms, (dsize[1], dsize[0]))
            input = np.c_[upms, self.fr_pan]
            self.fr_test_max_c_ms, self.fr_test_ms_blocks = self.sequential_split(ms, int(self.block_size // 4))
            self.fr_test_max_c, self.fr_test_blocks = self.sequential_split(input, self.block_size)
            for ms, record in zip(self.fr_test_ms_blocks, self.fr_test_blocks):
                # ms = ms[:,:,:]
                upms = record[:,:,:-1]
                pan = record[:, :, -1:]
                samples.append(([ms, pan], upms))
        elif self.network == 'pmfnet':
            ms = self.fr_test_lrms
            upms = self._upsample(self.fr_test_lrms, (dsize[1], dsize[0]))
            input = np.c_[upms, self.fr_pan]
            self.fr_test_max_c_ms, self.fr_test_ms_blocks = self.sequential_split(ms, int(self.block_size // 4))
            self.fr_test_max_c, self.fr_test_blocks = self.sequential_split(input, self.block_size)
            for ms, record in zip(self.fr_test_ms_blocks, self.fr_test_blocks):
                # ms = ms[:,:,:]
                upms = record[:,:,:-1]
                pan = record[:, :, -1:]
                samples.append(([ms, pan], upms))       
        elif self.network == 'nlunet':
            ms = self.fr_test_lrms
            upms = self._upsample(self.fr_test_lrms, (dsize[1], dsize[0]))
            input = np.c_[upms, self.fr_pan]
            self.fr_test_max_c_ms, self.fr_test_ms_blocks = self.sequential_split(ms, int(self.block_size // 4))
            self.fr_test_max_c, self.fr_test_blocks = self.sequential_split(input, self.block_size)
            for ms, record in zip(self.fr_test_ms_blocks, self.fr_test_blocks):
                # ms = ms[:,:,:]
                upms = record[:,:,:-1]
                pan = record[:, :, -1:]
                samples.append(([ms, pan], upms))              
        elif self.network == 'nlrnet':
            channels = 4 if dataset == 'gf2' else 8
            lrms = self.fr_test_lrms
            ms = self._upsample(lrms, (dsize[1], dsize[0]))
            lrpan = self._downsample(self.fr_pan, (dsize[1] // 4, dsize[0] // 4))[:,:,np.newaxis]
            input = np.c_[ms, self.fr_pan]
            self.fr_test_max_c, hr_input_blocks = self.sequential_split(input, self.block_size)

            self.block_size = self.block_size // 4
            input = np.c_[lrms, lrpan]
            _, lr_input_blocks = self.sequential_split(input, self.block_size)
            self.block_size = self.block_size * 4

            for ix in range(len(hr_input_blocks)):
                ms = hr_input_blocks[ix][:, :, :channels]
                pan = hr_input_blocks[ix][:, :, channels]

                lrms = lr_input_blocks[ix][:, :, :channels]
                lrpan = lr_input_blocks[ix][:, :, channels]
                # print(ms.shape, lrms.shape, lrpan.shape, pan.shape)
                samples.append(([ms, lrms, lrpan[:,:,np.newaxis], pan[:,:,np.newaxis]], ms))

        elif self.network == 'lsgan':
            channels = 4 if dataset == 'gf2' else 8
            ms = self._upsample(self.fr_test_lrms, (dsize[1], dsize[0]))
            input = np.c_[ms, self.fr_pan]
            self.fr_test_max_c, self.fr_test_blocks = self.sequential_split(input, self.block_size)
            for record in self.fr_test_blocks:
                ms = record[:, :, :channels]
                pan = record[:, :, channels]

                A_1 = -(np.eye(128, k=-127) + np.eye(128, k=1)) + np.eye(128)
                A_2 = -(np.eye(128, k=-126) + np.eye(128, k=2)) + np.eye(128)
                A_1 = A_1.astype(np.float32)
                A_2 = A_2.astype(np.float32)
                Pan1_h = np.dot(pan, A_1)[:, :, np.newaxis]
                Pan1_v = np.dot(A_1, pan)[:, :, np.newaxis]
                pan1 = np.c_[Pan1_h, Pan1_v]  # 一阶差??
                Pan2_h = np.dot(pan, A_2)[:, :, np.newaxis]
                Pan2_v = np.dot(A_2, pan)[:, :, np.newaxis]
                pan2 = np.c_[Pan2_h, Pan2_v]  # 二阶差分
                samples.append(([ms, pan1, pan2], ms))
        elif self.network == 'tcrnet':
            channels = 4 if dataset == 'gf2' else 8
            lrms = self.fr_test_lrms
            ms = self._upsample(lrms, (dsize[1], dsize[0]))
            input = np.c_[ms, self.fr_pan]
            self.fr_test_max_c, self.fr_test_blocks = self.sequential_split(input, self.block_size)
            for record in self.fr_test_blocks:
                ms = record[:, :, :channels]
                pan = record[:, :, channels]
                # ms_diff1_h, ms_diff1_v = self.getMSdiff(ms, channels)
                # ms_diff = np.c_[ms_diff1_h, ms_diff1_v]
                # samples.append(([ms, pan[:, :, np.newaxis], ms_diff], ms))
                samples.append(([ms, pan[:, :, np.newaxis]], ms))

        elif self.network == 'sescgan':
            channels = 4 if dataset == 'gf2' else 8
            ms = self._upsample(self.fr_test_lrms, (dsize[1], dsize[0]))
            input = np.c_[ms, self.fr_pan]
            self.fr_test_max_c, self.fr_test_blocks = self.sequential_split(input, self.block_size)
            self.fr_test_max_c_m, self.fr_test_blocks_m = self.sequential_split(self.fr_test_lrms, self.block_size//4)
            for lrms, record in zip(self.fr_test_blocks_m,self.fr_test_blocks):
                ms = record[:, :, :channels]
                pan = record[:, :, channels]
                pan = pan.reshape(self.block_size, self.block_size, 1)
                pan_v = self.diff_gradient(pan, gtype="ver")
                pan_h = self.diff_gradient(pan, gtype="hon")
                pan_vh = pan_v+pan_h
                # concat_pan
                #pan_vh = np.c_[pan_v, pan_h]
                # pan
                #pan_vh = pan
                samples.append(([ms,pan_vh,lrms], ms))
        elif self.network == 'hrnet':
            channels = 4 if dataset == 'gf2' else 8
            lrms = self.fr_test_lrms
            ms = self._upsample(lrms, (dsize[1], dsize[0]))

            input = np.c_[ms, self.fr_pan]
            self.fr_test_max_c, hr_input_blocks = self.sequential_split(input, self.block_size)

            self.block_size = self.block_size // 4
            _, lr_input_blocks = self.sequential_split(lrms, self.block_size)
            self.block_size = self.block_size * 4

            for ix in range(len(hr_input_blocks)):
                ms = hr_input_blocks[ix][:, :, :channels]
                pan = hr_input_blocks[ix][:, :, channels]
                lrms = lr_input_blocks[ix][:, :, :channels]
                samples.append(([ms, pan[:, :, np.newaxis], lrms], ms))

        else:
            self.fr_test_max_c = 0
            samples = []
            # print('没有对应的预处理方法')

        self.fr_test_blocks = samples

    #读取给定文件路径中的数据，返回的是一个测试数据和训练数据列表
    def get_train_data_ls(self, dir='resource\qb'):
        test_data = {}
        train_data_ls = {}
        train_data_ls["big"] = list()
        train_data_ls["small"] = list()
        dir, _, fname_ls = list(walk(dir))[0]

        for fname in fname_ls:
            print(fname)
            if self.test_img == fname:
                test_data["big"] = loadmat(join(dir, fname))['record']
                test_data["small"] = loadmat(join(dir, fname))['lrms']
            else:
                train_data_ls["big"].append(loadmat(join(dir, fname))['record'])
                train_data_ls["small"].append(loadmat(join(dir, fname))['lrms'])
   
        return test_data, train_data_ls

    def random_split(self, data):
        #  random split  the block size img
        blocks = {}
        blocks['big'] = []
        blocks['small'] = []

        coord = set()
        for ix, ts in enumerate(zip(data["small"], data["big"])):
            for _ in range(self.block_num):
                row, col, _ = ts[0].shape
                b_row, b_col, _ = ts[1].shape
                while True:
                    rr = np.random.randint(0, row)
                    rc = np.random.randint(0, col)
                    id_str = '{}{}'.format(rr, rc)

                    if id_str not in coord and rr + self.block_size//4 < row and rc + self.block_size//4 < col:
                        coord.add(id_str)

                        dr = rr + self.block_size//4
                        dc = rc + self.block_size//4
                        s_block = data['small'][ix][rr:dr, rc:dc, :]
                        blocks['small'].append(s_block)

                        dr = rr*4 + self.block_size
                        dc = rc*4 + self.block_size
                        b_block = data['big'][ix][rr*4:dr, rc*4:dc, :]

                        blocks['big'].append(b_block)
                        break

        return blocks

    def sequential_split(self, data, block_size = 64):
        # TODO test_max_c有误
        test_max_c, blocks = 0, []
        if(not isinstance(data, list)):
            data_ls = [data]
        else:
            data_ls = data
        for ix, td in enumerate(data_ls):
            row, col, _ = td.shape
            # make the data fit block_size
            max_r = (row // block_size) * block_size
            max_c = (col // block_size) * block_size
            test_max_c += max_c

            x = [r_ix for r_ix in range(0, max_r, block_size)]
            y = [c_ix for c_ix in range(0, max_c, block_size)]
            # the patch of pan/lrms/ms
            test_block_coord = list(product(x, y))

            for r, c in test_block_coord:
                dr = r + block_size
                dc = c + block_size
                block = td[r:dr, c:dc, :]
                blocks.append(block)
        return test_max_c, blocks

    def diff_gradient(self, block, gtype='hon'):

        g = np.eye(self.block_size, k=-1) - np.eye(self.block_size, k=0)
        g[0][-1] = -1

        shape = block.shape
        if len(shape) == 3:
            gblock = np.zeros_like(block)
            for b in range(shape[2]):
                gblock[:, :, b] = self.diff_gradient(block[:, :, b], gtype=gtype)
            return gblock
        else:
            return block.dot(g) if gtype == 'hon' else block.T.dot(g).T

    def test_img_restore(self, output, fr=False):
        # _output = reduce(lambda x, y: np.r_[x, y], output)
        _output = output[-1] if len(output)==2 else output

        rows, row, = [], []

        col_blocks = self.fr_test_max_c // self.block_size if fr else self.test_max_c // self.block_size
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
        #print(test_img.shape)
        return test_img

    def _high_pass_filter(self, img, ksize=(5, 5)):
        blur = cv.blur(img, ksize)
        high_pass_filtered = img - blur
        return high_pass_filtered

    def _downsample(self, img, dsize, ksize=(7, 7), interpolation=cv.INTER_AREA):
        blur = cv.GaussianBlur(img, ksize, 0)
        downsampled = cv.resize(img, dsize, interpolation=interpolation)
        return downsampled

    def _upsample(self, img, dsize, interpolation=cv.INTER_CUBIC):
        upsampled = cv.resize(img, dsize, interpolation=interpolation)
        return upsampled
    def _preprocess_msdcnn(self, record, dsize, dataset='gf2'):
        channels = 4 if dataset == 'gf2' else 8
        hrms, pan = record[:, :, :channels], record[:, :, channels:channels+1]
        pan = pan.reshape(self.block_size, self.block_size, 1)
        lrms = self._downsample(hrms, (dsize, dsize))
        ms = self._upsample(lrms, (self.block_size, self.block_size))
        return [np.c_[ms, pan]], hrms

    def _preprocess_psgan(self, record, dsize, dataset='gf2'):
        channels = 4 if dataset == 'gf2' else 8
        hrms, pan, ms = record[0][:, :, :channels].astype(np.float32), record[0][:, :, channels:channels+1].astype(np.float32), record[0][:, :, channels+1:].astype(np.float32)
        # hrms, pan = record[:, :, :channels], record[:, :, channels:channels+1]
        # lrms = self._downsample(hrms, (dsize, dsize))
        # ms = self._upsample(lrms, (self.block_size, self.block_size))
        return [ms, pan.reshape(self.block_size, self.block_size, 1)], hrms

    def _preprocess_mddl(self, record, dsize, dataset='gf2'):
        channels = 4 if dataset == 'gf2' else 8
        hrms, pan, ms = record[0][:, :, :channels].astype(np.float32), record[0][:, :, channels:channels+1].astype(np.float32), record[0][:, :, channels+1:].astype(np.float32)
        # print(hrms.shape, pan.shape, ms.shape)
        lrms = record[1].astype(np.float32)
        # print(lrms.shape)
        ms_hp = self._upsample(self._high_pass_filter(lrms), (self.block_size, self.block_size))

        return [ms, ms_hp, pan], hrms

    def _preprocess_pcnn(self, record, dsize, dataset='gf2'):
        channels = 4 if dataset == 'gf2' else 8
        hrms, pan, ms = record[0][:, :, :channels].astype(np.float32), record[0][:, :, channels:channels+1].astype(np.float32), record[0][:, :, channels+1:].astype(np.float32)
        # hrms, pan = record[:, :, :channels], record[:, :, channels:channels+1]
        # lrms = self._downsample(hrms, (dsize, dsize))
        # ms = self._upsample(lrms, (self.block_size, self.block_size))
        return [ms, pan.reshape(self.block_size, self.block_size,1)], hrms
        
    def _preprocess_fusionnet(self, record, dsize, dataset='gf2'):
        channels = 4 if dataset == 'gf2' else 8
        hrms, pan, ms = record[0][:, :, :channels].astype(np.float32), record[0][:, :, channels:channels+1].astype(np.float32), record[0][:, :, channels+1:].astype(np.float32)
        # hrms, pan = record[:, :, :channels], record[:, :, channels:channels+1]
        # lrms = self._downsample(hrms, (dsize, dsize))
        # ms = self._upsample(lrms, (self.block_size, self.block_size))
        return [ms, pan.reshape(self.block_size, self.block_size,1)], hrms

    def _preprocess_gfrnet(self, record, dsize, dataset='gf2'):
        channels = 4 if dataset == 'gf2' else 8
        hrms, pan = record[:, :, :channels].astype('uint8'), record[:, :, channels:channels + 1].astype('uint8')
        pan = pan.reshape(self.block_size, self.block_size, 1)
        lrms = self._downsample(hrms, (dsize, dsize))
        ms = self._upsample(lrms, (self.block_size, self.block_size))
        dst = cv.ximgproc.guidedFilter(pan, ms[:, :, 0][:, :, np.newaxis], 2, 0.001, -1)
        sum_filter = dst[:, :, np.newaxis]
        for i in range(1, channels):
            single_brand = ms[:, :, i][:, :, np.newaxis]
            dst = cv.ximgproc.guidedFilter(pan, single_brand, 2, 0.001, -1)
            sum_filter = np.c_[sum_filter, dst[:, :, np.newaxis]]

        return [sum_filter, ms], hrms

    def _preprocess_pannet(self, record, dsize, dataset='gf2'):
        channels = 4 if dataset == 'gf2' else 8
        # hrms, pan = record[:, :, :channels], record[:, :, channels:channels+1]
        hrms, pan, ms = record[0][:, :, :channels].astype(np.float32), record[0][:, :, channels:channels+1].astype(np.float32), record[0][:, :, channels+1:].astype(np.float32)
        # lrms = self._downsample(hrms, (dsize, dsize))
        lrms = record[1].astype(np.float32)
        # ms = self._upsample(lrms, (self.block_size, self.block_size))
        pan = pan.reshape(self.block_size, self.block_size)
        up_hp_lrms = self._upsample(self._high_pass_filter(lrms), (self.block_size, self.block_size))
        hp_pan = self._high_pass_filter(pan)
        hp_pan = hp_pan.reshape(self.block_size, self.block_size, 1)

        return [np.c_[up_hp_lrms, hp_pan], ms], hrms

    def _preprocess_gppnn(self, record, dsize, dataset='gf2'):
        channels = 4 if dataset == 'gf2' else 8
        hrms, pan, ms = record[0][:, :, :channels].astype(np.float32), record[0][:, :, channels:channels+1].astype(np.float32), record[0][:, :, channels+1:].astype(np.float32)
        lrms = record[1][:, :, :channels].astype(np.float32)
        pan = pan.reshape(self.block_size, self.block_size, 1)
        return [lrms, pan], hrms
        
    def _preprocess_pmfnet(self, record, dsize, dataset='gf2'):
        channels = 4 if dataset == 'gf2' else 8
        hrms, pan, ms = record[0][:, :, :channels].astype(np.float32), record[0][:, :, channels:channels+1].astype(np.float32), record[0][:, :, channels+1:].astype(np.float32)
        lrms = record[1][:, :, :channels].astype(np.float32)
        pan = pan.reshape(self.block_size, self.block_size, 1)
        return [lrms, pan], hrms 
        
    def _preprocess_nlunet(self, record, dsize, dataset='gf2'):
        channels = 4 if dataset == 'gf2' else 8
        hrms, pan, ms = record[0][:, :, :channels].astype(np.float32), record[0][:, :, channels:channels+1].astype(np.float32), record[0][:, :, channels+1:].astype(np.float32)
        lrms = record[1][:, :, :channels].astype(np.float32)
        pan = pan.reshape(self.block_size, self.block_size, 1)
        return [lrms, pan], hrms 
               
    def _preprocess_nlrnet(self, record, dsize, dataset='gf2'):
        channels = 4 if dataset == 'gf2' else 8
        # hrms, pan = record[:, :, :channels], record[:, :, channels]
        hrms, pan, ms = record[0][:, :, :channels].astype(np.float32), record[0][:, :, channels:channels+1].astype(np.float32), record[0][:, :, channels+1:].astype(np.float32)
        # lrms = self._downsample(hrms, (dsize, dsize))
        lrms = record[1].astype(np.float32)
        # ms = self._upsample(lrms, (self.block_size, self.block_size))
        lrpan = self._downsample(pan, (dsize, dsize)).reshape(dsize, dsize, 1)
        return [ms, lrms, lrpan, pan.reshape(self.block_size, self.block_size, 1)], hrms

    def _preprocess_mhfnet(self, record, dsize, dataset='gf2'):
        channels = 4 if dataset == 'gf2' else 8
        hrms, pan, ms = record[0][:, :, :channels].astype(np.float32), record[0][:, :, channels:channels+1].astype(np.float32), record[0][:, :, channels+1:].astype(np.float32)
        # hrms, pan = record[:, :, :channels], record[:, :, channels]
        # lrms = self._downsample(hrms, (dsize, dsize))
        lrms = record[1][:, :, :channels].astype(np.float32)
        return [lrms,pan.reshape(self.block_size, self.block_size, 1), ms],hrms

    def _preprocess_vpnet(self, record, dsize, dataset='gf2'):
        channels = 4 if dataset == 'gf2' else 8
        hrms, pan, ms = record[0][:, :, :channels].astype(np.float32), record[0][:, :, channels:channels+1].astype(np.float32), record[0][:, :, channels+1:].astype(np.float32)
        # hrms, pan = record[:, :, :channels], record[:, :, channels]
        # lrms = self._downsample(hrms, (dsize, dsize))
        lrms = record[1][:, :, :channels].astype(np.float32)
        return [lrms,pan.reshape(self.block_size, self.block_size, 1), ms],hrms
    def getMSdiff(self, ms,channels):
        ms = np.transpose(ms, (2, 0, 1))
        A1 = -(np.eye(self.block_size, k=-(self.block_size-1)) + np.eye(self.block_size, k=1)) + np.eye(self.block_size)
        A1 = A1.astype(np.float32)[np.newaxis, :, :]
        A_1 = A1
        for i in range(channels-1):
            A_1 = np.r_[A_1, A1]
        ms_diff1_h = np.matmul(ms, A_1)
        ms_diff1_v = np.matmul(A_1, ms)
        return np.transpose(ms_diff1_h, (1, 2, 0)), np.transpose(ms_diff1_v, (1, 2, 0))
    def getHHR_Pan(self, pan, ms,channels):
        mean_pan = np.mean(pan)
        std_pan = np.std(pan, ddof=1)
        ms_1 = ms[:, :, 0]
        z = (pan - mean_pan) / std_pan * np.std(ms_1) + np.mean(ms_1)
        z = z[:, :, np.newaxis]
        for i in range(1, channels):
            ms_i = ms[:, :, i]
            z_i = (pan - mean_pan) / std_pan * np.std(ms_i) + np.mean(ms_i)
            z_i = z_i[:, :, np.newaxis]
            z = np.c_[z, z_i]
        return z
    def _preprocess_lsgan(self, record, dsize, dataset='gf2'):
        # channels = 4 if dataset == 'gf2' else 8
        # hrms, pan = record[:, :, :channels], record[:, :, channels]
        # lrms = self._downsample(hrms, (dsize, dsize))
        # ms = self._upsample(lrms, (self.block_size, self.block_size))
        #
        # A_1 = -(np.eye(128, k=-127) + np.eye(128, k=1)) + np.eye(128)
        # A_2 = -(np.eye(128, k=-126) + np.eye(128, k=2)) + np.eye(128)
        # A_1 = A_1.astype(np.float32)
        # A_2 = A_2.astype(np.float32)
        # Pan1_h = np.dot(pan, A_1)[:, :, np.newaxis]
        # Pan1_v = np.dot(A_1, pan)[:, :, np.newaxis]
        # pan1 = np.c_[Pan1_h, Pan1_v]    #一阶差�?        # Pan2_h = np.dot(pan, A_2)[:, :, np.newaxis]
        # Pan2_v = np.dot(A_2, pan)[:, :, np.newaxis]
        # pan2 = np.c_[Pan2_h, Pan2_v]    #二阶差分
        # return [ms, pan1, pan2],hrms
        channels = 4 if dataset == 'gf2' else 8
        hrms, pan, ms = record[0][:, :, :channels].astype(np.float32), record[0][:, :, channels].astype(np.float32), record[0][:, :, channels+1:].astype(np.float32)

        A_1 = -(np.eye(128, k=-127) + np.eye(128, k=1)) + np.eye(128)
        A_2 = -(np.eye(128, k=-126) + np.eye(128, k=2)) + np.eye(128)
        A_1 = A_1.astype(np.float32)
        A_2 = A_2.astype(np.float32)
        Pan1_h = np.dot(pan, A_1)[:, :, np.newaxis]
        Pan1_v = np.dot(A_1, pan)[:, :, np.newaxis]
        pan1 = np.c_[Pan1_h, Pan1_v]    #一阶差�?        
        Pan2_h = np.dot(pan, A_2)[:, :, np.newaxis]
        Pan2_v = np.dot(A_2, pan)[:, :, np.newaxis]
        pan2 = np.c_[Pan2_h, Pan2_v]    #二阶差分
        return [ms, pan1, pan2],hrms
    def _preprocess_dircnn(self,record, dsize, dataset='gf2'):
        channels = 4 if dataset == 'gf2' else 8
        hrms, pan = record[:, :, :channels], record[:, :, channels]
        lrms = self._downsample(hrms, (dsize, dsize))
        ms = self._upsample(lrms, (self.block_size, self.block_size))
        ms_diff1_h, ms_diff1_v = self.getMSdiff(ms,channels)
        HHR_Pan = self.getHHR_Pan(pan,ms,channels)
        ms_diff = np.c_[ms_diff1_h, ms_diff1_v]
        input1 = HHR_Pan - ms
        input2 = np.c_[input1, ms_diff]
        return [input2, input1, HHR_Pan],hrms
    def _preprocess_tcrnet(self,record, dsize, dataset='gf2'):
        channels = 4 if dataset == 'gf2' else 8
        # hrms, pan = record[:, :, :channels], record[:, :, channels]
        hrms, pan, ms = record[0][:, :, :channels].astype(np.float32), record[0][:, :, channels].astype(np.float32), record[0][:, :, channels+1:].astype(np.float32)
        # lrms = self._downsample(hrms, (dsize, dsize))
        # ms = self._upsample(lrms, (self.block_size, self.block_size))
        # ms_diff1_h, ms_diff1_v = self.getMSdiff(ms,channels)
        # ms_diff = np.c_[ms_diff1_h, ms_diff1_v]
        # return [ms, pan.reshape(self.block_size, self.block_size,1), ms_diff],hrms
        return [ms, pan.reshape(self.block_size, self.block_size,1)],hrms
    def _preprocess_sescgan(self, record, dsize, dataset='gf2'):
        channels = 4 if dataset == 'gf2' else 8
        hrms, pan = record[:, :, :channels], record[:, :, channels]
        lrms = self._downsample(hrms, (dsize, dsize))
        ms = self._upsample(lrms, (self.block_size, self.block_size))
        pan = pan.reshape(self.block_size, self.block_size, 1)
        pan_v = self.diff_gradient(pan, gtype="ver")
        pan_h = self.diff_gradient(pan, gtype="hon")
        pan_vh = pan_v + pan_h
        # concat_pan
        #pan_vh = np.c_[pan_v,pan_h]
        # pan
        #pan_vh = pan
        return [ms, pan_vh, lrms],hrms
    def _preprocess_resdense(self, record, dsize, dataset='gf2'):
        channels = 4 if dataset == 'gf2' else 8
        hrms, pan = record[:, :, :channels], record[:, :, channels:channels+1]
        lrms = self._downsample(hrms, (dsize, dsize))
        ms = self._upsample(lrms, (self.block_size, self.block_size))
        return [ms, pan.reshape(self.block_size, self.block_size,1)], hrms
    def _preprocess_hrnet(self,record, dsize, dataset='gf2'):
        channels = 4 if dataset == 'gf2' else 8
        hrms, pan, ms = record[0][:, :, :channels].astype(np.float32), record[0][:, :, channels:channels+1].astype(np.float32), record[0][:, :, channels+1:].astype(np.float32)
        lrms = record[1].astype(np.float32)
        # print(hrms.shape,pan.shape, ms.shape, lrms.shape)
        return [ms, pan.reshape(self.block_size, self.block_size,1), lrms], hrms


if __name__ == '__main__':
    pass
    # ResourceManager()
