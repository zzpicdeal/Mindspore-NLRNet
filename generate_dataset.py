# -*- coding: utf-8 -*-
"""
Created on 2020/1/11 14:59

@author: Evan Chen
"""

from resource_manager import ResourceManager


class GenerDataSet():
    def __init__(self, resource_manager: ResourceManager, for_train=True, dtype='fr_test'):
        self.for_train = for_train
        self.dtype = dtype

        #self.to_tensor = transforms.ToTensor()
        self.resource_manager = resource_manager

        if dtype == 'train':
            self.count = len(self.resource_manager.train_blocks)
        elif dtype == 'test':
            self.count = len(self.resource_manager.test_blocks)
        else:
            self.count = len(self.resource_manager.fr_test_blocks)

    def __getitem__(self, index):
        """return input, label"""
        if self.dtype == 'train':
            input, label = self.resource_manager.train_blocks[index]
        elif self.dtype == 'test':
            input, label = self.resource_manager.test_blocks[index]
        else:
            input, label = self.resource_manager.fr_test_blocks[index]

        #return [self.to_tensor(item) for item in input], self.to_tensor(label)
        return [item.transpose((2,0,1)) for item in input], label.transpose((2,0,1))

    def __len__(self):
        return self.count
