
from mindspore.nn.loss.loss import LossBase
import mindspore
import mindspore.nn as nn
import math
import numpy as np
import mindspore.ops as ops



class do_Loss(LossBase):

    def __init__(self,reduction='mean'):
        """Initialize L1Loss."""
        super(do_Loss, self).__init__(reduction)
        self.l1 = mindspore.nn.L1Loss(reduction='mean')
        self.metric = nn.CosineSimilarity()
        self.cat = ops.Concat(0)
    def construct(self, output, _label):
        spatital_loss = self.l1(output, _label) * 85
        N,C,H,W = output.shape
        spectral_loss = 0 
        for i in range(N):
            x = self.cat([output[i].reshape(1,C*H*W),_label[i].reshape(1,C*H*W)])

            self.metric.update(x)
            spectral_loss = spectral_loss +(1- self.metric.eval()[0,1])

            self.metric.clear()
        spectral_loss = (spectral_loss/N)* 15

        # band shuffle
        sq = [1, 2, 3, 4, 5, 6, 7, 0]
        # shuffle real_img
        base = _label[:, sq, :, :]
        new_label = _label - base
        # shuffle fake_img
        base = output[:, sq, :, :]
        new_fake = output - base
        spectral_loss2 = self.l1(new_label, new_fake) * 15
        loss = spatital_loss + spectral_loss + spectral_loss2
        return self.get_loss(loss)





