
from mindspore.nn.loss.loss import LossBase
import mindspore
import mindspore.nn as nn
import math
import numpy as np
import mindspore.ops as ops

import mindspore.numpy as mnp

class CosineSimilarity(LossBase):
    """
    Computes representation similarity.

    Args:
        similarity (str): 'dot' or 'cosine'. Default: 'cosine'.
        reduction (str): 'none', 'sum', 'mean' (all along dim -1). Default: 'none'.
        zero_diagonal (bool): If True,  diagonals of results will be set to zero. Default: True.

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        >>> import numpy as np
        >>> from mindspore import nn
        >>>
        >>> test_data = np.array([[1, 3, 4, 7], [2, 4, 2, 5], [3, 1, 5, 8]])
        >>> metric = nn.CosineSimilarity()
        >>> metric.clear()
        >>> metric.update(test_data)
        >>> square_matrix = metric.eval()
        >>> print(square_matrix)
        [[0.  0.94025615  0.95162452]
         [0.94025615  0.  0.86146098]
         [0.95162452  0.86146098  0.]]
    """
    def __init__(self, similarity='cosine', reduction='none', zero_diagonal=True):
        super().__init__()
        similarity_list = ['dot', 'cosine']
        reduction_list = ['none', 'sum', 'mean']
        self.similarity = similarity


    def construct(self, inputs):
        """
        Updates the internal evaluation result with 'inputs'.

        Args:
            inputs (Union[Tensor, list, numpy.ndarray]): The input matrix.
        """
        input_data = inputs
        
        if self.similarity == 'cosine':
            data = mnp.norm(input_data, ord=2, axis=1)
            input_data = input_data / mnp.expand_dims(data, 1)

        sqr_mtx_res = mnp.dot(input_data, input_data.transpose(1, 0)) 
        sqr_mtx_res = mnp.mean(sqr_mtx_res, axis=-1)

        return sqr_mtx_res

class do_Loss(LossBase):

    def __init__(self,reduction='mean'):
        """Initialize L1Loss."""
        super(do_Loss, self).__init__(reduction)
        self.l1 = mindspore.nn.L1Loss(reduction='mean')
        #self.metric = nn.CosineSimilarity()
        self.metric = CosineSimilarity()
        self.cat = ops.Concat(0)
    def construct(self, output, _label):
        spatital_loss = self.l1(output, _label) * 85
        N,C,H,W = output.shape
        spectral_loss = 0 

        for i in range(N):
            x = self.cat([output[i].reshape(1,C*H*W),_label[i].reshape(1,C*H*W)])

            #self.metric.update(x)
            #y = self.metric.eval()[0,1]
            y = self.metric(x)
            spectral_loss = spectral_loss +(1- y[0])

            #self.metric.clear()
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





