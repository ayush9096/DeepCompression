import numpy as np
import math
import torch
import os

import torch.nn as Parameter
from torch.nn.modules.module import Module
import torch.nn.functional as F

class PruningModule(Module):

    def prune_by_percentile(self, q = 0.5, **kwargs):
        """
        Note : The pruning percentile is based on all layer's parameters concatenated
        Args : 
            q (float) : percentile in float
            **kwargs : 
        """

        # Calculate percentile value
        alive_parameters = []
        for name, p in self.named_parameters():
            # Not pruning Bias 
            if 'bias' in name or 'mask' in name:
                continue
            tensor = p.data.cpu().numpy()
            alive = tensor[np.nonzero(tensor)]
            alive_parameters.append(alive)
        
        all_alives = np.concatenate(alive_parameters)
        percentile_value = np.percentile(abs(all_alives), q)

        print(" Pruning with Threhold : ", percentile_value)

        # Prune the weight and mask
        for name, module in self.named_modules():
            if name in ['fc1', 'fc2', 'fc3']:
                module.prune(threshold=percentile_value)
        
    
    def prune_by_std(self, s = 0.25):
        """
        Note : 's' is a quality parameter / senstivity value according to paper
        The pruning threshold is chosen as a quality parameter multiplied by the standard deviation of a layer's weights.
        """

        for name, module in self.named_modules():
            if name in ['fc1', 'fc2', 'fc3']:
                threshold = np.std(module.weight.data.cpu().numpy()) * s
                print(" Pruning with threshold : ", threshold, " for layer name : ", name)
                module.prune(threshold)
    


    
class MaskedLinear(module):
    """
        Applies a Masked Linear Transformation to the Incoming Data:
        :math: `y = (A * M)x + b`

        Args:
            in_features : size of each input sample
            out_features : size of each output sample
            bias : If set to False, the layer will not learn an additive bias
                Default: ``True``
        
        Shape:
            - Input: :math:`(N, *, inFeatures) where `*` means any number of parameters
            - Output: :math:`(N, *, outFeatures) where `*` means any number of parameters

        Attributes:
            weight: the learnable weights of the module of shape (outFeatures, inFeatures)
            bias: the learnable bias of the module of shape (outFeatures)
            mask: the unlearnable mask for the weight of shape (outFeatures, inFeatures)
    """


    def __init__(self, inFeatures, outFeatures, bias=True):
        super(MaskedLinear, self).__init__()

        self.inFeatures = inFeatures
        self.outFeatures = outFeatures
        self.weight = Parameter(torch.Tensor(outFeatures, inFeatures))

        # Mask initialized with ones
        self.mask = Parameter(torch.ones([outFeatures, inFeatures]), requires_grad = False)
        if bias:
            self.bias = Parameter(torch.Tensor(outFeatures))
        else:
            self.register_parameter('bias', None)
        
        self.reset_parameters()
    

    def reset_parameters(self):

        stdv = 1.0/ math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)
    
    def forward(self, input):
        return F.linear(input, self.weight * self.mask, self.bias)
    
    
    def __repr__(self):
        return self.__class__.__name__+ '(' + 'inFeatures=' + str(self.inFeatures) 
        + ', outFeatures=' + str(self.outFeatures) + ', bias=' + str(self.bias is not None) + ')'


    def prune(self, threshold):
        weightDev = self.weight.device
        maskDev = self.mask.device

        # Convert Tensors to numpy and calculate
        tensor = self.weight.data.cpu().numpy()
        mask = self.mask.data.cpu().numpy()

        new_mask = np.where(abs(tensor) < threshold, 0, mask)

        # Apply New Weight and Mask
        self.weight.data = torch.from_numpy(tensor * new_mask).to(weightDev)
        self.mask.data = torch.from_numpy(new_mask).to(maskDev)

    