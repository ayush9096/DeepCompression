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
            