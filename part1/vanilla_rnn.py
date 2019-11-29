################################################################################
# MIT License
#
# Copyright (c) 2019
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to conditions.
#
# Author: Deep Learning Course | Fall 2019
# Date Created: 2019-09-06
################################################################################

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
import torch.nn as nn

################################################################################

class VanillaRNN(nn.Module):

    def __init__(self, seq_length, input_dim, num_hidden, num_classes, device='cpu'):
        super(VanillaRNN, self).__init__()

        self.num_classes = num_classes
        self.device = device
        self.seq_length = seq_length
        self.num_hidden = num_hidden

        self.w_hx = nn.Parameter(data=torch.nn.init.xavier_uniform_(torch.empty(num_hidden, input_dim)), requires_grad=True)
        self.w_hh = nn.Parameter(data=torch.nn.init.xavier_uniform_(torch.empty(num_hidden, num_hidden)), requires_grad=True)
        self.w_ph = nn.Parameter(data=torch.nn.init.xavier_uniform_(torch.empty(num_classes, num_hidden)), requires_grad=True)
        self.b_h = nn.Parameter(data=torch.zeros(num_hidden), requires_grad=True)
        self.b_p = nn.Parameter(data=torch.zeros(num_classes), requires_grad=True)

        self.hidden_h = []

        super().to(device)

    def forward(self, x):
        # Implementation here ...
        prev_h = torch.autograd.Variable(torch.zeros(len(x), self.num_hidden), requires_grad=True).to(self.device)
        
        for i in range(self.seq_length):
            self.hidden_h.append(prev_h)
            h_t = torch.tanh(x[:,i].reshape(-1,1) @ self.w_hx.t() + prev_h @ self.w_hh.t() + self.b_h)
            prev_h = h_t
        
        self.hidden_h.append(prev_h)
        p_t = h_t @ self.w_ph.t() + self.b_p
            
        out = p_t

        return out


