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

class LSTM(nn.Module):

    def __init__(self, seq_length, input_dim, num_hidden, num_classes, device='cpu'):
        super(LSTM, self).__init__()
        # Initialization here ...
        self.seq_length = seq_length
        self.input_dim = input_dim
        self.num_hidden = num_hidden
        self.device = device

        self.w_gx = torch.nn.Parameter(data=torch.nn.init.xavier_uniform_(torch.empty(num_hidden, input_dim)), requires_grad=True)
        self.w_ix = torch.nn.Parameter(data=torch.nn.init.xavier_uniform_(torch.empty(num_hidden, input_dim)), requires_grad=True)
        self.w_fx = torch.nn.Parameter(data=torch.nn.init.xavier_uniform_(torch.empty(num_hidden, input_dim)), requires_grad=True)
        self.w_ox = torch.nn.Parameter(data=torch.nn.init.xavier_uniform_(torch.empty(num_hidden, input_dim)), requires_grad=True)
        self.w_gh = torch.nn.Parameter(data=torch.nn.init.xavier_uniform_(torch.empty(num_hidden, num_hidden)), requires_grad=True)
        self.w_ih = torch.nn.Parameter(data=torch.nn.init.xavier_uniform_(torch.empty(num_hidden, num_hidden)), requires_grad=True)
        self.w_fh = torch.nn.Parameter(data=torch.nn.init.xavier_uniform_(torch.empty(num_hidden, num_hidden)), requires_grad=True)
        self.w_oh = torch.nn.Parameter(data=torch.nn.init.xavier_uniform_(torch.empty(num_hidden, num_hidden)), requires_grad=True)
        self.w_ph = torch.nn.Parameter(data=torch.nn.init.xavier_uniform_(torch.empty(num_classes, num_hidden)), requires_grad=True)
        self.b_g = torch.nn.Parameter(data=torch.zeros(num_hidden), requires_grad=True)
        self.b_i = torch.nn.Parameter(data=torch.zeros(num_hidden), requires_grad=True)
        self.b_f = torch.nn.Parameter(data=torch.ones(num_hidden), requires_grad=True)
        self.b_o = torch.nn.Parameter(data=torch.zeros(num_hidden), requires_grad=True)
        self.b_p = torch.nn.Parameter(data=torch.zeros(num_classes), requires_grad=True)

        self.hidden_h = []

        super().to(device)

    def forward(self, x):
        # Implementation here ...
        prev_h = torch.autograd.Variable(torch.zeros(len(x), self.num_hidden), requires_grad = True).to(self.device)
        prev_c = torch.zeros(len(x), self.num_hidden).to(self.device)

        for i in range(self.seq_length):

            self.hidden_h.append(prev_h)

            g_t = torch.tanh(x[:,i].reshape(-1,1) @ self.w_gx.t() + prev_h @ self.w_gh.t() + self.b_g)
            i_t = torch.sigmoid(x[:,i].reshape(-1,1) @ self.w_ix.t() + prev_h @ self.w_ih.t() + self.b_i)
            f_t = torch.sigmoid(x[:,i].reshape(-1,1) @ self.w_fx.t() + prev_h @ self.w_fh.t() + self.b_f)
            o_t = torch.sigmoid(x[:,i].reshape(-1,1) @ self.w_ox.t() + prev_h @ self.w_oh.t() + self.b_o)
            c_t = g_t * i_t + prev_c * f_t
            h_t = torch.tanh(c_t) * o_t

            p_t = h_t @ self.w_ph.t() + self.b_p

            prev_c = c_t
            prev_h = h_t
        
        self.hidden_h.append(prev_h)
        out = p_t

        return out