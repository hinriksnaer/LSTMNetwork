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

import argparse
import time
from datetime import datetime
import numpy as np
# Added for vscode debug functionality
import multiprocessing

import torch
from torch.utils.data import DataLoader

from dataset import PalindromeDataset
from vanilla_rnn import VanillaRNN
from lstm import LSTM

"""part1."""
"""part1."""
"""part1."""

# You may want to look into tensorboard for logging
# from torch.utils.tensorboard import SummaryWriter

################################################################################

def train(config):

    assert config.model_type in ('RNN', 'LSTM')

    # Initialize the device which to run the model on
    device = torch.device(config.device)

    # Initialize the model that we are going to use
    model = None

    if config.model_type == 'LSTM':
        model = LSTM(
            config.input_length,
            config.input_dim,
            config.num_hidden,
            config.num_classes,
            config.device,
            )
    elif config.model_type == 'RNN':
        model = VanillaRNN(
            config.input_length,
            config.input_dim,
            config.num_hidden,
            config.num_classes,
            config.device,
            )
    else:
        print('Your model type input is neither \'RNN\' or \'LSTM\'')
        return

    # Initialize the dataset and data loader (note the +1)
    dataset = PalindromeDataset(config.input_length+1)
    data_loader = DataLoader(dataset, config.batch_size, num_workers=1)

    # Setup the loss and optimizer
    criterion = torch.nn.CrossEntropyLoss()  
    optimizer = torch.optim.RMSprop(model.parameters(), lr=config.learning_rate)


    # added for vscode debug functionality
    multiprocessing.set_start_method('spawn', True)

    for step, (batch_inputs, batch_targets) in enumerate(data_loader):

        batch_inputs = batch_inputs.to(device)
        batch_targets = batch_targets.to(device)

        # Only for time measurement of step through network
        t1 = time.time()

        # Add more code here ...
        optimizer.zero_grad()
        output = model.forward(batch_inputs)

        loss = criterion.forward(output, batch_targets)

        h_grads = model.hidden_h

        for grad in h_grads:
            grad.retain_grad()
        

        loss.backward()
        ############################################################################
        # QUESTION: what happens here and why?
        ############################################################################
        '''
        ANSWER:
        This function ‘clips’ the norm of the gradients by scaling the gradients down 
        by the same amount in order to reduce the norm to an acceptable level. In 
        practice this places a limit on the size of the parameter updates.
        '''
        torch.nn.utils.clip_grad_norm(model.parameters(), max_norm=config.max_norm)
        ############################################################################
        print(len(h_grads))
        for grad in h_grads:
            print(grad.grad.norm().item())
        break
 ################################################################################
 ################################################################################

if __name__ == "__main__":

    # Parse training configuration
    parser = argparse.ArgumentParser()

    # Model params
    parser.add_argument('--model_type', type=str, default="RNN", help="Model type, should be 'RNN' or 'LSTM'")
    parser.add_argument('--input_length', type=int, default=10, help='Length of an input sequence')
    parser.add_argument('--input_dim', type=int, default=1, help='Dimensionality of input sequence')
    parser.add_argument('--num_classes', type=int, default=10, help='Dimensionality of output sequence')
    parser.add_argument('--num_hidden', type=int, default=128, help='Number of hidden units in the model')
    parser.add_argument('--learning_rate', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--train_steps', type=int, default=10000, help='Number of training steps')
    parser.add_argument('--max_norm', type=float, default=10.0)
    parser.add_argument('--device', type=str, default="cuda:0", help="Training device 'cpu' or 'cuda:0'")

    config = parser.parse_args()
    config.batch_size = 1
    # Train the model
    train(config)