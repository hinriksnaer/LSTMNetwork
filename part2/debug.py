# MIT License
#
# Copyright (c) 2019 Tom Runia
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

import os
import time
from datetime import datetime
import argparse

import numpy as np

import torch
import torch.optim as optim
from torch.utils.data import DataLoader

from dataset import TextDataset
from model import TextGenerationModel

import multiprocessing
################################################################################

def train(config):

    if torch.cuda.is_available():  
        dev = "cuda:0" 
    else:  
        dev = "cpu"  
    # Initialize the device which to run the model on
    device = torch.device(dev) 


    # Initialize the dataset and data loader (note the +1)
    dataset = TextDataset(config.txt_file, config.seq_length)  # fixme
    data_loader = DataLoader(dataset, config.batch_size, num_workers=1)
    # Initialize the model that we are going to use
    model = TextGenerationModel(
        config.batch_size, 
        config.seq_length, 
        dataset.vocab_size,
        config.lstm_num_hidden,
        config.lstm_num_layers,
        device
        ).to(device) # fixme

    # Setup the loss and optimizer
    criterion = torch.nn.CrossEntropyLoss()  # fixme
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate, weight_decay = config.learning_rate_decay)  # fixme

    # added for vscode debug functionality
    multiprocessing.set_start_method('spawn', True)

    total_steps = 0

    while config.train_steps > total_steps:

        for step, (batch_inputs, batch_targets) in enumerate(data_loader):

            total_steps += 1

            if total_steps > config.train_steps: break

            batch_inputs = batch_inputs.to(device)
            batch_targets = batch_targets.to(device)
            # Only for time measurement of step through network
            t1 = time.time()

            #######################################################
            # Add more code here ...
            #######################################################
            batch_inputs = torch.nn.functional.one_hot(batch_inputs, dataset.vocab_size)
            optimizer.zero_grad()
            output = model.forward(batch_inputs)


            loss = 0.0
            for i in range(len(output[0])):
                pred = output[:,i,:]
                target = batch_targets[:,i]
                loss += criterion.forward(pred, target)/len(output[0])
                

            loss.backward()

            optimizer.step()
            with torch.no_grad():
                accuracy = 0.0  

                total_size = 0
                correct = 0

                for i in range(len(output[0])):
                    pred = torch.nn.functional.softmax(output[:,i,:], dim=1)
                    pred = torch.max(pred, 1)[1]

                    correct += pred.eq(batch_targets[:,i]).sum().item()
                    total_size += len(pred)

                    
                    
                accuracy = correct/total_size

                # Just for time measurement
                t2 = time.time()
                examples_per_second = config.batch_size/float(t2-t1)

                if total_steps % config.print_every == 0:
                    
                    print("[{}] Train Step {:04d}/{:04d}, Batch Size = {}, Examples/Sec = {:.2f}, "
                        "Accuracy = {:.2f}, Loss = {:.3f}".format(
                            datetime.now().strftime("%Y-%m-%d %H:%M"), total_steps,
                            int(config.train_steps), config.batch_size, examples_per_second,
                            accuracy, loss
                    ))
                    

                if total_steps % config.sample_every == -1232342342152345236526:
                    # Generate some sentences by sampling from the model
                    text = torch.zeros((1,1)).long().random_(0, dataset.vocab_size).to(device)
                    text = torch.nn.functional.one_hot(text, dataset.vocab_size)
                    for i in range(config.seq_length - 1):
                        prediction = model.forward(text)
                        pred = torch.nn.functional.softmax(prediction[:,i,:], dim=1)
                        pred = torch.max(pred, 1)[1]
                        pred = torch.nn.functional.one_hot(pred, dataset.vocab_size)
                        pred = pred.unsqueeze(0)
                        text = torch.cat((text, pred), 1)
                        stuff = torch.argmax(text[0], 1)
                        sentence = dataset.convert_to_string(stuff.tolist())
                    print(sentence)

                if total_steps == config.train_steps:
                    # If you receive a PyTorch data-loader error, check this bug report:
                    # https://github.com/pytorch/pytorch/pull/9655
                    break

    print('Done training.')

    
 ################################################################################
 ################################################################################

if __name__ == "__main__":

    # Parse training configuration
    parser = argparse.ArgumentParser()

    # Model params
    parser.add_argument('--txt_file', type=str, required=True, help="Path to a .txt file to train on")
    parser.add_argument('--seq_length', type=int, default=30, help='Length of an input sequence')
    parser.add_argument('--lstm_num_hidden', type=int, default=128, help='Number of hidden units in the LSTM')
    parser.add_argument('--lstm_num_layers', type=int, default=2, help='Number of LSTM layers in the model')

    # Training params
    parser.add_argument('--batch_size', type=int, default=64, help='Number of examples to process in a batch')
    parser.add_argument('--learning_rate', type=float, default=2e-3, help='Learning rate')

    # It is not necessary to implement the following three params, but it may help training.
    parser.add_argument('--learning_rate_decay', type=float, default=0.96, help='Learning rate decay fraction')
    parser.add_argument('--learning_rate_step', type=int, default=5000, help='Learning rate step')
    parser.add_argument('--dropout_keep_prob', type=float, default=1.0, help='Dropout keep probability')

    parser.add_argument('--train_steps', type=int, default=1e6, help='Number of training steps')
    parser.add_argument('--max_norm', type=float, default=5.0, help='--')

    # Misc params
    parser.add_argument('--summary_path', type=str, default="./summaries/", help='Output path for summaries')
    parser.add_argument('--print_every', type=int, default=5, help='How often to print training progress')
    parser.add_argument('--sample_every', type=int, default=100, help='How often to sample from the model')

    config = parser.parse_args()

    # Train the model
    train(config)
