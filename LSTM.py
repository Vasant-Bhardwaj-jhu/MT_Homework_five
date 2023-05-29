#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import unicode_literals, print_function, division

from typing import Type

# import matplotlib
# #if you are running on the gradx/ugradx/ another cluster, 
# #you will need the following line
# #if you run on a local machine, you can comment it out
# matplotlib.use('agg') 
# import matplotlib.pyplot as plt
# import matplotlib.ticker as ticker
import torch
import torch.nn as nn
import time

class LSTM_gate(nn.Module):
    def __init__(self, *input_sizes : int, hidden_size : int, activation : Type[nn.Module] = nn.Sigmoid):
        super().__init__()

        self.num_of_inputs = len(input_sizes) + 1 # len(self.linears)
        self.range_over_inputs = range(self.num_of_inputs)

        self.linears = nn.ModuleList(
              [nn.Linear(input_size, hidden_size, bias=False) for input_size in input_sizes]
            + [nn.Linear(hidden_size, hidden_size, bias=False)]
        )

        self.bias = nn.Parameter(torch.randn(hidden_size))

        self.activate = activation()
    
    # Assume that input, hidden are 1-d tensors
    # Or if multi-dimensional, that the dimension to apply
    # model on is the last
    # Additionally assume the last tensor in input represents
    # the hidden state
    def forward(self, *input : torch.Tensor):
        return self.activate(
            #   sum(linear(input) for linear, input in zip(self.linears, input))
            sum(self.linears[i](input[i]) for i in self.range_over_inputs)
            + self.bias
        )

        # changing the iteration from a zip to a range over the length of
        # self.linears could also be attempted for performance




# def create_LSTM_gate(input_size : int, hidden_size : int, activation : Type = nn.Sigmoid):
#     concat_size = input_size + hidden_size
#     return nn.Sequential(
#         nn.Linear(concat_size, hidden_size),
#         activation(),
#     )

class LSTM(nn.Module):
    """the class for the LSTM
    """
    def __init__(self, *input_sizes : int, hidden_size : int):
        # super(LSTM, self).__init__() # Pre python-3.3 style
        super().__init__()
        self.hidden_size = hidden_size

        # total_input_size = sum(input_sizes)

        self.forget_gate = LSTM_gate(*input_sizes, hidden_size=hidden_size)
        self.input_gate  = LSTM_gate(*input_sizes, hidden_size=hidden_size)
        self.output_gate = LSTM_gate(*input_sizes, hidden_size=hidden_size)
        self.cell_input  = LSTM_gate(*input_sizes, hidden_size=hidden_size, activation=nn.Tanh)
    
    # For speed, allow hidden_h and hidden_c to be positional rather than keyword
    # by placing them before inputs
    def forward(self, hidden_h : torch.Tensor, hidden_c : torch.Tensor, *inputs : torch.Tensor):
        
        forget_act = self.forget_gate(*inputs, hidden_h)
        input_act  = self.input_gate(*inputs, hidden_h)
        output_act = self.output_gate(*inputs, hidden_h)
        cell_act   = self.cell_input(*inputs, hidden_h)


        out_c = forget_act * hidden_c + input_act * cell_act
        out_h = output_act * torch.tanh(out_c)
        return out_h, out_c
    
    def get_initial_states(self, device):
        return torch.zeros(self.hidden_size, device=device), torch.zeros(self.hidden_size, device=device)