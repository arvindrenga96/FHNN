#!/usr/bin/env python
# coding: utf-8

"""
This file is part of the accompanying code to our manuscript:
Kratzert, F., Klotz, D., Shalev, G., Klambauer, G., Hochreiter, S., Nearing, G., "Benchmarking
a Catchment-Aware Long Short-Term Memory Network (LSTM) for Large-Scale Hydrological Modeling".
submitted to Hydrol. Earth Syst. Sci. Discussions (2019)
You should have received a copy of the Apache-2.0 license along with the code. If not,
see <https://opensource.org/licenses/Apache-2.0>
"""

from typing import Tuple
import torch

class LSTM(torch.nn.Module):
    """Implementation of the standard LSTM.
    TODO: Include ref and LaTeX equations
    Parameters
    ----------
    input_size : int
        Number of input features
    hidden_size : int
        Number of hidden/memory cells.
    batch_first : bool, optional
        If True, expects the batch inputs to be of shape [batch, seq, features] otherwise, the
        shape has to be [seq, batch, features], by default True.
    initial_forget_bias : int, optional
        Value of the initial forget gate bias, by default 0
    """

    def __init__(self,
                 input_size: int,
                 hidden_size: int,
                 batch_first: bool = True,
                 initial_forget_bias: int = 0):
        super(LSTM, self).__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.batch_first = batch_first
        self.initial_forget_bias = initial_forget_bias

        # create tensors of learnable parameters
        self.weight_ih = torch.nn.Parameter(torch.FloatTensor(input_size, 4 * hidden_size))
        self.weight_hh = torch.nn.Parameter(torch.FloatTensor(hidden_size, 4 * hidden_size))
        self.bias_ih = torch.nn.Parameter(torch.FloatTensor(4 * hidden_size))
        self.bias_hh = torch.nn.Parameter(torch.FloatTensor(4 * hidden_size))

        # initialize parameters
        self.reset_parameters()

    def reset_parameters(self):
        """Initialize all learnable parameters of the LSTM"""
        torch.nn.init.orthogonal_(self.weight_ih.data)

        weight_hh_data = torch.eye(self.hidden_size)
        weight_hh_data = weight_hh_data.repeat(1, 4)
        self.weight_hh.data = weight_hh_data

        torch.nn.init.constant_(self.bias_ih.data, val=0)
        torch.nn.init.constant_(self.bias_hh.data, val=0)

        if self.initial_forget_bias != 0:
            self.bias_hh.data[:self.hidden_size] = self.initial_forget_bias

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """[summary]

        Parameters
        ----------
        x : torch.Tensor
            Tensor, containing a batch of input sequences. Format must match the specified format,
            defined by the batch_first agrument.
        Returns
        -------
        h_n : torch.Tensor
            The hidden states of each time step of each sample in the batch.
        c_n : torch.Tensor]
            The cell states of each time step of each sample in the batch.
        """
        if self.batch_first:
            x = x.transpose(0, 1)

        seq_len, batch_size, _ = x.size()

        h_0 = x.data.new(batch_size, self.hidden_size).zero_()
        c_0 = x.data.new(batch_size, self.hidden_size).zero_()
        h_x = (h_0, c_0)

        # empty lists to temporally store all intermediate hidden/cell states
        h_n, c_n = [], []

        # expand bias vectors to batch size
        bias_ih_batch = (self.bias_ih.unsqueeze(0).expand(batch_size, *self.bias_ih.size()))
        bias_hh_batch = (self.bias_hh.unsqueeze(0).expand(batch_size, *self.bias_hh.size()))

        # perform forward steps over input sequence
        for t in range(seq_len):
            h_0, c_0 = h_x

            # calculate gates
            gates_hh = torch.addmm(bias_hh_batch, h_0, self.weight_hh)
            gates_ih = torch.addmm(bias_ih_batch, x[t], self.weight_ih)
            gates = gates_hh + gates_ih
            f, i, o, g = gates.chunk(4, 1)

            c_1 = torch.sigmoid(f) * c_0 + torch.sigmoid(i) * torch.tanh(g)
            h_1 = torch.sigmoid(o) * torch.tanh(c_1)

            # store intermediate hidden/cell state in list
            h_n.append(h_1)
            c_n.append(c_1)

            h_x = (h_1, c_1)

        h_n = torch.stack(h_n, 0)
        c_n = torch.stack(c_n, 0)

        if self.batch_first:
            h_n = h_n.transpose(0, 1)
            c_n = c_n.transpose(0, 1)

        return h_n, c_n


class EALSTM(torch.nn.Module):
    """Implementation of the Entity-Aware-LSTM (EA-LSTM)
    TODO: Include paper ref and latex equations
    Parameters
    ----------
    input_size_dyn : int
        Number of dynamic features, which are those, passed to the LSTM at each time step.
    input_size_stat : int
        Number of static features, which are those that are used to modulate the input gate.
    hidden_size : int
        Number of hidden/memory cells.
    batch_first : bool, optional
        If True, expects the batch inputs to be of shape [batch, seq, features] otherwise, the
        shape has to be [seq, batch, features], by default True.
    initial_forget_bias : int, optional
        Value of the initial forget gate bias, by default 0
    """

    def __init__(self,
                 input_size_dyn: int,
                 input_size_stat: int,
                 hidden_size: int,
                 batch_first: bool = True,
                 initial_forget_bias: int = 0):
        super(EALSTM, self).__init__()

        self.input_size_dyn = input_size_dyn
        self.input_size_stat = input_size_stat
        self.hidden_size = hidden_size
        self.batch_first = batch_first
        self.initial_forget_bias = initial_forget_bias

        # create tensors of learnable parameters
        self.weight_ih = torch.nn.Parameter(torch.FloatTensor(input_size_dyn, 3 * hidden_size))
        self.weight_hh = torch.nn.Parameter(torch.FloatTensor(hidden_size, 3 * hidden_size))
        self.weight_sh = torch.nn.Parameter(torch.FloatTensor(input_size_stat, hidden_size))
        self.bias_ih = torch.nn.Parameter(torch.FloatTensor(3 * hidden_size))
        self.bias_hh = torch.nn.Parameter(torch.FloatTensor(3 * hidden_size))
        self.bias_sh = torch.nn.Parameter(torch.FloatTensor(hidden_size))

        # initialize parameters
        self.reset_parameters()

    def reset_parameters(self):
        """Initialize all learnable parameters of the LSTM"""
        torch.nn.init.orthogonal_(self.weight_ih.data)
        torch.nn.init.orthogonal_(self.weight_sh.data)

        weight_hh_data = torch.eye(self.hidden_size)
        weight_hh_data = weight_hh_data.repeat(1, 3)
        self.weight_hh.data = weight_hh_data

        torch.nn.init.constant_(self.bias_ih.data, val=0)
        torch.nn.init.constant_(self.bias_hh.data, val=0)
        torch.nn.init.constant_(self.bias_sh.data, val=0)

        if self.initial_forget_bias != 0:
            self.bias_hh.data[:self.hidden_size] = self.initial_forget_bias

    def forward(self, x_d: torch.Tensor, x_s: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """[summary]
        Parameters
        ----------
        x_d : torch.Tensor
            Tensor, containing a batch of sequences of the dynamic features. Shape has to match
            the format specified with batch_first.
        x_s : torch.Tensor
            Tensor, containing a batch of static features.
        Returns
        -------
        h_n : torch.Tensor
            The hidden states of each time step of each sample in the batch.
        c_n : torch.Tensor]
            The cell states of each time step of each sample in the batch.
        """
        if self.batch_first:
            x_d = x_d.transpose(0, 1)

        seq_len, batch_size, _ = x_d.size()

        h_0 = x_d.data.new(batch_size, self.hidden_size).zero_()
        c_0 = x_d.data.new(batch_size, self.hidden_size).zero_()
        h_x = (h_0, c_0)

        # empty lists to temporally store all intermediate hidden/cell states
        h_n, c_n = [], []

        # expand bias vectors to batch size
        bias_ih_batch = (self.bias_ih.unsqueeze(0).expand(batch_size, *self.bias_ih.size()))
        bias_hh_batch = (self.bias_hh.unsqueeze(0).expand(batch_size, *self.bias_hh.size()))

        # calculate input gate only once because inputs are static
        bias_sh_batch = (self.bias_sh.unsqueeze(0).expand(batch_size, *self.bias_sh.size()))
        i = torch.sigmoid(torch.addmm(bias_sh_batch, x_s, self.weight_sh))

        # perform forward steps over input sequence
        for t in range(seq_len):
            h_0, c_0 = h_x

            # calculate gates
            gates_hh = torch.addmm(bias_hh_batch, h_0, self.weight_hh)
            gates_ih = torch.addmm(bias_ih_batch, x_d[t], self.weight_ih)
            gates = gates_hh + gates_ih
            f, o, g = gates.chunk(3, 1)

            c_1 = torch.sigmoid(f) * c_0 + i * torch.tanh(g)
            h_1 = torch.sigmoid(o) * torch.tanh(c_1)

            # store intermediate hidden/cell state in list
            h_n.append(h_1)
            c_n.append(c_1)

            h_x = (h_1, c_1)

        h_n = torch.stack(h_n, 0)
        c_n = torch.stack(c_n, 0)

        if self.batch_first:
            h_n = h_n.transpose(0, 1)
            c_n = c_n.transpose(0, 1)

        return h_n, c_n


class DCLSTM(torch.nn.Module):
    """Implementation of the Deep and Cross LSTM (DC-LSTM)
    TODO: Include paper ref and latex equations
    Parameters
    ----------
    input_size_dyn : int
        Number of dynamic features, which are those, passed to the LSTM at each time step.
    input_size_stat : int
        Number of static features, which are those that are used to modulate the input gate.
    hidden_size : int
        Number of hidden/memory cells.
    batch_first : bool, optional
        If True, expects the batch inputs to be of shape [batch, seq, features] otherwise, the
        shape has to be [seq, batch, features], by default True.
    initial_forget_bias : int, optional
        Value of the initial forget gate bias, by default 0
    """

    def __init__(self,
                 input_size_dyn: int,
                 input_size_stat: int,
                 hidden_size: int,
                 batch_first: bool = True,
                 initial_forget_bias: int = 0):
        super(DCLSTM, self).__init__()

        self.input_size_dyn = input_size_dyn
        self.input_size_stat = input_size_stat
        self.hidden_size = hidden_size
        self.batch_first = batch_first
        self.initial_forget_bias = initial_forget_bias

        # create tensors of learnable parameters
        self.weight_ih = torch.nn.Parameter(torch.FloatTensor(input_size_dyn, 4 * hidden_size))
        self.weight_hh = torch.nn.Parameter(torch.FloatTensor(hidden_size, 4 * hidden_size))
        self.bias_ih = torch.nn.Parameter(torch.FloatTensor(4 * hidden_size))
        self.bias_hh = torch.nn.Parameter(torch.FloatTensor(4 * hidden_size))

        # initialize parameters
        self.reset_parameters()

    def reset_parameters(self):
        """Initialize all learnable parameters of the LSTM"""
        torch.nn.init.orthogonal_(self.weight_ih.data)

        weight_hh_data = torch.eye(self.hidden_size)
        weight_hh_data = weight_hh_data.repeat(1, 4)
        self.weight_hh.data = weight_hh_data

        torch.nn.init.constant_(self.bias_ih.data, val=0)
        torch.nn.init.constant_(self.bias_hh.data, val=0)

        if self.initial_forget_bias != 0:
            self.bias_hh.data[:self.hidden_size] = self.initial_forget_bias

    def forward(self, x_d: torch.Tensor, x_s: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """[summary]

        Parameters
        ----------
        x : torch.Tensor
            Tensor, containing a batch of input sequences. Format must match the specified format,
            defined by the batch_first agrument.

        Returns
        -------
        h_n : torch.Tensor
            The hidden states of each time step of each sample in the batch.
        c_n : torch.Tensor]
            The cell states of each time step of each sample in the batch.
        """

        if self.batch_first:
            x_d = x_d.transpose(0, 1)

        seq_len, batch_size, _ = x_d.size()

        h_0 = x_d.data.new(batch_size, self.hidden_size).zero_()
        c_0 = x_d.data.new(batch_size, self.hidden_size).zero_()
        h_x = (h_0, c_0)

        # empty lists to temporally store all intermediate hidden/cell states
        h_n, c_n = [], []

        # expand bias vectors to batch size
        bias_ih_batch = (self.bias_ih.unsqueeze(0).expand(batch_size, *self.bias_ih.size()))
        bias_hh_batch = (self.bias_hh.unsqueeze(0).expand(batch_size, *self.bias_hh.size()))

        # calculate static embedding
        x_s = torch.repeat_interleave(x_s, 4, dim=1)

        # perform forward steps over input sequence
        for t in range(seq_len):
            h_0, c_0 = h_x

            # calculate gates
            gates_hh = x_s * torch.addmm(bias_hh_batch, h_0, self.weight_hh) + torch.repeat_interleave(h_0, 4, dim=1)
            gates_ih = x_s * torch.addmm(bias_ih_batch, x_d[t], self.weight_ih) + torch.repeat_interleave(x_d[t], 4, dim=1)
            gates = gates_hh + gates_ih
            f, i, o, g = gates.chunk(4, 1)
            
            c_1 = torch.sigmoid(f) * c_0 + torch.sigmoid(i) * torch.tanh(g)
            h_1 = torch.sigmoid(o) * torch.tanh(c_1)
            
            # store intermediate hidden/cell state in list
            h_n.append(h_1)
            c_n.append(c_1)

            h_x = (h_1, c_1)

        h_n = torch.stack(h_n, 0)
        c_n = torch.stack(c_n, 0)
        
        if self.batch_first:
            h_n = h_n.transpose(0, 1)
            c_n = c_n.transpose(0, 1)

        return h_n, c_n

    
class TAMLSTM(torch.nn.Module):
    """Implementation of the standard LSTM.
    TODO: Include ref and LaTeX equations
    Parameters
    ----------
    input_size : int
        Number of input features
    latent_size : int
        Number of latent features
    hidden_size : int
        Number of hidden/memory cells.
    batch_first : bool, optional
        If True, expects the batch inputs to be of shape [batch, seq, features] otherwise, the
        shape has to be [seq, batch, features], by default True.
    initial_forget_bias : int, optional
        Value of the initial forget gate bias, by default 0
    """

    def __init__(self,
                 input_size: int,
                 latent_size: int,
                 hidden_size: int,
                 batch_first: bool = True,
                 initial_forget_bias: int = 0):
        super(TAMLSTM, self).__init__()

        self.input_size = input_size
        self.latent_size = latent_size
        self.hidden_size = hidden_size
        self.batch_first = batch_first
        self.initial_forget_bias = initial_forget_bias
        
        self._embeddings = torch.nn.ModuleList()
        for dim in range(4):
            self._embeddings.append(torch.nn.Linear(self.latent_size, 2*self.hidden_size))

        # create tensors of learnable parameters
        self.weight_ih = torch.nn.Parameter(torch.FloatTensor(input_size, 4 * hidden_size))
        self.weight_hh = torch.nn.Parameter(torch.FloatTensor(hidden_size, 4 * hidden_size))
        self.bias_ih = torch.nn.Parameter(torch.FloatTensor(4 * hidden_size))
        self.bias_hh = torch.nn.Parameter(torch.FloatTensor(4 * hidden_size))

        # initialize parameters
        self.reset_parameters()

    def reset_parameters(self):
        """Initialize all learnable parameters of the LSTM"""
        torch.nn.init.orthogonal_(self.weight_ih.data)

        weight_hh_data = torch.eye(self.hidden_size)
        weight_hh_data = weight_hh_data.repeat(1, 4)
        self.weight_hh.data = weight_hh_data

        torch.nn.init.constant_(self.bias_ih.data, val=0)
        torch.nn.init.constant_(self.bias_hh.data, val=0)

        if self.initial_forget_bias != 0:
            self.bias_hh.data[:self.hidden_size] = self.initial_forget_bias
            
    def conditional_layer(self, f, i, o, g, embedding):
        
        gate_list = [f,i,o,g]
        for i,x in enumerate(gate_list):
            gammas, betas = torch.split(embedding[i], x.size(1), dim=-1)
            gammas = gammas + torch.ones_like(gammas)
            x = x * gammas + betas
            gate_list[i] = x
        return gate_list

    def forward(self, x: torch.Tensor, x_latent: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor,torch.Tensor]:
        """[summary]

        Parameters
        ----------
        x : torch.Tensor
            Tensor, containing a batch of input sequences. Format must match the specified format,
            defined by the batch_first agrument.
        Returns
        -------
        h_n : torch.Tensor
            The hidden states of each time step of each sample in the batch.
        c_n : torch.Tensor]
            The cell states of each time step of each sample in the batch.
        """
        if self.batch_first:
            x = x.transpose(0, 1)
            x_latent = x_latent.transpose(0, 1)
        
        seq_len, batch_size, _ = x.size()
        

        h_0 = x.data.new(batch_size, self.hidden_size).zero_()
        c_0 = x.data.new(batch_size, self.hidden_size).zero_()
        h_x = (h_0, c_0)

        # empty lists to temporally store all intermediate hidden/cell states
        h_n, c_n = [], []

        # expand bias vectors to batch size
        bias_ih_batch = (self.bias_ih.unsqueeze(0).expand(batch_size, *self.bias_ih.size()))
        bias_hh_batch = (self.bias_hh.unsqueeze(0).expand(batch_size, *self.bias_hh.size()))

        # perform forward steps over input sequence
        for t in range(seq_len):
            h_0, c_0 = h_x

            # calculate gates
            gates_hh = torch.addmm(bias_hh_batch, h_0, self.weight_hh)
            gates_ih = torch.addmm(bias_ih_batch, x[t], self.weight_ih)
            gates = gates_hh + gates_ih
            
            
            f, i, o, g = gates.chunk(4, 1)
            
            out_embeddings = []
            for embedding in self._embeddings:
                out_embeddings.append(embedding(x_latent[t]))
            
            f, i, o, g = self.conditional_layer(f, i, o, g, out_embeddings) 
            

            c_1 = torch.sigmoid(f) * c_0 + torch.sigmoid(i) * torch.tanh(g)
            h_1 = torch.sigmoid(o) * torch.tanh(c_1)

            # store intermediate hidden/cell state in list
            h_n.append(h_1)
            c_n.append(c_1)

            h_x = (h_1, c_1)

        h_n = torch.stack(h_n, 0)
        c_n = torch.stack(c_n, 0)

        if self.batch_first:
            h_n = h_n.transpose(0, 1)
            c_n = c_n.transpose(0, 1)

        return h_n, c_n