#!/usr/bin/env python
# coding: utf-8

# # IMPORT LIBRARIES

# In[1]:


import numpy as np
import torch


# # DECODER MODELS

# ## LSTM MODELS

# In[2]:


class lstm(torch.nn.Module):

    def __init__(self, input_channels, latent_code_dim, output_channels, dropout):
        super(lstm,self).__init__()

        # PARAMETERS
        self.input_channels = input_channels
        self.latent_code_dim = latent_code_dim
        self.output_channels = output_channels

        # LAYERS
        self.encoder = torch.nn.LSTM(input_size=self.input_channels, hidden_size=self.latent_code_dim, batch_first=True)
        self.out = torch.nn.Linear(in_features=self.latent_code_dim, out_features=self.output_channels)
        self.dropout = torch.nn.Dropout(p=dropout)

        # INITIALIZATION
        for m in self.modules():
            if isinstance(m, torch.nn.Conv2d) or isinstance(m, torch.nn.Linear):
                torch.nn.init.xavier_uniform_(m.weight)

    def forward(self, x_dynamic, forecast):

        # GET SHAPES
        batch, window, _ = x_dynamic.shape

        # OPERATIONS
        x_encoder, _ = self.encoder(x_dynamic)
        x_encoder = self.dropout(x_encoder)
        out = self.out(x_encoder)
        out = out[:, -forecast:]
        out = out.view(batch, forecast, self.output_channels)

        return out


# In[3]:


class lstm_ar(torch.nn.Module):

    def __init__(self, input_channels, forward_code_dim, output_channels, dropout):
        super(lstm_ar,self).__init__()

        # PARAMETERS
        self.input_channels = input_channels
        self.forward_code_dim = forward_code_dim
        self.output_channels = output_channels

        # LAYERS
        self.encoder = torch.nn.LSTM(input_size=self.input_channels+self.output_channels, hidden_size=self.forward_code_dim, batch_first=True)
        self.out = torch.nn.Linear(in_features=self.forward_code_dim, out_features=self.output_channels)
        self.dropout = torch.nn.Dropout(p=dropout)

        # INITIALIZATION
        for m in self.modules():
            if isinstance(m, torch.nn.Conv2d) or isinstance(m, torch.nn.Linear):
                torch.nn.init.xavier_uniform_(m.weight)

    def forward(self, x_dynamic, y, forecast):

        # GET SHAPES
        batch, window, _ = x_dynamic.shape

        # OPERATIONS
        x = torch.cat((x_dynamic, y), dim=-1)
        x_encoder, _ = self.encoder(x)
        x_encoder = self.dropout(x_encoder)
        out = self.out(x_encoder)
        out = out[:, -forecast:]
        out = out.view(batch, forecast, self.output_channels)

        return out


# # ENCODER MODELS

# ## LSTM BIDIRECTIONAL MODELS

# In[4]:


class lstm_bidirectional(torch.nn.Module):

    def __init__(self, input_channels, latent_code_dim, output_channels, dropout):
        super(lstm_bidirectional,self).__init__()

        # PARAMETERS
        self.input_channels = input_channels
        self.latent_code_dim = latent_code_dim
        self.output_channels = output_channels

        # LAYERS
        self.encoder = torch.nn.LSTM(input_size=self.input_channels, hidden_size=self.latent_code_dim, bidirectional=True, batch_first=True)
        self.out_encoder = torch.nn.Linear(in_features=self.latent_code_dim, out_features=self.output_channels)
        self.dropout = torch.nn.Dropout(p=dropout)

        # INITIALIZATION
        for m in self.modules():
            if isinstance(m, torch.nn.Conv2d) or isinstance(m, torch.nn.Linear):
                torch.nn.init.xavier_uniform_(m.weight)

    def forward(self, x_dynamic, forecast=1):

        # GET SHAPES
        batch, window, _ = x_dynamic.shape

        # OPERATIONS
        _, (x_encoder, _) = self.encoder(x_dynamic)
        x_encoder = torch.sum(x_encoder, axis=0)
        x_encoder = self.dropout(x_encoder)
        out = self.out_encoder(x_encoder)
        out = out.view(batch, forecast, self.output_channels)

        return out


# ## Multiple Bidertional LSTM

# In[5]:


class lstm_mul_enc(torch.nn.Module):

    def __init__(self, input_channels, latent_code_dim, output_channels, dropout):
        super(lstm_mul_enc,self).__init__()

        # PARAMETERS
        self.input_channels = input_channels
        self.latent_code_dim = latent_code_dim
        self.output_channels = output_channels

        # LAYERS
        self.encoder_daily = torch.nn.LSTM(input_size=self.input_channels, hidden_size=self.latent_code_dim, bidirectional=True, batch_first=True)
        self.encoder_biweekly = torch.nn.LSTM(input_size=self.input_channels, hidden_size=self.latent_code_dim, bidirectional=True, batch_first=True)
        self.encoder_seasonally = torch.nn.LSTM(input_size=self.input_channels, hidden_size=self.latent_code_dim, bidirectional=True, batch_first=True)
        self.out_encoder = torch.nn.Linear(in_features=self.latent_code_dim, out_features=self.output_channels)
        self.dropout = torch.nn.Dropout(p=dropout)

        # INITIALIZATION
        for m in self.modules():
            if isinstance(m, torch.nn.Conv2d) or isinstance(m, torch.nn.Linear):
                torch.nn.init.xavier_uniform_(m.weight)

    def forward(self, x_dynamic, forecast=1):

        # GET SHAPES
        batch, window, _ = x_dynamic.shape

        # OPERATIONS
        x_encoder = x_dynamic[:, :-forecast]
        x_encoder, (h_daily, c_daily) = self.encoder_daily(x_encoder)
        x_encoder = x_encoder[:, :, :self.latent_code_dim] + x_encoder[:, :, self.latent_code_dim:]
        h_daily, c_daily = torch.sum(h_daily, axis=0), torch.sum(c_daily, axis=0)

        x_encoder = x_dynamic[:, :-forecast]
        x_encoder, (h_biweekly, c_biweekly) = self.encoder_biweekly(x_encoder[:, -14:])
        x_encoder = x_encoder[:, :, :self.latent_code_dim] + x_encoder[:, :, self.latent_code_dim:]
        h_biweekly, c_biweekly = torch.sum(h_biweekly, axis=0), torch.sum(c_biweekly, axis=0)

        x_encoder = x_dynamic[:, :-forecast]
        x_encoder, (h_seasonally, c_seasonally) = self.encoder_seasonally(x_encoder[:, -(12*7):])
        x_encoder = x_encoder[:, :, :self.latent_code_dim] + x_encoder[:, :, self.latent_code_dim:]
        h_seasonally, c_seasonally = torch.sum(h_seasonally, axis=0), torch.sum(c_seasonally, axis=0)

        h = h_daily+h_biweekly+h_seasonally
        c = c_daily+c_biweekly+c_seasonally
        
        x_encoder = self.dropout(h)
        out_encoder = self.out_encoder(x_encoder)
        out_encoder = out_encoder.view(batch, forecast, self.output_channels)


        return out_encoder


# ## Multiple hierarchical Bidertional LSTM

# In[6]:


class lstm_hierarchical_enc(torch.nn.Module):

    def __init__(self, input_channels, latent_code_dim, output_channels, dropout):
        super(lstm_hierarchical_enc,self).__init__()

        # PARAMETERS
        self.input_channels = input_channels
        self.latent_code_dim = latent_code_dim
        self.output_channels = output_channels

        # LAYERS
        self.encoder_daily = torch.nn.LSTM(input_size=self.input_channels, hidden_size=self.latent_code_dim, bidirectional=True, batch_first=True)
        self.encoder_biweekly = torch.nn.LSTM(input_size=self.latent_code_dim, hidden_size=self.latent_code_dim, bidirectional=True, batch_first=True)
        self.encoder_seasonally = torch.nn.LSTM(input_size=self.latent_code_dim, hidden_size=self.latent_code_dim, bidirectional=True, batch_first=True)
        self.out_encoder = torch.nn.Linear(in_features=self.latent_code_dim, out_features=self.output_channels)
        self.dropout = torch.nn.Dropout(p=dropout)

        # INITIALIZATION
        for m in self.modules():
            if isinstance(m, torch.nn.Conv2d) or isinstance(m, torch.nn.Linear):
                torch.nn.init.xavier_uniform_(m.weight)

    def forward(self, x_dynamic, forecast=1):

        # GET SHAPES
        batch, window, _ = x_dynamic.shape

        # OPERATIONS
        x_encoder = x_dynamic[:, :-forecast]
        x_encoder, (h_daily, c_daily) = self.encoder_daily(x_encoder)
        x_encoder = x_encoder[:, :, :self.latent_code_dim] + x_encoder[:, :, self.latent_code_dim:]
        h_daily, c_daily = torch.sum(h_daily, axis=0), torch.sum(c_daily, axis=0)
        

        x_encoder, (h_biweekly, c_biweekly) = self.encoder_biweekly(x_encoder.flip(1)[:, ::14].flip(1))
        x_encoder = x_encoder[:, :, :self.latent_code_dim] + x_encoder[:, :, self.latent_code_dim:]
        h_biweekly, c_biweekly = torch.sum(h_biweekly, axis=0), torch.sum(c_biweekly, axis=0)

        x_encoder, (h_seasonally, c_seasonally) = self.encoder_seasonally(x_encoder.flip(1)[:, ::6].flip(1))
        x_encoder = x_encoder[:, :, :self.latent_code_dim] + x_encoder[:, :, self.latent_code_dim:]
        h_seasonally, c_seasonally = torch.sum(h_seasonally, axis=0), torch.sum(c_seasonally, axis=0)

        h = h_daily+h_biweekly+h_seasonally
        c = c_daily+c_biweekly+c_seasonally
        
        x_encoder = self.dropout(h)
        out_encoder = self.out_encoder(x_encoder)
        out_encoder = out_encoder.view(batch, forecast, self.output_channels)


        return out_encoder


# In[7]:


# architecture = "lstm_hierarchical_enc"
# model = globals()[architecture](input_channels=len(dynamic_channels), latent_code_dim=latent_code_dim, output_channels=len(output_channels), dropout=dropout)
# model = model.to(device)
# pytorch_total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
# out = model(x_dynamic=x_dynamic, forecast=1)
# print(x_dynamic.shape, out.shape, "#:{}".format(pytorch_total_params), architecture)


# # ENCODER-DECODER MODELS

# ## LSTM ENC-DEC MODELS

# In[8]:


class lstm_enc_dec(torch.nn.Module):

    def __init__(self, input_channels, latent_code_dim, forward_code_dim, output_channels, dropout):
        super(lstm_enc_dec,self).__init__()

        # PARAMETERS
        self.input_channels = input_channels
        self.latent_code_dim = latent_code_dim
        self.forward_code_dim = forward_code_dim
        self.output_channels = output_channels

        # LAYERS
        self.encoder = torch.nn.LSTM(input_size=self.input_channels, hidden_size=self.latent_code_dim, bidirectional=True, batch_first=True)
        self.out_encoder = torch.nn.Linear(in_features=self.forward_code_dim, out_features=self.output_channels)
        self.decoder = torch.nn.LSTM(input_size=self.input_channels, hidden_size=self.forward_code_dim, batch_first=True)
        self.out_decoder = torch.nn.Linear(in_features=self.forward_code_dim, out_features=self.output_channels)
        self.dropout = torch.nn.Dropout(p=dropout)

        # INITIALIZATION
        for m in self.modules():
            if isinstance(m, torch.nn.Conv2d) or isinstance(m, torch.nn.Linear):
                torch.nn.init.xavier_uniform_(m.weight)

    def forward(self, x_dynamic, forecast):

        # GET SHAPES
        batch, window, _ = x_dynamic.shape

        # OPERATIONS
        x_encoder = x_dynamic[:, :-forecast]
        _, (h, c) = self.encoder(x_encoder)
        h, c = torch.sum(h, axis=0), torch.sum(c, axis=0)
        x_encoder = self.dropout(h)
        out_encoder = self.out_encoder(x_encoder)
        out_encoder = out_encoder.view(batch, 1, self.output_channels)

        x_decoder, _ = self.decoder(x_dynamic[:, -forecast:], (torch.unsqueeze(h, axis=0),torch.unsqueeze(c, axis=0)))
        x_decoder = self.dropout(x_decoder)
        out_decoder = self.out_decoder(x_decoder)
        out_decoder = out_decoder.view(batch, forecast, self.output_channels)

        return out_encoder, out_decoder


# In[9]:


class lstm_enc_with_label_dec(torch.nn.Module):
#Nearing equivalent FHNN paper
    def __init__(self, input_channels, latent_code_dim, forward_code_dim, output_channels, dropout):
        super(lstm_enc_with_label_dec,self).__init__()

        # PARAMETERS
        self.input_channels = input_channels
        self.latent_code_dim = latent_code_dim
        self.forward_code_dim = forward_code_dim
        self.output_channels = output_channels

        # LAYERS
        self.encoder = torch.nn.LSTM(input_size=self.input_channels+self.output_channels, hidden_size=self.latent_code_dim, bidirectional=True, batch_first=True)
        self.out_encoder = torch.nn.Linear(in_features=self.forward_code_dim, out_features=self.output_channels)
        self.decoder = torch.nn.LSTM(input_size=self.input_channels, hidden_size=self.forward_code_dim, batch_first=True)
        self.out_decoder = torch.nn.Linear(in_features=self.forward_code_dim, out_features=self.output_channels)
        self.dropout = torch.nn.Dropout(p=dropout)

        # INITIALIZATION
        for m in self.modules():
            if isinstance(m, torch.nn.Conv2d) or isinstance(m, torch.nn.Linear):
                torch.nn.init.xavier_uniform_(m.weight)

    def forward(self, x_dynamic, y,forecast):

        # GET SHAPES
        batch, window, _ = x_dynamic.shape

        # OPERATIONS
        x_encoder = torch.cat((x_dynamic[:, :-forecast], y[:, :-forecast]), dim=-1)
        
        _, (h, c) = self.encoder(x_encoder)
        h, c = torch.sum(h, axis=0), torch.sum(c, axis=0)
        x_encoder = self.dropout(h)
        out_encoder = self.out_encoder(x_encoder)
        out_encoder = out_encoder.view(batch, 1, self.output_channels)

        x_decoder, _ = self.decoder(x_dynamic[:, -forecast:], (torch.unsqueeze(h, axis=0),torch.unsqueeze(c, axis=0)))
        x_decoder = self.dropout(x_decoder)
        out_decoder = self.out_decoder(x_decoder)
        out_decoder = out_decoder.view(batch, forecast, self.output_channels)

        return out_encoder, out_decoder


# ## LSTM MULTIPLE ENC MODELS

# In[10]:


class lstm_mul_enc_dec(torch.nn.Module):

    def __init__(self, input_channels, latent_code_dim, forward_code_dim, output_channels, dropout):
        super(lstm_mul_enc_dec,self).__init__()

        # PARAMETERS
        self.input_channels = input_channels
        self.latent_code_dim = latent_code_dim
        self.forward_code_dim = forward_code_dim
        self.output_channels = output_channels

        # LAYERS
        self.encoder_daily = torch.nn.LSTM(input_size=self.input_channels, hidden_size=self.latent_code_dim, bidirectional=True, batch_first=True)
        self.encoder_biweekly = torch.nn.LSTM(input_size=self.input_channels, hidden_size=self.latent_code_dim, bidirectional=True, batch_first=True)
        self.encoder_seasonally = torch.nn.LSTM(input_size=self.input_channels, hidden_size=self.latent_code_dim, bidirectional=True, batch_first=True)
        self.out_encoder = torch.nn.Linear(in_features=self.forward_code_dim, out_features=self.output_channels)
        self.decoder = torch.nn.LSTM(input_size=self.input_channels, hidden_size=self.forward_code_dim, batch_first=True)
        self.out_decoder = torch.nn.Linear(in_features=self.forward_code_dim, out_features=self.output_channels)
        self.dropout = torch.nn.Dropout(p=dropout)

        # INITIALIZATION
        for m in self.modules():
            if isinstance(m, torch.nn.Conv2d) or isinstance(m, torch.nn.Linear):
                torch.nn.init.xavier_uniform_(m.weight)

    def forward(self, x_dynamic, forecast):

        # GET SHAPES
        batch, window, _ = x_dynamic.shape

        # OPERATIONS
        x_encoder = x_dynamic[:, :-forecast]
        x_encoder, (h_daily, c_daily) = self.encoder_daily(x_encoder)
        x_encoder = x_encoder[:, :, :self.latent_code_dim] + x_encoder[:, :, self.latent_code_dim:]
        h_daily, c_daily = torch.sum(h_daily, axis=0), torch.sum(c_daily, axis=0)

        x_encoder = x_dynamic[:, :-forecast]
        # print(x_encoder.shape)
        x_encoder, (h_biweekly, c_biweekly) = self.encoder_biweekly(x_encoder[:, -14:])
        x_encoder = x_encoder[:, :, :self.latent_code_dim] + x_encoder[:, :, self.latent_code_dim:]
        h_biweekly, c_biweekly = torch.sum(h_biweekly, axis=0), torch.sum(c_biweekly, axis=0)

        x_encoder = x_dynamic[:, :-forecast]
        x_encoder, (h_seasonally, c_seasonally) = self.encoder_seasonally(x_encoder[:, -(12*7):])
        x_encoder = x_encoder[:, :, :self.latent_code_dim] + x_encoder[:, :, self.latent_code_dim:]
        h_seasonally, c_seasonally = torch.sum(h_seasonally, axis=0), torch.sum(c_seasonally, axis=0)

        x_encoder = self.dropout(h_seasonally)
        out_encoder = self.out_encoder(x_encoder)
        out_encoder = out_encoder.view(batch, 1, self.output_channels)

        h = h_daily+h_biweekly+h_seasonally
        c = c_daily+c_biweekly+c_seasonally
        x_decoder, _ = self.decoder(x_dynamic[:, -forecast:], (torch.unsqueeze(h, axis=0),torch.unsqueeze(c, axis=0)))
        x_decoder = self.dropout(x_decoder)
        out_decoder = self.out_decoder(x_decoder)
        out_decoder = out_decoder.view(batch, forecast, self.output_channels)

        return out_encoder, out_decoder


# ## LSTM HIERARCHICAL ENC MODELS

# In[11]:


class lstm_hierarchical_enc_dec(torch.nn.Module):

    def __init__(self, input_channels, latent_code_dim, forward_code_dim, output_channels, dropout):
        super(lstm_hierarchical_enc_dec,self).__init__()

        # PARAMETERS
        self.input_channels = input_channels
        self.latent_code_dim = latent_code_dim
        self.forward_code_dim = forward_code_dim
        self.output_channels = output_channels

        # LAYERS
        self.encoder_daily = torch.nn.LSTM(input_size=self.input_channels, hidden_size=self.latent_code_dim, bidirectional=True, batch_first=True)
        self.encoder_biweekly = torch.nn.LSTM(input_size=self.latent_code_dim, hidden_size=self.latent_code_dim, bidirectional=True, batch_first=True)
        self.encoder_seasonally = torch.nn.LSTM(input_size=self.latent_code_dim, hidden_size=self.latent_code_dim, bidirectional=True, batch_first=True)
        self.out_encoder = torch.nn.Linear(in_features=self.forward_code_dim, out_features=self.output_channels)
        self.decoder = torch.nn.LSTM(input_size=self.input_channels, hidden_size=self.forward_code_dim, batch_first=True)
        self.out_decoder = torch.nn.Linear(in_features=self.forward_code_dim, out_features=self.output_channels)
        self.dropout = torch.nn.Dropout(p=dropout)

        # INITIALIZATION
        for m in self.modules():
            if isinstance(m, torch.nn.Conv2d) or isinstance(m, torch.nn.Linear):
                torch.nn.init.xavier_uniform_(m.weight)

    def forward(self, x_dynamic, forecast):

        # GET SHAPES
        batch, window, _ = x_dynamic.shape

        # OPERATIONS
        x_encoder = x_dynamic[:, :-forecast]
        x_encoder, (h_daily, c_daily) = self.encoder_daily(x_encoder)
        x_encoder = x_encoder[:, :, :self.latent_code_dim] + x_encoder[:, :, self.latent_code_dim:]
        h_daily, c_daily = torch.sum(h_daily, axis=0), torch.sum(c_daily, axis=0)

        x_encoder, (h_biweekly, c_biweekly) = self.encoder_biweekly(x_encoder.flip(1)[:, ::14].flip(1))
        x_encoder = x_encoder[:, :, :self.latent_code_dim] + x_encoder[:, :, self.latent_code_dim:]
        h_biweekly, c_biweekly = torch.sum(h_biweekly, axis=0), torch.sum(c_biweekly, axis=0)

        x_encoder, (h_seasonally, c_seasonally) = self.encoder_seasonally(x_encoder.flip(1)[:, ::6].flip(1))
        x_encoder = x_encoder[:, :, :self.latent_code_dim] + x_encoder[:, :, self.latent_code_dim:]
        h_seasonally, c_seasonally = torch.sum(h_seasonally, axis=0), torch.sum(c_seasonally, axis=0)

        
        h = h_daily+h_biweekly+h_seasonally
        c = c_daily+c_biweekly+c_seasonally
        
        x_encoder = self.dropout(h)
        out_encoder = self.out_encoder(x_encoder)
        out_encoder = out_encoder.view(batch, 1, self.output_channels)


        x_decoder, _ = self.decoder(x_dynamic[:, -forecast:], (torch.unsqueeze(h, axis=0),torch.unsqueeze(c, axis=0)))
        x_decoder = self.dropout(x_decoder)
        out_decoder = self.out_decoder(x_decoder)
        out_decoder = out_decoder.view(batch, forecast, self.output_channels)

        return out_encoder, out_decoder


# In[12]:


class lstm_hierarchical_enc_with_label_dec(torch.nn.Module):

    def __init__(self, input_channels, latent_code_dim, forward_code_dim, output_channels, dropout):
        super(lstm_hierarchical_enc_with_label_dec,self).__init__()

        # PARAMETERS
        self.input_channels = input_channels
        self.latent_code_dim = latent_code_dim
        self.forward_code_dim = forward_code_dim
        self.output_channels = output_channels

        # LAYERS
        self.encoder_daily = torch.nn.LSTM(input_size=self.input_channels+self.output_channels, hidden_size=self.latent_code_dim, bidirectional=True, batch_first=True)
        self.encoder_biweekly = torch.nn.LSTM(input_size=self.latent_code_dim, hidden_size=self.latent_code_dim, bidirectional=True, batch_first=True)
        self.encoder_seasonally = torch.nn.LSTM(input_size=self.latent_code_dim, hidden_size=self.latent_code_dim, bidirectional=True, batch_first=True)
        self.out_encoder = torch.nn.Linear(in_features=self.forward_code_dim, out_features=self.output_channels)
        self.decoder = torch.nn.LSTM(input_size=self.input_channels, hidden_size=self.forward_code_dim, batch_first=True)
        self.out_decoder = torch.nn.Linear(in_features=self.forward_code_dim, out_features=self.output_channels)
        self.dropout = torch.nn.Dropout(p=dropout)

        # INITIALIZATION
        for m in self.modules():
            if isinstance(m, torch.nn.Conv2d) or isinstance(m, torch.nn.Linear):
                torch.nn.init.xavier_uniform_(m.weight)

    def forward(self, x_dynamic, y, forecast):

        # GET SHAPES
        batch, window, _ = x_dynamic.shape

        # OPERATIONS
        x = torch.cat((x_dynamic, y), dim=-1)

        # OPERATIONS
        x_encoder = x[:, :-forecast]
        x_encoder, (h_daily, c_daily) = self.encoder_daily(x_encoder)
        x_encoder = x_encoder[:, :, :self.latent_code_dim] + x_encoder[:, :, self.latent_code_dim:]
        h_daily, c_daily = torch.sum(h_daily, axis=0), torch.sum(c_daily, axis=0)

        x_encoder, (h_biweekly, c_biweekly) = self.encoder_biweekly(x_encoder.flip(1)[:, ::14].flip(1))
        x_encoder = x_encoder[:, :, :self.latent_code_dim] + x_encoder[:, :, self.latent_code_dim:]
        h_biweekly, c_biweekly = torch.sum(h_biweekly, axis=0), torch.sum(c_biweekly, axis=0)

        x_encoder, (h_seasonally, c_seasonally) = self.encoder_seasonally(x_encoder.flip(1)[:, ::6].flip(1))
        x_encoder = x_encoder[:, :, :self.latent_code_dim] + x_encoder[:, :, self.latent_code_dim:]
        h_seasonally, c_seasonally = torch.sum(h_seasonally, axis=0), torch.sum(c_seasonally, axis=0)

        
#         h = h_daily+h_biweekly+h_seasonally
#         c = c_daily+c_biweekly+c_seasonally
        h = torch.cat((h_daily, h_biweekly, h_seasonally), dim=-1)
        c = torch.cat((c_daily, c_biweekly, c_seasonally), dim=-1)
        
        x_encoder = self.dropout(h)
        out_encoder = self.out_encoder(x_encoder)
        out_encoder = out_encoder.view(batch, 1, self.output_channels)

        x_decoder, _ = self.decoder(x_dynamic[:, -forecast:], (torch.unsqueeze(h, axis=0),torch.unsqueeze(c, axis=0)))
        x_decoder = self.dropout(x_decoder)
        out_decoder = self.out_decoder(x_decoder)
        out_decoder = out_decoder.view(batch, forecast, self.output_channels)

        return out_encoder, out_decoder


# # TFT

# In[13]:


import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple

class GatedResidualNetwork(nn.Module):
    def __init__(self, input_size: int, hidden_size: int, output_size: int, dropout: float = 0.1, context_size: Optional[int] = None):
        super(GatedResidualNetwork, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size  
        self.context_size = context_size
        self.dropout = dropout
        


        # Calculate actual input size based on context
        self.actual_input_size = input_size + (context_size or 0)

        # Initialize layers with correct dimensions
        self.fc1 = nn.Linear(self.actual_input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, output_size)
        self.elu = nn.ELU()
        
        self.gate = nn.Linear(self.actual_input_size, output_size)
        self.norm = nn.LayerNorm(output_size)
        self.dropout_layer = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, c: Optional[torch.Tensor] = None) -> torch.Tensor:
        if c is not None:
            # Expand context if needed
            if len(c.shape) == 2:
                c = c.unsqueeze(1).expand(-1, x.shape[1], -1)
            x = torch.cat([x, c], dim=-1)
            
        # Main branch
        eta2 = self.fc1(x)
        eta2 = self.dropout_layer(eta2)
        eta2 = self.elu(eta2)
        eta1 = self.fc2(eta2)
        
        # Gating mechanism
        gate = torch.sigmoid(self.gate(x))
        eta1 = gate * eta1
        
#         print("eta1", eta1.shape)
        
#         print("x befopre skip", x.shape)
        
        # Add skip connection from original input
        x_skip = x[..., :self.output_size]  # Take only the first output_size dimensions
        
#         print("xskip", x_skip.shape)
        return self.norm(x_skip + eta1)

class VariableSelectionNetwork(nn.Module):
    def __init__(self, input_size: int, num_features: int, hidden_size: int, dropout: float = 0.1, context_size: Optional[int] = None):
        super(VariableSelectionNetwork, self).__init__()
        self.hidden_size = hidden_size
        self.num_features = num_features
        
        # GRN for variable selection weights
        self.weight_grn = GatedResidualNetwork(
            input_size=input_size * num_features,
            hidden_size=hidden_size,
            output_size=num_features,
            dropout=dropout,
            context_size=context_size
        )
        
        # Individual GRNs for processing each variable
        self.feature_grns = nn.ModuleList([
            GatedResidualNetwork(
                input_size=input_size,
                hidden_size=hidden_size,
                output_size=hidden_size,
                dropout=dropout
            ) for _ in range(num_features)
        ])
        
    def forward(self, x: torch.Tensor, c: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        batch_size = x.shape[0]
        time_steps = x.shape[1]
        
        # Transform each feature individually
        processed_features = []
        for i, grn in enumerate(self.feature_grns):
            processed_features.append(grn(x[..., i, :]))
        processed_features = torch.stack(processed_features, dim=-2)
        
        # Get variable selection weights
        flat_x = x.view(batch_size, time_steps, -1)
        weights = self.weight_grn(flat_x, c)
        weights = torch.softmax(weights, dim=-1)
        
        # Apply weights to processed features
        weighted_features = weights.unsqueeze(-1) * processed_features
        combined_features = weighted_features.sum(dim=-2)
        
        return combined_features, weights

class TFT(nn.Module):
    def __init__(
        self,
        output_channels: int,
        input_channels: int,
        num_static: Optional[int] = None,
        num_observed: Optional[int] = None,
        hidden_size: int = 32,
        latent_code_dim: int = 32,
        nhead: int = 4,
        dropout: float = 0.1,
        context_length: int = 24,
        forecast_steps: int = 12
    ):
        super(TFT, self).__init__()
        
        
        
        self.hidden_size = hidden_size
        self.latent_code_dim = latent_code_dim
        self.context_length = context_length
        self.forecast_steps = forecast_steps
        
        # Input projections
        self.target_proj = nn.Linear(output_channels, hidden_size)
        self.dynamic_proj = nn.Linear(input_channels, hidden_size)
        
        # Static variables processing
        if num_static is not None:
            self.static_proj = nn.Linear(num_static, hidden_size)
            self.static_grn = GatedResidualNetwork(
                input_size=hidden_size,
                hidden_size=hidden_size,
                output_size=hidden_size,
                dropout=dropout
            )
        else:
            self.static_proj = None
            self.static_grn = None
            
        # Observed variables processing
        if num_observed is not None:
            self.observed_proj = nn.Linear(num_observed, hidden_size)
        else:
            self.observed_proj = None
            
        # Count number of features for variable selection
        num_features = 2  # target + dynamic
        if num_observed is not None:
            num_features += 1
            
        # Variable selection network
        self.temporal_vsn = VariableSelectionNetwork(
            input_size=hidden_size,
            num_features=num_features,
            hidden_size=hidden_size,
            dropout=dropout,
            context_size=hidden_size if num_static is not None else None
        )
        
        # Static enrichment
        self.static_enrichment = GatedResidualNetwork(
            input_size=hidden_size,
            hidden_size=hidden_size,
            output_size=hidden_size,
            dropout=dropout,
            context_size=hidden_size
        )
        
        # Temporal processing
        self.lstm_encoder = nn.LSTM(
            input_size=hidden_size,
            hidden_size=latent_code_dim,
            batch_first=True
        )
        
        self.lstm_decoder = nn.LSTM(
            input_size=hidden_size,
            hidden_size=latent_code_dim,
            batch_first=True
        )
        
        # Attention
        self.multihead_attn = nn.MultiheadAttention(
            embed_dim=latent_code_dim,
            num_heads=nhead,
            dropout=dropout,
            batch_first=True
        )
        
        # Output processing
        self.pre_output_grn = GatedResidualNetwork(
            input_size=latent_code_dim,
            hidden_size=hidden_size,
            output_size=hidden_size,
            dropout=dropout
        )
        self.output_layer = nn.Linear(hidden_size, output_channels)
        
    def forward(
        self,
        x_dynamic: torch.Tensor,
        y: torch.Tensor,
        forecast: int,
        static_input: Optional[torch.Tensor] = None,
        observed_input: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        
        
        y = y [:,:self.context_length]   #removing future targets
        batch_size = y.shape[0]
        
        # Process static inputs if available
        static_context = None
        if static_input is not None and self.static_proj is not None:
            static_embedded = self.static_proj(static_input)
            static_context = self.static_grn(static_embedded)
        
        
        # Process inputs
        y_embedded = self.target_proj(y)
        dynamic_embedded = self.dynamic_proj(x_dynamic)
        
        # Stack inputs for variable selection
        temporal_features = [
            y_embedded,
            dynamic_embedded[:, :self.context_length]
        ]
        
        if observed_input is not None and self.observed_proj is not None:
            observed_embedded = self.observed_proj(observed_input)
            temporal_features.append(observed_embedded)
            
        temporal_features = torch.stack(temporal_features, dim=2)
        
        # Variable selection
        temporal_features, _ = self.temporal_vsn(temporal_features, static_context)
        
        # Static enrichment
        if static_context is not None:
            enrichment_context = static_context.unsqueeze(1).expand(-1, temporal_features.size(1), -1)
            temporal_features = self.static_enrichment(temporal_features, enrichment_context)
        
        
        
#         print("temporal_features", temporal_features.shape)  #5X720X24
        
        # Temporal processing
        encoder_output, (h_n, c_n) = self.lstm_encoder(temporal_features)  
        
#         print("encoder_output", encoder_output.shape)     #5X720X8
        
        decoder_input = dynamic_embedded[:, self.context_length:]      #5X30X8
        decoder_output, _ = self.lstm_decoder(decoder_input, (h_n, c_n))
        
        # Combine encoder and decoder outputs
        sequence = torch.cat([encoder_output, decoder_output], dim=1)
        
#         print("sequence", sequence.shape)     #5 X 750 X8
        
        # Self-attention
        attn_output, _ = self.multihead_attn(sequence, sequence, sequence)
        
#         print(attn_output.shape)  #5 X 750X 8
        
        # Final processing
        output = self.pre_output_grn(attn_output)
        predictions = self.output_layer(output)
        
        # Return predictions for forecast horizon
        return predictions[:, -self.forecast_steps:]

# Example usage
if __name__ == "__main__":
    # Example dimensions
    batch_size = 64
    output_channels = 1
    input_channels = 5
    num_static = 27
    num_observed = 0
    context_length = 365
    forecast_steps = 7
    
    # Create model
    model = TFT(
        output_channels=output_channels,
        input_channels=input_channels,
        num_static=num_static,
        num_observed=num_observed,
        hidden_size=32,
        latent_code_dim=32,
        nhead=4,
        dropout=0.1,
        context_length=context_length,
        forecast_steps=forecast_steps
    )
    
    # Create example inputs
    y = torch.randn(batch_size, context_length, output_channels)
    x_dynamic = torch.randn(batch_size, context_length + forecast_steps, input_channels)
    observed_input = torch.randn(batch_size, context_length, num_observed)
    static_input = torch.randn(batch_size, num_static)
    
    print("Input shapes:")
    print("y:", y.shape)
    print("x_dynamic:", x_dynamic.shape)
    print("observed_input:", observed_input.shape)
    print("static_input:", static_input.shape)
    
    # Forward pass
    outputs = model(x_dynamic, y, forecast_steps, static_input, observed_input)
    print("\nOutput shape:", outputs.shape)  # [batch_size, forecast_steps, output_channels]


# # RR-FORMER

# In[15]:


import torch
import torch.nn as nn
import math

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_seq_length=1000):
        super().__init__()
        position = torch.arange(max_seq_length).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_seq_length, d_model)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return x + self.pe[:x.size(1)]

class RRformer(nn.Module):
    def __init__(self, 
                 input_channels,
                 output_channels,
                 context_length=720,    # Historical context length
                 forecast_steps=28,     # Number of steps to forecast
                 latent_code_dim=64,           
                 nhead=4,              
                 num_encoder_layers=4,  
                 num_decoder_layers=4,  
                 forward_code_dim=256,   
                 dropout=0.1):
        super().__init__()
        
        self.context_length = context_length
        self.forecast_steps = forecast_steps
        self.total_length = context_length + forecast_steps
        
        # Input projections
        self.input_projection = nn.Linear(input_channels, latent_code_dim)
        self.output_projection = nn.Linear(output_channels, latent_code_dim)
        
        # Positional encodings
        self.pos_encoder = PositionalEncoding(latent_code_dim, max_seq_length=self.total_length)
        self.pos_decoder = PositionalEncoding(latent_code_dim, max_seq_length=self.total_length)
        
        # Transformer layers
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=latent_code_dim,
            nhead=nhead,
            dim_feedforward=forward_code_dim,
            dropout=dropout,
            batch_first=True
        )
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=latent_code_dim,
            nhead=nhead,
            dim_feedforward=forward_code_dim,
            dropout=dropout,
            batch_first=True
        )
        
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_encoder_layers)
        self.transformer_decoder = nn.TransformerDecoder(decoder_layer, num_decoder_layers)
        
        # Single output layer
        self.output_layer = nn.Linear(latent_code_dim, output_channels)
        
        # Layer norm
        self.norm = nn.LayerNorm(latent_code_dim)
        
    def generate_square_subsequent_mask(self, sz):
        """Generate mask for decoder self-attention"""
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask
        
    def forward(self, x_dynamic, y, forecast):
        """
        
        
        Returns: Predictions for forecast period [batch, forecast_steps, output_dim]
        """
        
        
        src = x_dynamic #src: Meteorological input data [batch, total_length, input_channels]
        
        tgt = y[:, :self.context_length]    # tgt: Target runoff data [batch, context_length, output_dim]
        
        batch_size = src.size(0)
        
        # Ensure input lengths match expected lengths
        assert src.size(1) == self.total_length, f"Expected source length {self.total_length}, got {src.size(1)}"
        assert tgt.size(1) == self.context_length, f"Expected target length {self.context_length}, got {tgt.size(1)}"
        
        # Create padded target sequence
        padded_target = torch.zeros(batch_size, self.total_length, tgt.size(-1), device=tgt.device)
        padded_target[:, :self.context_length] = tgt
        
        # Create masks
        tgt_mask = self.generate_square_subsequent_mask(self.total_length).to(src.device)
        
        # Input projections and positional encoding
        src = self.input_projection(src)
        src = self.pos_encoder(src)
        
        padded_target = self.output_projection(padded_target)
        padded_target = self.pos_decoder(padded_target)
        
        # Transformer encoder-decoder
        memory = self.transformer_encoder(src)
        decoder_output = self.transformer_decoder(padded_target, memory, tgt_mask)
        
        # Generate predictions
        predictions = self.output_layer(decoder_output)
        
        # Return only the forecast period
        return predictions[:, self.context_length:, :]




# # TEST MODELS

# In[23]:


if __name__ == "__main__":
    batch = 5
    context = 365
    forecast = 7
    window = context + forecast
    stride = 30
    channels = list(range(33))
    static_channels = channels[5:-1]
    dynamic_channels = channels[:5]
    output_channels = [channels[-1]]
    data = torch.randn(batch, window, len(static_channels)+len(dynamic_channels)+len(output_channels))
    print(data.shape, "DATA")

    unknown = -999
    latent_code_dim = 256
    forward_code_dim = 256
    device = torch.device("cpu")
    dropout = 0.4

    x_dynamic = data[:,:,dynamic_channels].to(device)
    y = torch.cat((torch.zeros(batch,1,len(output_channels)), data[:,:-1,output_channels]), dim=1).to(device)

    architecture = "lstm"
    model = globals()[architecture](input_channels=len(dynamic_channels), latent_code_dim=latent_code_dim, output_channels=len(output_channels), dropout=dropout)
    model = model.to(device)
    pytorch_total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    out = model(x_dynamic=x_dynamic, forecast=forecast)
    print(x_dynamic.shape, out.shape, "#:{}".format(pytorch_total_params), architecture)

    architecture = "lstm_bidirectional"
    model = globals()[architecture](input_channels=len(dynamic_channels), latent_code_dim=latent_code_dim, output_channels=len(output_channels), dropout=dropout)
    model = model.to(device)
    pytorch_total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    out = model(x_dynamic=x_dynamic, forecast=1)
    print(x_dynamic.shape, out.shape, "#:{}".format(pytorch_total_params), architecture)

    architecture = "lstm_mul_enc"
    model = globals()[architecture](input_channels=len(dynamic_channels), latent_code_dim=latent_code_dim, output_channels=len(output_channels), dropout=dropout)
    model = model.to(device)
    pytorch_total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    out = model(x_dynamic=x_dynamic, forecast=1)
    print(x_dynamic.shape, out.shape, "#:{}".format(pytorch_total_params), architecture)

    architecture = "lstm_hierarchical_enc"
    model = globals()[architecture](input_channels=len(dynamic_channels), latent_code_dim=latent_code_dim, output_channels=len(output_channels), dropout=dropout)
    model = model.to(device)
    pytorch_total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    out = model(x_dynamic=x_dynamic, forecast=1)
    print(x_dynamic.shape, out.shape, "#:{}".format(pytorch_total_params), architecture)

    architecture = "lstm_enc_dec"
    model = globals()[architecture](input_channels=len(dynamic_channels), latent_code_dim=latent_code_dim, forward_code_dim=forward_code_dim, output_channels=len(output_channels), dropout=dropout)
    model = model.to(device)
    pytorch_total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    out_encoder, out_decoder = model(x_dynamic=x_dynamic, forecast=forecast)
    print(x_dynamic.shape, out_encoder.shape, out_decoder.shape, "#:{}".format(pytorch_total_params), architecture)

    architecture = "lstm_ar"
    model = globals()[architecture](input_channels=len(dynamic_channels), forward_code_dim=forward_code_dim, output_channels=len(output_channels), dropout=dropout)
    model = model.to(device)
    pytorch_total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    out_decoder = model(x_dynamic=x_dynamic,y=y, forecast=forecast)
    print(x_dynamic.shape, out_decoder.shape, "#:{}".format(pytorch_total_params), architecture)

    architecture = "lstm_mul_enc_dec"
    model = globals()[architecture](input_channels=len(dynamic_channels), latent_code_dim=latent_code_dim, forward_code_dim=forward_code_dim, output_channels=len(output_channels), dropout=dropout)
    model = model.to(device)
    pytorch_total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    out_encoder, out_decoder = model(x_dynamic=x_dynamic, forecast=forecast)
    print(x_dynamic.shape, out_encoder.shape, out_decoder.shape, "#:{}".format(pytorch_total_params), architecture)

    architecture = "lstm_hierarchical_enc_dec"
    model = globals()[architecture](input_channels=len(dynamic_channels), latent_code_dim=latent_code_dim, forward_code_dim=forward_code_dim, output_channels=len(output_channels), dropout=dropout)
    model = model.to(device)
    pytorch_total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    out_encoder, out_decoder = model(x_dynamic=x_dynamic, forecast=forecast)
    print(x_dynamic.shape, out_encoder.shape, out_decoder.shape, "#:{}".format(pytorch_total_params), architecture)


    architecture = "lstm_enc_with_label_dec"
    model = globals()[architecture](input_channels=len(dynamic_channels), latent_code_dim=latent_code_dim, forward_code_dim=forward_code_dim, output_channels=len(output_channels), dropout=dropout)
    model = model.to(device)
    pytorch_total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    out_encoder, out_decoder = model(x_dynamic=x_dynamic,y=y, forecast=forecast)
    print(x_dynamic.shape, y.shape,out_encoder.shape, out_decoder.shape, "#:{}".format(pytorch_total_params), architecture)

    latent_code_dim = 85
    forward_code_dim = 255
    architecture = "lstm_hierarchical_enc_with_label_dec"
    model = globals()[architecture](input_channels=len(dynamic_channels), latent_code_dim=latent_code_dim, forward_code_dim=forward_code_dim, output_channels=len(output_channels), dropout=dropout)
    model = model.to(device)
    pytorch_total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    out_encoder, out_decoder = model(x_dynamic=x_dynamic,y=y, forecast=forecast)
    print(x_dynamic.shape, y.shape,out_encoder.shape, out_decoder.shape, "#:{}".format(pytorch_total_params), architecture)
    
    
    latent_code_dim = 128
    architecture = "TFT"
    model = globals()[architecture](input_channels=len(dynamic_channels), latent_code_dim=latent_code_dim, hidden_size=latent_code_dim, output_channels=len(output_channels), dropout=dropout, forecast_steps = forecast, context_length = context, nhead=4 )
    model = model.to(device)
    pytorch_total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    out = model(x_dynamic=x_dynamic, y=y, forecast=forecast)
    print(x_dynamic.shape, out.shape, "#:{}".format(pytorch_total_params), architecture)
    
    forward_code_dim = 128
    latent_code_dim = 96
    
    architecture = "RRformer"
    model = globals()[architecture](input_channels=len(dynamic_channels), latent_code_dim=latent_code_dim, forward_code_dim=forward_code_dim, output_channels=len(output_channels), dropout=dropout, forecast_steps = forecast, context_length = context, nhead=4,num_encoder_layers=4, num_decoder_layers=4 )
    model = model.to(device)
    pytorch_total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    out = model(x_dynamic=x_dynamic, y=y, forecast=forecast)
    print(x_dynamic.shape, out.shape, "#:{}".format(pytorch_total_params), architecture)
    
    forward_code_dim = 128
    latent_code_dim = 96
    architecture = "ExoTST"
    model = globals()[architecture](input_channels=len(dynamic_channels), latent_code_dim=latent_code_dim, forward_code_dim=forward_code_dim, output_channels=len(output_channels), dropout=dropout, forecast_steps = forecast, context_length = context, nhead=8,num_transformer_layers=3 )
    model = model.to(device)
    pytorch_total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    out = model(x_dynamic=x_dynamic, y=y, forecast=forecast)
    print(x_dynamic.shape, out.shape, "#:{}".format(pytorch_total_params), architecture)
    
    
    
    


# In[ ]:





# In[ ]:





# In[ ]:




