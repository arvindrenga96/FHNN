#!/usr/bin/env python
# coding: utf-8

# # IMPORT LIBRARIES

# In[1]:


from tqdm import *
import sys
sys.path.append("../")

import config
import MODEL
import UTILS

import os
import time
import argparse
import math
import numpy as np
import pandas as pd
from collections import Counter
import random

import torch
from torch.utils.data.dataset import Dataset
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import dask.array as da

torch.manual_seed(0)


# # HYPERPARAMETERS

# In[2]:


# parser = argparse.ArgumentParser(
#     description='arguments')
# parser.add_argument('--init', type=int, default=0, help='init number')
# parser.add_argument('--fold', type=int, default=0, help='fold number')
# parser.add_argument('--model_name', type=str, default='ctlstm', help='model_name')
# parser.add_argument('--date', type=str, default='20230420', help='date')

# args = parser.parse_args()


# In[3]:


# TIME SERIES INFO
window = config.window
stride = config.stride

# CHANNELS INFO
dynamic_channels = config.dynamic_channels
static_channels = config.static_channels
output_channels = config.output_channels

# LABELS INFO
unknown = config.unknown

# MODEL INFO
model_name = "bilstm_encoder" #args.model_name
latent_code_dim = config.latent_code_dim
device = torch.device(config.device)
dropout = config.dropout

# TRAIN INFO
train = config.train
batch_size = 64#config.batch_size
epochs = 2# config.epochs
learning_rate = config.learning_rate
init = 0#args.init
fold = 0#args.fold
forecast = 1 #config.forecast
context = config.context

print("Hyperparameters:{}".format(model_name))
print("window : {}".format(window))
print("stride : {}".format(stride))
print("dynamic_channels : {}".format(dynamic_channels))
print("static_channels : {}".format(static_channels))
print("output_channels : {}".format(output_channels))
print("unknown : {}".format(unknown))
print("model_name : {}".format(model_name))
print("latent_code_dim : {}".format(latent_code_dim))
print("context : {}".format(context))
print("forecast : {}".format(forecast))
print("device : {}".format(device))
print("dropout : {}".format(dropout))
print("train : {}".format(train))
print("batch_size : {}".format(batch_size))
print("epochs : {}".format(epochs))
print("learning_rate : {}".format(learning_rate))
print("init : {}".format(init))
print("fold : {}".format(fold))


# # DEFINE DIRECTORIES

# In[4]:


DATE = "20230613"# args.date
PREPROCESSED_DIR = config.PREPROCESSED_DIR
RESULT_DIR = os.path.join(config.RESULT_DIR, DATE)
MODEL_DIR = os.path.join(config.MODEL_DIR, DATE)


# # LOAD DATA

# In[5]:


def load_dataset(file):
    return np.load(os.path.join(PREPROCESSED_DIR, f"{file}.npz"), allow_pickle=True)

def get_data(dataset, preprocessed=True):
    data = dataset["data"]
    #data = da.from_array(dataset["data"], chunks="auto")
    
    if preprocessed:
        data_mean = dataset["train_data_mean"]
        data_std = dataset["train_data_std"]
        
        # Vectorized normalization using np.where
        data = np.where(data_std != 0, (data - data_mean) / data_std, data)
    
    # Replace NaNs with unknown value
    data = np.nan_to_num(data, nan=unknown)
    
    return data



# # TRAIN MODEL

# In[6]:


if train:
    print("fold:{}\tinit:{}".format(fold, init))
    
    # Usage
    file, index = "strided_train", "in_indices"
    dataset = load_dataset(file)
    data = get_data(dataset)

    file, index = "strided_valid", "in_indices"
    dataset = load_dataset(file)
    data_val = get_data(dataset)
    
    
    # BUILD MODEL
    model = getattr(MODEL, "lstm_bidirectional")(input_channels=len(dynamic_channels)+ len(static_channels), latent_code_dim=latent_code_dim, output_channels=len(output_channels), dropout=dropout)
    model = model.to(device)
    pytorch_total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print("#Parameters:{}".format(pytorch_total_params))
    print(model)
    criterion = torch.nn.MSELoss(reduction="none")
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    train_loss = []
    valid_loss = []
    min_loss = 10000

    for epoch in range(1,epochs+1):

        start = time.time()

        # LOSS ON TRAIN SET
        model.train()

        # LOAD DATA
        nodes, years, window, channels = data.shape
        print(nodes, years, window, channels)

        # GET RANDOM YEARS
        random_years = np.zeros((nodes, years))
        for node in range(nodes):
            random_years[node] = random.sample(range(years), years)
        random_years = random_years.astype(np.int64)
        # print(random_years.shape)

        # LOSS
        epoch_loss = 0
        for year in range(random_years.shape[1]):
            if not year%100:
                print(year)

            #Get instance for each node
            node_data = data[np.arange(nodes), random_years[:, year]]
            # print(node_data.shape)

            random_batches = random.sample(range(node_data.shape[0]),node_data.shape[0])
            for batch in range(math.ceil(nodes/batch_size)):

                optimizer.zero_grad()

                # GET BATCH DATA AND LABEL
                random_batch = random_batches[batch*batch_size:(batch+1)*batch_size]
                batch_data = torch.from_numpy(node_data[random_batch]).to(device)
                batch_dynamic_input = batch_data[:, :, dynamic_channels].to(device)
                batch_static_input = batch_data[:, :, static_channels].to(device)
                batch_input = torch.concatenate([batch_dynamic_input, batch_static_input], axis =-1)
                batch_label = batch_data[:, -1, output_channels].to(device)
                # print(batch_dynamic_input.shape, batch_static_input.shape, batch_label.shape)

                #Select context to feed into LSTM
                batch_input = batch_input[:,:context]


                # GET OUTPUT from encoder model
                batch_pred = model(x_dynamic=batch_input,forecast=forecast).squeeze(1)

                # CALCULATE LOSS
                batch_loss = criterion(batch_label, batch_pred)											# PER CHANNEL LOSS
                batch_loss = torch.mean(batch_loss)
                # mask = (batch_label!=unknown).float()													# CREATE MASK
                # batch_loss = batch_loss * mask															# MULTIPLY MASK
                # batch_loss, mask = torch.sum(batch_loss, dim=2), (torch.sum(mask, dim=2)>0).float()		# PER INSTANCE LOSS
                # batch_loss = torch.sum(batch_loss)/torch.sum(mask)										# MEAN SEQUENCE LOSS
                # print(batch_loss.shape)

                # LOSS BACKPROPOGATE
                batch_loss.backward()
                optimizer.step()

                # AGGREGATE LOSS
                epoch_loss += batch_loss.item()

        epoch_loss /= ((batch+1)*(year+1))
        print('Epoch:{}\tTrain Loss:{:.4f}'.format(epoch, epoch_loss), end="\t")
        train_loss.append(epoch_loss)

        # SCORE ON VALIDATION SET
        model.eval()

        # LOAD DATA
        nodes, years, window, channels = data_val.shape
        # print(nodes, years, window, channels)

        # SCORE
        epoch_loss = 0
        for year in range(years):

            #Get instance for each node
            node_data = data_val[np.arange(nodes), year]
            # print(node_data.shape)

            for batch in range(math.ceil(nodes/batch_size)):

                # GET BATCH DATA AND LABEL
                random_batch = random_batches[batch*batch_size:(batch+1)*batch_size]
                batch_data = torch.from_numpy(node_data[random_batch]).to(device)
                batch_dynamic_input = batch_data[:, :, dynamic_channels].to(device)
                batch_static_input = batch_data[:, :, static_channels].to(device)
                batch_input = torch.concatenate([batch_dynamic_input, batch_static_input], axis =-1)
                batch_label = batch_data[:, -1, output_channels].to(device)
                # print(batch_dynamic_input.shape, batch_static_input.shape, batch_label.shape)

                #Select context to feed into LSTM
                batch_input = batch_input[:,:context]


                # GET OUTPUT from encoder model
                batch_pred = model(x_dynamic=batch_input,forecast=forecast).squeeze(1)

                # CALCULATE LOSS
                batch_loss = criterion(batch_label, batch_pred)											# PER CHANNEL LOSS
                batch_loss = torch.mean(batch_loss)
                # mask = (batch_label!=unknown).float()													# CREATE MASK
                # batch_loss = batch_loss * mask															# MULTIPLY MASK
                # batch_loss, mask = torch.sum(batch_loss, dim=2), (torch.sum(mask, dim=2)>0).float()		# PER SEQUENCE LOSS
                # batch_loss = torch.sum(batch_loss)/torch.sum(mask)										# MEAN SEQUENCE LOSS
                # print(batch_loss.shape)

                # AGGREGATE LOSS
                epoch_loss += batch_loss.item()

        epoch_loss /= ((batch+1)*(year+1))
        print("Val Loss:{:.4f}\tMin Loss:{:.4f}".format(epoch_loss, min_loss), end="\t")
        valid_loss.append(epoch_loss)
        if min_loss>epoch_loss:
            min_loss = epoch_loss
            torch.save(model.state_dict(), os.path.join(MODEL_DIR, "{}".format(model_name)))
        end = time.time()
        print("Time:{:.4f}".format(end-start))

    # PLOT LOSS
    fig = plt.figure(figsize=(10,10))
    ax1 = fig.add_subplot(111)
    ax1.set_xlabel("#Epoch", fontsize=50)

    # PLOT TRAIN LOSS
    lns1 = ax1.plot(train_loss, color='red', marker='o', linewidth=4, label="TRAIN LOSS")

    # PLOT VALIDATION SCORE
    ax2 = ax1.twinx()
    lns2 = ax2.plot(valid_loss, color='blue', marker='o', linewidth=4, label="VAL LOSS")

    # added these three lines
    lns = lns1+lns2
    labs = [l.get_label() for l in lns]
    ax1.legend(lns, labs, loc="upper right", fontsize=40, frameon=False)

    plt.tight_layout(pad=0.0,h_pad=0.0,w_pad=0.0)
    plt.savefig(os.path.join(RESULT_DIR, "{}_SCORE.pdf".format(model_name)), format = "pdf")
    plt.close()


# In[ ]:





# # TEST MODEL

# ## IN DISTRIBUTION

# In[ ]:


print("IN\tfold:{}\tinit:{}".format(fold, init))

# BUILD MODEL
model = getattr(MODEL, "ctlstm")(input_dynamic_channels=len(dynamic_channels), input_static_channels=len(static_channels), hidden_dim=forward_code_dim, output_channels=len(output_channels), dropout=dropout)
model = model.to(device)
pytorch_total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
# print("#Parameters:{}".format(pytorch_total_params))
# print(model)

# LOAD MODEL
model.load_state_dict(torch.load(os.path.join(MODEL_DIR, "{}".format(model_name))))
model.eval()

# LOAD DATA
file, index = "strided_test", "in_indices"
dataset = load_dataset(file)
data = get_data(dataset, index,fold=fold)
nodes, years, window, channels = data.shape
print(nodes, years, window, channels)

dataset_true = unknown*np.ones((nodes, years, window, len(output_channels)), dtype=np.float32)
dataset_pred = unknown*np.ones((nodes, years, window, len(output_channels)), dtype=np.float32)
for year in range(years):

    #Get instance for each node
    node_data = data[np.arange(nodes), year]
    # print(node_data.shape)

    for batch in range(math.ceil(nodes/batch_size)):

        # GET BATCH DATA AND LABEL
        batch_data = torch.from_numpy(node_data[batch*batch_size:(batch+1)*batch_size]).to(device)
        batch_dynamic_input = batch_data[:, :, dynamic_channels].to(device)
        batch_static_input = batch_data[:, :, static_channels].to(device)
        batch_label = batch_data[:, :, output_channels].to(device)
        # print(batch_dynamic_input.shape, batch_static_input.shape, batch_label.shape)

        # GET OUTPUT
        batch_pred = model(x_dynamic=batch_dynamic_input, x_static=batch_static_input)
        # print(batch_pred.shape)

        # STORE OUTPUT
        dataset_true[batch*batch_size:(batch+1)*batch_size, year] = batch_label.detach().cpu().numpy()
        dataset_pred[batch*batch_size:(batch+1)*batch_size, year] = batch_pred.detach().cpu().numpy()

dataset_true = (dataset_true*dataset["train_data_stds"][fold][output_channels])+dataset["train_data_means"][fold][output_channels]
dataset_pred = (dataset_pred*dataset["train_data_stds"][fold][output_channels])+dataset["train_data_means"][fold][output_channels]
dataset_true = UTILS.unstride_array(dataset_true)
dataset_pred = UTILS.unstride_array(dataset_pred)
dataset_true = dataset_true[:, stride:]
dataset_pred = dataset_pred[:, stride:]

print(dataset_true.shape)

per_sample_RMSE = UTILS.per_sample_RMSE(dataset_true, dataset_pred, unknown)
_, per_node_RMSE = UTILS.per_node_RMSE(dataset_true, dataset_pred, unknown)
per_sample_R2 = UTILS.per_sample_R2(dataset_true, dataset_pred, unknown)
_, per_node_R2 = UTILS.per_node_R2(dataset_true, dataset_pred, unknown)
print("Per Sample RMSE:{:.4f}\tPer Node RMSE:{:.4f}\tPer Sample R2:{:.4f}\tPer Node R2:{:.4f}".format(per_sample_RMSE, per_node_RMSE, per_sample_R2, per_node_R2))
np.save(os.path.join(RESULT_DIR, "{}_{}_{}".format(file, index, "true_{}".format(fold))), dataset_true)
np.save(os.path.join(RESULT_DIR, "{}_{}_{}".format(file, index, model_name)), dataset_pred)

