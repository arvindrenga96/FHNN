#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import numpy as np

#DATASET INFO
DATASET = "camels_us_531"
folds=5

PROJECT = "FHNN"

# FILES INFO
DATA_DIR = os.path.join("..", "..", "..", "..", "DATA")
RAW_DIR = os.path.join("/", "home", "kumarv", "renga", "Public", "DATA", "{}".format(DATASET), "RAW")
PREPROCESSED_DIR = os.path.join(DATA_DIR, "{}".format(DATASET), "{}".format(PROJECT),"PREPROCESSED")
RESULT_DIR = os.path.join(DATA_DIR, "{}".format(DATASET), "{}".format(PROJECT),"RESULT")
MODEL_DIR = os.path.join(DATA_DIR, "{}".format(DATASET), "{}".format(PROJECT),"MODEL")

if not os.path.exists(PREPROCESSED_DIR):
	os.makedirs(PREPROCESSED_DIR)
if not os.path.exists(RESULT_DIR):
	os.makedirs(RESULT_DIR)
if not os.path.exists(MODEL_DIR):
	os.makedirs(MODEL_DIR)

# TIME SERIES INFO
# train_year = {"start":1989, "end":1999}
# valid_year = {"start":1999, "end":2001}
# test_year = {"start":2001, "end":2009}
#chaopeng shen
train_year = {"start":1985, "end":1993}
valid_year = {"start":1993, "end":1995}
test_year = {"start":1995, "end":2005}

window = 365
context = 365
forecast = 1
stride = 1 #window//2
adapt_year = 2

# CHANNELS INFO
channels_names = np.array([
	"p_mean", "pet_mean", "p_seasonality", "frac_snow", "aridity", "high_prec_freq", "high_prec_dur", "low_prec_freq", "low_prec_dur",																#Static Features
	"carbonate_rocks_frac", "geol_permeability", "soil_depth_pelletier", "soil_depth_statsgo", "soil_porosity", "soil_conductivity", "max_water_content", "sand_frac", "silt_frac", "clay_frac",	#Static Features
	"elev_mean", "slope_mean", "area_gages2", "frac_forest", "lai_max", "lai_diff", "gvf_max", "gvf_diff",																							#Static Features
	"PRCP(mm/day)", "SRAD(W/m2)", "Tmax(C)", "Tmin(C)", "Vp(Pa)",																																	#Dynamic Features
	"SF"																																															#StreamFlow
])

channels = list(range(len(channels_names)))
static_channels = channels[:27]
dynamic_channels = channels[27:32]
output_channels = [channels[-1]]


# LABELS INFO
add = 0.005
unknown = -999

# MODEL INFO
forward_code_dim = 255
latent_code_dim = 85
device = "cuda"
recon_weight = 1.0
contrastive_weight = 1.0
static_weight = 0.0
kl_weight = 1.0
forward_weight = 1.0		# KGSSL, VAE
dropout = 0.4

# TRAIN INFO
inits = 5
train = True
batch_size = 128
epochs = 50
learning_rate = 1e-3
meta_learning_rate = 1e-3

# INFERENCE INFO
runs = 100