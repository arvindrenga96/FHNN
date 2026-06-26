#!/bin/bash -l

###############################

# SETUP RESOURCE
#SBATCH --time=2:00:00
#SBATCH --ntasks=1
#SBATCH --mem=250gb
#SBATCH --mail-type=ALL
#SBATCH --mail-user=renga016@umn.edu
#SBATCH -p kgml02
#SBATCH --gres=gpu:1
#SBATCH --output=../../../../DATA/camels_us_531/FHNN/RESULT/20250207/tft_4_0.txt

###############################

# FIXED COMMANDS
eval "$(conda shell.bash hook)"
conda activate main_a100
cd ../
NOTEBOOK="MODEL"
jupyter nbconvert --to script $NOTEBOOK.ipynb

# ###############################

# CONFIGURE CONFIG
DATASET="camels_us_531"
PROJECT="FHNN"
forward_code_dim=128
latent_code_dim=128
device="cuda"
dropout=0.4
train="False"
batch_size=128
epochs=50
learning_rate=1e-3
fold=0
init=4
forecast=7

sed -i'' 's/DATASET = .*/DATASET = \"'$DATASET'\"/g' config.py 
sed -i'' 's/PROJECT = .*/PROJECT = \"'$PROJECT'\"/g' config.py 
sed -i'' 's/forward_code_dim = .*/forward_code_dim = '$forward_code_dim'/g' config.py
sed -i'' 's/latent_code_dim = .*/latent_code_dim = '$latent_code_dim'/g' config.py
sed -i'' 's/device = .*/device = \"'$device'\"/g' config.py
sed -i'' 's/dropout = .*/dropout = '$dropout'/g' config.py
sed -i'' 's/train = .*/train = '$train'/g' config.py
sed -i'' 's/batch_size = .*/batch_size = '$batch_size'/g' config.py
sed -i'' 's/epochs = .*/epochs = '$epochs'/g' config.py
sed -i'' 's/learning_rate = .*/learning_rate = '$learning_rate'/g' config.py
sed -i'' 's/datasets = .*/datasets = '$datasets'/g' config.py
sed -i'' 's/forecast = .*/forecast = '$forecast'/g' config.py

###############################

# RUN CODE
DATE="20250207"
cd MODELS
ARCHITECTURE="tft"
NOTEBOOK=$ARCHITECTURE
jupyter nbconvert --to script $NOTEBOOK.ipynb
model_name=$ARCHITECTURE"_"$forward_code_dim"_"$batch_size"_"$epochs"_"$learning_rate"_"$init"_"$fold

python -u $NOTEBOOK.py --init $init --fold $fold --model_name $model_name --date $DATE
rm $NOTEBOOK.py
###############################