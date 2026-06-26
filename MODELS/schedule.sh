#!/bin/bash -l

DATASET="camels_us_531"
PROJECT="FHNN"

################################
DATE="20250318"
email="renga@umn.edu" 
msi_partition="a100-4" # options {"a100-4", "v100", kgml01}
conda_env="main_a100" # change this

for file in "lstm_ar"
do
    for init in 5
    do
        for fold in 0
        do
            time="8:00:00"
            chmod +x $file.sh
            mkdir -p ../../../../DATA/$DATASET/$PROJECT/RESULT/$DATE
            mkdir -p ../../../../DATA/$DATASET/$PROJECT/MODEL/$DATE
            sed -i'' 's/\#SBATCH --time=.*/#SBATCH --time='$time'/g' $file.sh
            sed -i'' 's/\#SBATCH --mail-user=.*/#SBATCH --mail-user='$email'/g' $file.sh
            sed -i'' 's/\#SBATCH -p .*/#SBATCH -p '$msi_partition'/g' $file.sh
            sed -i'' 's/\#SBATCH --output=.*/#SBATCH --output=..\/..\/..\/..\/DATA\/'$DATASET'\/'$PROJECT'\/RESULT\/'$DATE'\/'$file'_'$init'_'$fold'.txt/g' $file.sh
            sed -i'' 's/conda activate .*/conda activate '$conda_env'/g' $file.sh
            sed -i'' 's/DATASET=.*/DATASET=\"'$DATASET'\"/g' $file.sh
            sed -i'' 's/PROJECT=.*/PROJECT=\"'$PROJECT'\"/g' $file.sh
            sed -i'' 's/DATE=.*/DATE=\"'$DATE'\"/g' $file.sh
            sed -i'' 's/ARCHITECTURE=.*/ARCHITECTURE=\"'$file'\"/g' $file.sh
            sed -i'' 's/init=.*/init='$init'/g' $file.sh
            sed -i'' 's/fold=.*/fold='$fold'/g' $file.sh
            
#             ./$file.sh
            sbatch $file.sh
            sleep 5s
        done
    done
done
