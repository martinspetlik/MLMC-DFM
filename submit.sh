#!/bin/bash

py_script=`pwd`/$1
script_path=${py_script%/*}
data_dir=$2
output_dir=$3

#export PYTHONPATH="${PYTHONPATH}:/home/martin_spetlik/MLMC-DFM"
ml PyTorch
#pip3 install typing_extensions==4.3.0


python3 -m venv venv --clear
source venv/bin/activate
#which python3

export PYTHONPATH=/home/martin_spetlik/MLMC-DFM/venv/bin/activate
export PYTHONPATH="${PYTHONPATH}:/home/martin_spetlik/MLMC-DFM"

python3 -m pip install -U setuptools wheel pip
python3 -m pip install -r /home/martin_spetlik/MLMC-DFM/requirements.txt
#-r /home/martin_spetlik/MLMC-DFM/requirements.txt

#/home/martin_spetlik/MLMC-DFM/setup.sh

#which python3

cd ${script_path}
python3 ${py_script} ${data_dir} ${output_dir}

deactivate
