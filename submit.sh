#!/bin/bash

py_script=`pwd`/$1
script_path=${py_script%/*}
data_dir=$2

export PYTHONPATH="${PYTHONPATH}:/home/martin_spetlik/MLMC-DFM"
ml PyTorch

python3 -m venv venv --clear
source venv/bin/activate
which python3

/home/martin_spetlik/MLMC-DFM/setup.sh

which python3


cd ${script_path}
python3 ${py_script} ${data_dir}

deactivate
