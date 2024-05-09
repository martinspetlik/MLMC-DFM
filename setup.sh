#!/bin/bash

export PYTHONPATH=venv/bin/activate
export PYTHONPATH="${PYTHONPATH}:/home/martin_spetlik/MLMC-DFM"

python3 -m pip install -U setuptools wheel pip
python3 -m pip install typing-extensions
python3 -m pip install optuna
python3 -m pip install torchvision==0.12.0
python3 -m pip install tensorboard==2.11.0
python3 -m pip install tensorboardX
python3 -m pip install joblib
python3 -m pip install pandas


