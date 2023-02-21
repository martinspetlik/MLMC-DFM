#!/bin/bash

set -x

py_script=`pwd`/$1
pbs_script=`pwd`/$1.pbs
script_path=${py_script%/*}

data_dir=$2

cat >$pbs_script <<EOF
#!/bin/bash
#PBS -S /bin/bash
#PBS -A OPEN-27-48
#PBS -l select=1:mem=8Gb
#PBS -l walltime=1:00:00
#PBS -q qcpu
#PBS -N PyTorch_test
#PBS -j oe


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
EOF

qsub $pbs_script