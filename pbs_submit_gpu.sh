#!/bin/bash

set -x

py_script=`pwd`/$1
pbs_script=`pwd`/$1.pbs


trials_config_path=$2
data_dir=$3
output_dir=$4
name=$5


cp ${py_script} ${output_dir}
cp ${pbs_script} ${output_dir}


py_script_name=`basename ${py_script}`
pbs_script_name=`basename ${pbs_script}`


py_script=${output_dir}/${py_script_name}
pbs_script=${output_dir}/${pbs_script_name}
script_path=${py_script%/*}


echo ${py_script}
echo ${pbs_script}
echo ${script_path}



cat >$pbs_script <<EOF
#!/bin/bash
#PBS -S /bin/bash
#PBS -A OPEN-27-48
#PBS -l select=1:ngpus=1
#PBS -l walltime=48:00:00
#PBS -q qgpu
#PBS -N ${name}
#PBS -j oe


#export PYTHONPATH="${PYTHONPATH}:/home/martin_spetlik/MLMC-DFM"
ml PyTorch
#pip3 install typing_extensions==4.3.0

cd ${output_dir}
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

#cd ${script_path}
python3 ${py_script} ${trials_config_path} ${data_dir} ${output_dir} -c

deactivate
EOF

qsub $pbs_script
