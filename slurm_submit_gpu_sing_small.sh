#!/bin/bash

#set -x

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
#SBATCH --job-name ${name}
#SBATCH --account project_465002075
#SBATCH --time 72:00:00
##SBATCH --nodes=1
##SBATCH --ntasks=1
##SBATCH --gpus-per-task=1
#SBATCH --gpus-per-node=1
#SBATCH --partition small-g
#SBATCH --mem=24G
#SBATCH --output=${name}.out
#SBATCH --error=${name}.err

# Make sure GPUs are up

rocm-smi


cd ${output_dir}

#module load CrayEnv
#module load PyTorch/2.2.0-rocm-5.6.1-python-3.10-singularity-20240315

export EBU_USER_PREFIX=/scratch/project_465002075/EasyBuild

module load LUMI/24.03
module load partition/L
module load PyTorch/2.4.1-rocm-6.1.3-python-3.12-singularity-20241007

#export ROCR_VISIBLE_DEVICES=$SLURM_LOCALID

echo ${PYTHONPATH}

export PYTHONPATH="${PYTHONPATH}:/users/petlikma/MLMC-DFM_3D"


singularity exec -B ${output_dir} -B ${data_dir} -B /users/petlikma/MLMC-DFM_3D $SIF python ${py_script} ${trials_config_path} ${data_dir} ${output_dir} -c


EOF

sbatch $pbs_script

