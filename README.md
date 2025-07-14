# MLMC-DFM: Multilevel Monte Carlo Method for Discrete-Fracture Matrix Models

This repository provides a pipeline for numerical homogenization of 3D Discrete Fracture Matrix (DFM) models and their convolutional neural network (CNN)-based surrogates. **Integration of surrogates into the Multilevel Monte Carlo (MLMC) framework is currently under development and not yet supported.**

---

## 🔧 Features

This repository consists of three components:

1. **Dataset Generation**  
   - Performs numerical homogenization 
   - Rasterizes bulk and fracture data  
   - Creates a Zarr-formatted dataset

2. **Surrogate Training**  
   - Preprocesses the dataset  
   - Trains a 3D CNN to predict an equivalent hydraulic conductivity tensor from rasterized inputs

3. **Surrogate Postprocessing**  
   - Applies trained models for prediction on a given dataset
   - Provides visualization tools and evaluation scripts

Each part can be run independently, using the provided data.

---

## 🛠 Installation & Requirements

- Developed and tested using **Python 3.8**
- Each pipeline component has its own dependency file:

| Component              | Requirements File                  |
|------------------------|------------------------------------|
| Dataset Generation     | [`requirements_data_generation.txt`](requirements_data_generation.txt)      |
| Surrogate Training     | [`requirements_training.txt`](requirements_training.txt)        |
| Surrogate Postprocessing | [`requirements_postprocess.txt`](requirements_postprocess.txt)    |

We recommend creating separate virtual environments for each part, depending on your compute environment.

#### Set up Python environment:

```bash
cd MLMC-DFM
export PYTHONPATH=.
```


---

## 📦 Dataset Generation

> **Prerequisite:** Ensure that both [Flow123d](https://flow123d.github.io/) and [GMSH](https://gmsh.info/) are installed and accessible from the command line.

To generate datasets as we did for our experiments (numerical homogenization, rasterization, Zarr formatting), run

```bash
python mlmc_dfm_3d.py run work_dir scratch_dir
```
- `work_dir`: Working directory (e.g. `test/01_cond_field` - has to contains simulation config)
- `scratch_dir`: Fast scratch directory (set to `""` - if not applicable or available)

> The paths to Flow123d and GMSH executables are configured inside the `set_environment_variables()` method in `mlmc_dfm_3d.py`.

---

## 🧠 Surrogate Training

To train the surrogate run:

```bash
python metamodel/cnn3D/models/train_cnn_optuna_3d.py configuration data_dir results_dir -c
```

- `configuration` (e.g. [`configs/cnn_3D/final_test_config.yaml`](configs/cnn_3D/final_test_config.yaml))
- `data_dir`: Path to the dataset (Zarr format - e.g. `data/samples_data_to_test.zarr` - small dataset (~1,000 samples))  
- `results_dir`: Where results and logs will be saved
- `-c`: Use GPU (CUDA or AMD ROCm) if available

### Full Training Setup

For full-scale training on 60,000 samples (22GB+), see:

- Config file:  
  [`configs/cnn_3D/configs_lumi/final_config.yaml`](configs/cnn_3D/configs_lumi/final_config.yaml)

- Slurm submission script for HPC/GPU training (e.g., on LUMI):  
  [`slurm_submit_gpu_sing_small.sh`](slurm_submit_gpu_sing_small.sh)

> The training dataset is large and can be provided upon **reasonable request**.

---

## 🔍 Surrogate Postprocessing

Trained surrogates can be used to make predictions on new datasets or analyze its performance:

```bash
python metamodel/cnn3D/postprocess/optuna_results.py \
  results_dir data_dir
```

- `results_dir`: Directory containing trained model (e.g., `optuna_runs/3D_cnn/lumi/cond_frac_1_3/trained_surrogate`)
- `data_dir`: Path to the evaluation dataset (in Zarr format - e.g. `data/samples_data_to_test.zarr`)

Provided trained surrogates for fracture-to-matrix hydraulic conductivity ratios `Kₓ/Kₘ ∈ {10³, 10⁵, 10⁷}`:
- `optuna_runs/3D_cnn/lumi/cond_frac_1_3/trained_surrogate/` for `Kₓ/Kₘ = 10³`
- `optuna_runs/3D_cnn/lumi/cond_frac_1_5/trained_surrogate/` for `Kₓ/Kₘ = 10⁵`
- `optuna_runs/3D_cnn/lumi/cond_frac_1_7/trained_surrogate/` for `Kₓ/Kₘ = 10⁷`

**Note:** Due to limited consecutive training time on our devices, the model was trained in multiple sessions by resuming from saved checkpoints to reach the desired number of epochs. As a result, the training metrics (e.g., loss curves) may not represent complete history over all epochs for presented trained surrogates.