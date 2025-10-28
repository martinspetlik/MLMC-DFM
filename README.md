# MLMC-DFM: Multilevel Monte Carlo Method for Discrete-Fracture Matrix Models

This repository provides a complete pipeline for numerical homogenization of 3D Discrete Fracture Matrix (DFM) models, training convolutional neural network (CNN)-based surrogates, and **integrating these surrogates into a Multilevel Monte Carlo (MLMC) framework** for efficient uncertainty quantification.

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

4. **MLMC run with Surrogates**  
   - Uses trained CNN surrogates for upscaling hydraulic conductivity within MLMC levels

5. **MLMC postprocessing**  
   - MLMC postprocessing using mainly mlmc library, includes mean and variance estimation of derived quantities, diagnostics plots, ...



Each part can be run independently, using the provided data.

---

## 🛠 Installation & Requirements

- Developed and tested using **Python 3.8** and **Python 3.10**
- Each pipeline component has its own dependency file:

| Component                | Requirements File                                                        |
|--------------------------|--------------------------------------------------------------------------|
| Dataset Generation       | [`requirements_data_generation.txt`](requirements_data_generation.txt)   |
| Surrogate Training       | [`requirements_training.txt`](requirements_training.txt)                 |
| Surrogate Postprocessing | [`requirements_postprocess.txt`](requirements_postprocess.txt)           |
| MLMC run with Surrogates | [`requirements_data_generation.txt`](requirements_data_generation.txt)   |
| MLMC postprocessing      | [`requirements_mlmc_postprocess.txt`](requirements_mlmc_postprocess.txt) |


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
- `work_dir`: Working directory (e.g. `test/01_cond_field` - has to contain simulation config - similar to [`test/01_cond_field/sim_config_3D_homogenization_samples.yaml`](test/01_cond_field/sim_config_3D_homogenization_samples.yaml)
- `scratch_dir`: Fast scratch directory (set to `""` - if not applicable or available)

> The paths to Flow123d and GMSH executables are configured inside the `set_environment_variables()` method in `mlmc_dfm_3d.py`.
> Use `self.n_levels = 1` in mlmc_dfm_3d.py

---

## 🧠 Surrogate Training

To train the surrogate run:

```bash
python metamodel/cnn3D/models/train_cnn_optuna_3d.py configuration data_dir results_dir -c
```

- `configuration` (e.g. [`configs/cnn_3D/final_test_config.yaml`](configs/cnn_3D/final_test_config.yaml))
- `data_dir`: Path to the dataset (Zarr format - e.g., `data/samples_data_to_test.zarr` - small dataset (~1,000 samples))  
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

Trained surrogates can be used to make predictions on new datasets or analyze their performance:

```bash
python metamodel/cnn3D/postprocess/optuna_results.py results_dir data_dir
```

- `results_dir`: Directory containing trained model (e.g., `optuna_runs/3D_cnn/lumi/cond_frac_1_3/trained_surrogate`)
- `data_dir`: Path to the evaluation dataset (in Zarr format - e.g., `data/samples_data_to_test.zarr`)

We provide compressed trained surrogates for fracture-to-matrix hydraulic conductivity ratios `Kₓ/Kₘ ∈ {10³, 10⁵, 10⁷}`:
- `optuna_runs/3D_cnn/lumi/cond_frac_1_3/trained_surrogate.zip` for `Kₓ/Kₘ = 10³`
- `optuna_runs/3D_cnn/lumi/cond_frac_1_5/trained_surrogate.zip` for `Kₓ/Kₘ = 10⁵`
- `optuna_runs/3D_cnn/lumi/cond_frac_1_7/trained_surrogate.zip` for `Kₓ/Kₘ = 10⁷`

**Note:** Due to limited consecutive training time on our devices, the model was trained in multiple sessions by resuming from saved checkpoints to reach the desired number of epochs. As a result, the training metrics (e.g., loss curves) may not represent the complete history over all epochs for the presented trained surrogates.


## 📦 MLMC run with surrogate

> **Prerequisites:**  
> Ensure that both [Flow123d](https://flow123d.github.io/) and [Gmsh](https://gmsh.info/) are installed and accessible from the command line.

Running the **MLMC (Multilevel Monte Carlo)** simulation with a surrogate model is very similar to generating datasets.  
The main difference lies in the configuration file — you must set the following parameters:

- `generate_hom_samples: false`  
- Provide the surrogate model path using `nn_path` or `nn_path_cond_frac`.

For an example configuration file, see:  
[`test/01_cond_field/sim_config_3D_MC_samples.yaml`](test/01_cond_field/sim_config_3D_MC_samples.yaml).  
The path to the simulation configuration file is currently set in the `setup_config()` method inside `mlmc_dfm_3d.py`.

Use the following command to start the MLMC simulation:

```bash
python mlmc_dfm_3d.py run work_dir scratch_dir
```
- `work_dir`: Path to the working directory (e.g. `test/01_cond_field`). This directory must contain a valid simulation configuration file, such as [`test/01_cond_field/sim_config_3D_MC_samples.yaml`](test/01_cond_field/sim_config_3D_MC_samples.yaml).
- `scratch_dir`: Path to a fast scratch directory for temporary files. Use an empty string "" if not applicable.

> The paths to Flow123d and GMSH executables are configured inside the `set_environment_variables()` method in `mlmc_dfm_3d.py`.

---

## 📊 MLMC Postprocessing

Once the MLMC simulation has completed, you can run the postprocessing step to analyze and visualize the results.

```bash
python postprocess_mlmc_dfm_3d.py work_dir
```
- `work_dir`: Path to the working directory (e.g. `test/01_cond_field`). This directory must contain the output file `mlmc_l.hdf5` file, where `l` corresponds
to the number of levels defined by the parameter `self.n_levels` set in `mlmc_dfm_3d.py`.

---