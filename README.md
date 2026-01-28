# TrajGuide: Trajectory-level property control for structure-based 3D ligand generation

This repository provides the reference implementation for our manuscript on **structure-based drug design (SBDD)** using a **Bayesian Flow Network (BFN)** backbone and a **geometry-aware conditional guidance** module for property control.

Our objective is to generate **3D ligands conditioned on protein pockets** while enabling controllable alignment to target molecular properties such as **QED** and **SA**, without sacrificing docking and geometric fidelity.

---

## Highlights

- Pocket-conditioned **BFN backbone** for 3D ligand generation  
- **Two-stage** training design: structural backbone and property guider are decoupled  
- Sampling-time **likelihood-based correction** for controllable property alignment  
- Evaluation on **CrossDock (ID)** and **PoseBusters (OOD)** protocols

---

## Method overview

Our framework follows a **two-stage design** that improves modularity and reproducibility:

- **Stage I: Pocket-conditioned BFN backbone**  
  Train a pocket-conditioned BFN generator to model ligand atom types and 3D coordinates under receptor constraints.  
  The backbone focuses on structural generation and does not require explicit property supervision in the default setting.

- **Stage II: Geometry-aware guidance network**  
  Construct a supervision dataset from intermediate noisy states and pocket context, then train a lightweight guidance model to predict property distributions from these intermediate states.

- **Inference: Conditional sampling with likelihood-based correction**  
  During reverse sampling, apply guidance derived from the learned property likelihood to steer intermediate states toward the target condition vector, while keeping the backbone unchanged.

This decoupling allows property control to be added or modified without retraining the backbone.

---

## Repository structure

- `train_bfn_twisted.py`  
  Backbone training and sampling entry point.
- `generate_condition_guidance_dataset.py`  
  Generates intermediate-state supervision for guidance training.
- `train_geometric_guidance_network.py`  
  Trains the geometry-aware guidance network.
- `configs/`  
  Experiment configurations for CrossDock (ID) and PoseBusters (OOD).
- `data/`  
  Expected location for processed datasets (paths configurable).
- `utils/`, `core/`, `models/`  
  Common modules used by training and inference.

---

## Environment setup

We recommend Linux with CUDA-enabled PyTorch.

### Install with `requirements.txt`

```bash
pip install -r requirements.txt
```


## Data preparation
The CrossDocked2020 dataset is publicly available at https://bits.csb.pitt.edu/files/crossdock2020/. 
The PoseBusters benchmark set is publicly available at https://github.com/maabuu/posebusters. 
All data used in this study were obtained from these resources and processed as described in the Methods.
### CrossDock (in-distribution)

Expected inputs (relative paths can be adjusted in configs):

- Processed LMDB  
  `data/crossdocked_v1.1_rmsd1.0_pocket10_processed_kekulize.lmdb`
- Split file  
  `data/crossdocked_pose_split_kekulize.pt`
- (Optional, for docking evaluation) test receptors  
  `data/test_set/`

### PoseBusters (out-of-distribution)

For OOD evaluation, we use a filtered PoseBusters benchmark set where complexes are removed if any receptor chain has >30% sequence identity to any chain in CrossDock training (MMseqs2-based filtering). The resulting subset is used only for evaluation.

> Please obtain datasets from their official sources and comply with their licenses.

---

## Reproducing the pipeline

All commands below assume you are in the repository root.

### Stage I: Train the BFN backbone (default, no property supervision)

```bash
python train_bfn_twisted.py \
  --config_file configs/crossdock_train_test_condition_aware.yaml \
  --sigma1_coord 0.05 --beta1 1.5 --beta1_bond 1.5 \
  --lr 5e-4 --self_condition \
  --epochs 30 --batch_size 16 \
  --destination_prediction True --use_discrete_t True \
  --num_samples 10 --sampling_strategy end_back_pmf --sample_num_atoms ref \
  --ligand_atom_mode add_aromatic \
  --gpu_device 0
```

Outputs are saved under `outputs/` (the log prints the run directory).

---

### Stage II: Build guidance dataset and train the guidance network

#### (1) Generate guidance dataset

```bash
python generate_condition_guidance_dataset.py \
  --config configs/crossdock_train_test_condition_aware.yaml \
  --output_dir data/condition_guidance_dataset
```

Outputs:
- `data/condition_guidance_dataset/{train,val,test}_condition_data.pkl`
- `data/condition_guidance_dataset/dataset_statistics.pkl`

#### (2) Train the geometry-aware guidance network

```bash
python train_geometric_guidance_network.py \
  --data_dir data/condition_guidance_dataset \
  --output_dir models/geometric_guidance \
  --config configs/crossdock_train_test_condition_aware.yaml \
  --backbone_ckpt <PATH_TO_BACKBONE_CKPT> \
  --epochs 20 --batch_size 16 --lr 1e-4
```

Output:
- `models/geometric_guidance/geometric_guidance_best.pt`

---

### Inference: Conditional sampling with guidance

```bash
python train_bfn_twisted.py \
  --config_file configs/crossdock_train_test_condition_aware.yaml \
  --ckpt_path <PATH_TO_BACKBONE_CKPT> \
  --test_only --exp_name guided_sampling --revision v1 \
  --eval_batch_size 16 \
  --guidance_model_path models/geometric_guidance/geometric_guidance_best.pt \
  --guidance_scale 2.0 \
  --target_qed 0.56 --target_sa 0.78 \
  --use_gradient_guidance \
  --gpu_device 0
```

Notes:
- `--guidance_scale` controls the strength of the likelihood correction.
- The guidance implementation uses a gradient-based form for likelihood correction during sampling. The manuscript provides the probabilistic formulation and its implementation rationale.

---

## Optional: condition-aware backbone (ablation)

The default manuscript setting does not require injecting property targets during backbone training.  
If you want to run an ablation where property targets are provided during Stage I, enable:

```bash
--condition_aware --condition_dim 2 --condition_use_prob 1 --condition_noise_std 0.00 \
--target_qed <...> --target_sa <...>
```

We recommend keeping this option disabled for the main results to match the two-stage decoupled setting.

---

## Reproducibility checklist

- Fix random seeds and log them per run.
- Save the config file and command line arguments for each experiment.
- Do not use PoseBusters OOD data for training or guidance dataset construction.
- Record docking tool versions and receptor preparation protocols when reporting docking metrics.

---

## License

See `LICENSE`.

---

## Acknowledgements

This codebase follows conventions from the MolPilot and TargetDiff ecosystems for structure-based data processing and evaluation. We thank the original authors for their open-source contributions.

---


