# Two-Stage Conditional Guidance for Structure-Based Molecule Generation (MolPilot-based)

This repository provides the research code for a **two-stage conditional guidance** pipeline for structure-based drug design (SBDD), built on top of **MolPilot** (a Bayesian Flow Network–based SBDD framework). The pipeline targets **property-controlled generation** (e.g., **QED** and **SA**) while maintaining 3D pocket–ligand consistency.

The implementation used for the four-stage workflow described below lives in `MolPilot/`.

## Method Overview (Four Stages)

1. **Condition-aware backbone training**: train a BFN backbone with conditioning signals (QED/SA) injected during training.
2. **Guidance dataset generation**: sample intermediate states from the backbone and compute target properties to build a supervised dataset.
3. **Geometric guidance network training**: train a geometry-aware network to predict the conditional distribution of target properties given intermediate states.
4. **Joint inference (guided sampling)**: run backbone sampling with the trained guidance model (gradient guidance) to steer generation toward target properties.

## Environment

We recommend using the environment provided by MolPilot:
- Docker setup: `MolPilot/docker/README.md`
- Dependency list: `MolPilot/docker/asset/requirements.txt`

## Data

Experiments are configured for **CrossDocked v1.1 (pocket10)**. MolPilot provides processed artifacts (LMDB + split file) and detailed preprocessing instructions in `MolPilot/README.md`.

Expected paths (relative to `MolPilot/`):
- Processed LMDB: `data/crossdocked_v1.1_rmsd1.0_pocket10_processed_kekulize.lmdb`
- Split file: `data/crossdocked_pose_split_kekulize.pt`
- (Optional, docking evaluation) Test set proteins: `data/test_set/`

## Reproducing the Four Stages

Run from `MolPilot/`:

```bash
cd MolPilot
```

### Stage 1: Condition-aware backbone training

```bash
python train_bfn_twisted.py \
  --config_file configs/crossdock_train_test_condition_aware.yaml \
  --sigma1_coord 0.05 --beta1 1.5 --beta1_bond 1.5 \
  --lr 5e-4 --time_emb_dim 1 --self_condition \
  --epochs 30 --batch_size 16 --max_grad_norm Q --scheduler plateau \
  --destination_prediction True --use_discrete_t True \
  --num_samples 10 --sampling_strategy end_back_pmf --sample_num_atoms ref \
  --ligand_atom_mode add_aromatic \
  --condition_aware --condition_dim 2 --condition_use_prob 1 --condition_noise_std 0.00 \
  --target_qed 0.56 --target_sa 0.78 \
  --gpu_device 0
```

Outputs (default): `MolPilot/outputs/condition_aware_integration/v1/`

### Stage 2: Guidance dataset generation and Geometric guidance network training

```bash
python generate_condition_guidance_dataset.py \
  --config configs/crossdock_train_test_condition_aware.yaml \
  --output_dir data/condition_guidance_dataset
```

Outputs: `data/condition_guidance_dataset/{train,val,test}_condition_data.pkl` and `data/condition_guidance_dataset/dataset_statistics.pkl`

```bash
python train_geometric_guidance_network.py \
  --data_dir data/condition_guidance_dataset \
  --output_dir models/geometric_guidance \
  --config configs/crossdock_train_test_condition_aware.yaml \
  --backbone_ckpt <PATH_TO_BACKBONE_CKPT> \
  --epochs 20 --batch_size 16 --lr 1e-4
```

Outputs: `models/geometric_guidance/geometric_guidance_best.pt`

### Stage 3: Joint inference (guided sampling)

```bash
python train_bfn_twisted.py \
  --config_file configs/crossdock_train_test_condition_aware.yaml \
  --ckpt_path <PATH_TO_BACKBONE_CKPT> \
  --test_only --exp_name guided_sampling --revision v1 \
  --eval_batch_size 16 \
  --condition_aware --guidance_scale 2.5 \
  --target_qed 0.60 --target_sa 0.78 \
  --guidance_model_path models/geometric_guidance/geometric_guidance_best.pt \
  --use_gradient_guidance \
  --gpu_device 0
```

## Key Implementation Files

Entry points:
- `MolPilot/train_bfn_twisted.py` (stage 1 & 3)
- `MolPilot/generate_condition_guidance_dataset.py` (stage 2)
- `MolPilot/train_geometric_guidance_network.py` (stage 2)

Main configuration:
- `MolPilot/configs/crossdock_train_test_condition_aware.yaml`

## Acknowledgements

This project is built upon **MolPilot** (ICML 2025) and its ecosystem (including TargetDiff preprocessing conventions). Please cite the original MolPilot paper when using this codebase.

## License

See `LICENSE`.

# SBDD-CGBFN
