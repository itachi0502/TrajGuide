import argparse
import os
import shutil
import resource
import gc
import signal
import sys
import atexit
import re
from contextlib import contextmanager

import torch
import numpy as np

from sklearn.metrics import roc_auc_score
from torch_geometric.loader import DataLoader
from torch_geometric.transforms import Compose

import datetime, pytz
import wandb

# Initialize NumPy compatibility early

from core.utils.numpy_compat import initialize_numpy_compatibility
initialize_numpy_compatibility()



# SOTA Resource Management System
try:
    from core.utils.resource_manager import (
        ResourceMonitor,
        FileHandleManager,
        safe_file_operation,
        safe_temp_file
    )
    RESOURCE_MANAGEMENT_AVAILABLE = True
except ImportError:
    RESOURCE_MANAGEMENT_AVAILABLE = False

from core.config.config import Config, parse_config, Struct
import subprocess
import time
from core.models.sbdd4train import SBDD4Train
from core.callbacks.basic import RecoverCallback, GradientClip, NormalizerCallback #, EMACallback
from core.callbacks.ema import EMACallback
from core.callbacks.validation_callback import (
    CondMolGenValidationCallback,
    MolVisualizationCallback,
    TwistedReconValidationCallback,
    DockingTestCallback,
)

import core.utils.transforms as trans
from core.datasets import get_dataset
from core.datasets.pl_data import FOLLOW_BATCH

import pytorch_lightning as pl

# SOTA Resource Management Configuration
def setup_resource_management():
    """Setup comprehensive resource management for MolPilot training"""

    # 1. Increase file descriptor limits

    soft, hard = resource.getrlimit(resource.RLIMIT_NOFILE)
    # Set to 80% of hard limit or 16384, whichever is smaller
    new_soft = min(int(hard * 0.8), 16384)
    if new_soft > soft:
        resource.setrlimit(resource.RLIMIT_NOFILE, (new_soft, hard))



    # 2. Setup emergency cleanup
    def emergency_cleanup(signum=None, frame=None):

        # Force garbage collection
        gc.collect()

        # Clear CUDA cache if available
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        # Close wandb if active
        try:
            if wandb.run is not None:
                wandb.finish()
        except:
            pass



        if signum is not None:
            sys.exit(1)

    # Register signal handlers
    signal.signal(signal.SIGTERM, emergency_cleanup)
    signal.signal(signal.SIGINT, emergency_cleanup)
    atexit.register(emergency_cleanup)

    # 3. Initialize resource monitor if available
    if RESOURCE_MANAGEMENT_AVAILABLE:
        global resource_monitor
        resource_monitor = ResourceMonitor(warning_threshold=0.7, critical_threshold=0.9)
        return resource_monitor

    return None

@contextmanager
def safe_dataloader_context(num_workers, docking_mode='vina_score'):
    """
    SOTA Context manager for safe DataLoader operations with docking-aware resource management

    Args:
        num_workers: Desired number of workers
        docking_mode: Docking mode ('vina_score', 'vina_dock') for resource adjustment
    """
    original_workers = num_workers

    # Adjust workers based on docking mode and system resources
    if RESOURCE_MANAGEMENT_AVAILABLE:
        if docking_mode == 'vina_dock':
            # Ultra-conservative for dock mode due to higher resource usage
            safe_workers = min(num_workers, 1) if num_workers > 0 else 0
        else:
            # Conservative approach for score mode
            safe_workers = min(num_workers, 2) if num_workers > 0 else 0
    else:
        # Even more conservative without resource management
        safe_workers = 0  # Force single-threaded mode

    try:
        yield safe_workers
    finally:

        # Force garbage collection multiple times for thorough cleanup
        for i in range(3):
            collected = gc.collect()
 

        # Clear CUDA cache if available
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        # Additional cleanup for resource manager
        if RESOURCE_MANAGEMENT_AVAILABLE:
            from core.utils.resource_manager import cleanup_resources
            cleanup_resources()


from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor, EarlyStopping
from pytorch_lightning import seed_everything
from pytorch_lightning.profilers import SimpleProfiler, PyTorchProfiler

from absl import logging
import glob
import pickle as pkl
from tqdm import tqdm


def get_dataloader(cfg):
    if cfg.data.name == 'pl_tr':
        dataset, subsets = get_dataset(config=cfg.data)
        train_set, val_set = subsets['train'], subsets['test']        
        cfg.dynamics.protein_atom_feature_dim = dataset.protein_atom_feature_dim
        cfg.dynamics.ligand_atom_feature_dim = dataset.ligand_atom_feature_dim
    else:
        protein_featurizer = trans.FeaturizeProteinAtom()
        ligand_featurizer = trans.FeaturizeLigandAtom(cfg.data.transform.ligand_atom_mode)
        transform_list = [
            protein_featurizer,
            ligand_featurizer,
            trans.FeaturizeLigandBond(),
        ]
        transform = Compose(transform_list)
        cfg.dynamics.protein_atom_feature_dim = protein_featurizer.feature_dim
        cfg.dynamics.ligand_atom_feature_dim = ligand_featurizer.feature_dim
        cfg.dynamics.ligand_atom_type_dim = ligand_featurizer.type_feature_dim
        cfg.dynamics.ligand_atom_charge_dim = ligand_featurizer.charge_feature_dim
        cfg.dynamics.ligand_atom_aromatic_dim = ligand_featurizer.aromatic_feature_dim


        condition_config = getattr(cfg.data, 'condition_aware', None)
        if condition_config and getattr(condition_config, 'enabled', False):
            from core.utils.condition_transforms import create_condition_aware_transform

            condition_aware_transform = create_condition_aware_transform(transform, condition_config)
            dataset, subsets = get_dataset(config=cfg.data, transform=condition_aware_transform)
            train_set, test_set = subsets['train'], subsets['test']

        else:
            dataset, subsets = get_dataset(config=cfg.data, transform=transform)
            train_set, test_set = subsets['train'], subsets['test']


    if 'val' in subsets and len(subsets['val']) > 0:
        val_set = subsets['val']
    elif 'valid' in subsets and len(subsets['valid']) > 0:
        val_set = subsets['valid']
    else:
        val_set = test_set

    if len(train_set) == 0:
        assert cfg.test_only, "No training data found"
        train_set = val_set

    if cfg.train.ckpt_freq > 1:
        # repeat train set for ckpt_freq times
        train_set = torch.utils.data.ConcatDataset([train_set] * cfg.train.ckpt_freq)
        cfg.train.ckpt_freq = 1

    # if using multiple GPUs, the batch size should be divided by the number of GPUs
    # and use DistributedSampler
    if torch.cuda.device_count() > 1:
        cfg.train.batch_size = cfg.train.batch_size // torch.cuda.device_count()

    dataset_smiles_set = compute_or_retrieve_dataset_smiles(train_set, cfg.data.smiles_path)

 
    collate_exclude_keys = ["ligand_nbh_list", "conditions", "reference_conditions"]
    # size-1 debug set
    if cfg.debug:
        # debug_id = 9618 # 5000 (29)
        # debug_id = 29008 # 4000 (75)
        debug_id = 0
        debug_set = torch.utils.data.Subset(test_set, [debug_id] * 100) #[0] * 1600)
        debug_set_val = torch.utils.data.Subset(test_set, list(range(10)))
        debug_batch_val = next(iter(DataLoader(debug_set_val, batch_size=cfg.train.batch_size, shuffle=False)))
        train_loader = DataLoader(debug_set,
            batch_size=cfg.train.batch_size,
            shuffle=False,  # set shuffle to False 
            num_workers=cfg.train.num_workers,
            follow_batch=FOLLOW_BATCH,
            exclude_keys=collate_exclude_keys
        )
        val_loader = DataLoader(
            debug_set_val, 
            batch_size=cfg.evaluation.batch_size,
            shuffle=False,
            follow_batch=FOLLOW_BATCH, 
            exclude_keys=collate_exclude_keys
        )
    else:
        logging.info(f"Training: {len(train_set)} Validation: {len(val_set)}")

        # SOTA: Use safe DataLoader configuration with docking-aware resource management
        docking_mode = cfg.evaluation.docking_config.mode if hasattr(cfg.evaluation, 'docking_config') else 'vina_score'

        with safe_dataloader_context(cfg.train.num_workers, docking_mode=docking_mode) as safe_workers:
            condition_config = getattr(cfg.data, 'condition_aware', None)


            train_loader = DataLoader(
                train_set,
                batch_size=cfg.train.batch_size,
                shuffle=True,
                num_workers=safe_workers,
                follow_batch=FOLLOW_BATCH,
                exclude_keys=collate_exclude_keys,
                pin_memory=True if safe_workers > 0 else False,
                persistent_workers=True if safe_workers > 0 else False,
                prefetch_factor=2 if safe_workers > 0 else None,
                # Use spawn method for better resource isolation
                multiprocessing_context='spawn' if safe_workers > 0 else None,
            )

            # Validation loader with minimal resource usage
            val_loader = DataLoader(
                val_set,
                batch_size=cfg.evaluation.batch_size,
                shuffle=False,
                num_workers=0,  # No workers for validation to save resources
                follow_batch=FOLLOW_BATCH,
                exclude_keys=collate_exclude_keys
            )
    cfg.train.scheduler.max_iters = cfg.train.epochs * len(train_loader)

    return train_loader, val_loader, dataset_smiles_set


def get_logger(cfg):
    """Get WandB logger with SOTA resource management"""
    os.makedirs(cfg.accounting.wandb_logdir, exist_ok=True)

    # SOTA: Configure WandB for minimal file descriptor usage
    wandb_config = {
        'project': cfg.project_name,
        'save_dir': cfg.accounting.wandb_logdir,
        'offline': cfg.no_wandb,
        # Reduce file operations
        'settings': {
            'disable_git': True,  # Disable git integration to reduce file ops
            'disable_code': True,  # Disable code saving to reduce file ops
            '_disable_stats': True,  # Disable system stats collection
            '_disable_meta': True,  # Disable metadata collection
        }
    }

    if cfg.wandb_resume_id is not None:
        wandb_logger = WandbLogger(
            id=cfg.wandb_resume_id,
            resume='must',
            **wandb_config
        )
    else: # start a new run
        wandb_logger = WandbLogger(
            name=f"{cfg.exp_name}_{str(cfg.revision)}"
            + f'_{datetime.datetime.now(pytz.timezone("Asia/Shanghai")).strftime("%Y-%m-%d-%H:%M:%S")}',
            **wandb_config
        )

    return wandb_logger

def compute_or_retrieve_dataset_smiles(
    dataset, save_path
):
    # create parent directory if it does not exist
    if save_path is None:
        return None
    if not os.path.exists(os.path.dirname(save_path)):
        os.makedirs(os.path.dirname(save_path))
    if not os.path.exists(save_path):
        all_smiles = []
        for data in tqdm(dataset, desc="SMILES"):
            all_smiles.append(data.ligand_smiles)
        with open(save_path, "wb") as f:
            pkl.dump(all_smiles, f)
    else:
        with open(save_path, "rb") as f:
            all_smiles = pkl.load(f)
    smiles_set = set([s for s in all_smiles])
    return smiles_set


if __name__ == "__main__":
    # SOTA: Initialize resource management FIRST
    resource_monitor = setup_resource_management()

    parser = argparse.ArgumentParser()

    # meta
    parser.add_argument("--config_file", type=str, default="configs/crossdock_train_test.yaml",)
    parser.add_argument("--exp_name", type=str, default="debug")
    parser.add_argument("--revision", type=str, default="default")
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--wandb_resume_id", type=str, default=None)
    parser.add_argument('--empty_folder', action='store_true')
    parser.add_argument("--test_only", action="store_true")
    parser.add_argument("--ckpt_path", type=str, default=None)
    
    # global config
    parser.add_argument('--seed', type=int, default=1234)
    parser.add_argument("--no_wandb", action="store_true")
    parser.add_argument("--logging_level", type=str, default="warning")

    # train data params
    parser.add_argument('--random_rot', action='store_true')
    parser.add_argument("--pos_noise_std", type=float, default=0)    
    parser.add_argument("--pos_normalizer", type=float, default=2.0)    
    parser.add_argument("--ligand_atom_mode", type=str, default="add_aromatic", choices=["basic", "basic_PDB", "basic_plus_charge_PDB", "add_aromatic", "add_aromatic_plus_charge", "basic_plus_aromatic", "basic_plus_full", "basic_plus_charge", "full"])
    parser.add_argument('--time_decoupled', action='store_true')
    parser.add_argument('--decouple_mode', type=str, default='none')
    
    # train params
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument('--v_loss_weight', type=float, default=1)
    parser.add_argument('--bond_loss_weight', type=float, default=10)
    parser.add_argument('--lr', type=float, default=5e-4)
    parser.add_argument('--scheduler', type=str, default='plateau', choices=['cosine', 'plateau'])
    parser.add_argument('--weight_decay', type=float, default=0)
    parser.add_argument('--max_grad_norm', type=str, default='Q')  # '8.0' for

    # bfn params
    parser.add_argument("--sigma1_coord", type=float, default=0.05)
    parser.add_argument("--beta1", type=float, default=1.0)
    parser.add_argument("--beta1_bond", type=float, default=1.0)
    parser.add_argument("--beta1_charge", type=float, default=1.5)
    parser.add_argument("--beta1_aromatic", type=float, default=3.0)
    # parser.add_argument("--no_diff_coord", type=eval, default=False)
    # parser.add_argument("--charge_discretised_loss", type=eval, default=False)
    parser.add_argument("--t_min", type=float, default=0.0001)
    parser.add_argument('--use_discrete_t', type=eval, default=True)
    parser.add_argument('--discrete_steps', type=int, default=1000)
    parser.add_argument('--destination_prediction', type=eval, default=True)
    parser.add_argument('--sampling_strategy', type=str, default='end_back_pmf') #vanilla or end_back

    # network params
    parser.add_argument(
        "--time_emb_mode", type=str, default="simple", choices=["simple", "sin", 'rbf', 'rbfnn']
    )
    parser.add_argument("--time_emb_dim", type=int, default=1)
    parser.add_argument('--pos_init_mode', type=str, default='zero', choices=['zero', 'randn'])
    parser.add_argument('--bond_net_type', type=str, default='lin', choices=['lin', 'pre_att', 'flowmol', 'semla', 'lin+x'])
    parser.add_argument('--hidden_dim', type=int, default=128)
    parser.add_argument('--num_layers', type=int, default=4)
    parser.add_argument('--num_blocks', type=int, default=1)
    parser.add_argument('--pred_given_all', action='store_true')
    parser.add_argument('--pred_connectivity', action='store_true')
    parser.add_argument('--self_condition', action='store_true')
    parser.add_argument('--adaptive_norm', type=eval, default=False)

    # eval params
    parser.add_argument("--num_samples", type=int, default=10)
    parser.add_argument("--sample_steps", type=int, default=100)
    parser.add_argument('--sample_num_atoms', type=str, default='ref', choices=['prior', 'ref'])
    parser.add_argument("--visual_chain", action="store_true")
    parser.add_argument("--best_ckpt", type=str, default="val_loss", choices=["mol_stable", "complete", "val_loss"])
    parser.add_argument("--fix_bond", action="store_true")
    parser.add_argument("--pos_grad_weight", type=float, default=0)
    # SOTA: Add mode parameter for loss grid generation
    parser.add_argument("--mode", type=str, default="val", choices=["train", "val"],
                       help="Mode for loss grid generation (train/val dataset)")
    parser.add_argument("--enable_loss_grid", action="store_true",
                       help="Enable loss grid generation during training")
    # SOTA: Add GPU device management
    parser.add_argument("--gpu_device", type=int, default=None,
                       help="Specific GPU device to use (0, 1, 2, etc.). If not specified, auto-select best available GPU")
    parser.add_argument("--force_cpu", action="store_true",
                       help="Force CPU usage even if GPU is available")
    parser.add_argument("--eval_batch_size", type=int, default=100)
    parser.add_argument("--skip_chem", action="store_true")
    parser.add_argument("--t_power", type=float, default=1.0)
    parser.add_argument("--time_scheduler_path", type=str, default=None)
    parser.add_argument("--time_coef", type=float, default=1.0)

    # SOTA: 条件感知参数
    parser.add_argument("--condition_aware", action="store_true",
                       help="Enable condition-aware training")
    parser.add_argument("--condition_dim", type=int, default=2,
                       help="Condition dimension (QED, SA) - FIXED: Always 2 dimensions")
    # 移除condition_weight参数，条件信息仅用于引导，不参与监督
    parser.add_argument("--condition_use_prob", type=float, default=0.7,
                       help="Probability of using condition information during training")
    parser.add_argument("--condition_noise_std", type=float, default=0.05,
                       help="Standard deviation of condition noise for robustness")

    # SOTA: 目标条件参数（用于推理）
    parser.add_argument("--target_qed", type=float, default=0.675,
                       help="Target QED value (0.60-0.75)")
    parser.add_argument("--target_sa", type=float, default=0.35,
                       help="Target SA value (normalized, ≤0.35 corresponds to ≤3.5)")
    # 移除MW和LogP目标参数，只保留QED和SA

    # SOTA: 条件引导推理参数
    parser.add_argument("--guidance_scale", type=float, default=1.0,
                       help="Guidance scale for condition-guided inference")
    parser.add_argument("--adaptive_guidance", action="store_true",
                       help="Enable adaptive guidance strength")
    parser.add_argument("--guidance_model_path", type=str, default=None,
                       help="Path to trained condition guidance model")
    parser.add_argument("--use_gradient_guidance", action="store_true",
                       help="Use gradient-based guidance (compute guidance direction from model gradients)")

   
    parser.add_argument("--ablation_no_multiplicative", action="store_true",
                       help="Ablation: Disable multiplicative guidance (W/O 乘性算法)")
    parser.add_argument("--ablation_no_geometric", action="store_true",
                       help="Ablation: Disable geometric awareness (W/O 几何感知)")
    parser.add_argument("--ablation_no_time_consistency", action="store_true",
                       help="Ablation: Disable time consistency (W/O 时间一致性)")


    parser.add_argument("--enable_enhanced_guidance", action="store_true",
                       help="Enable enhanced guidance system for QED/SA boost")
    parser.add_argument("--amplification_level", type=str, default="moderate",
                       choices=["conservative", "moderate", "aggressive"],
                       help="Amplification level for enhanced guidance")
    parser.add_argument("--enable_post_processing", action="store_true",
                       help="Enable post-processing fine-tuner for additional optimization")


    parser.add_argument("--use_direct_logits_guidance", action="store_true",
                       help="Use direct logits guidance (mathematically correct)")
    parser.add_argument("--direct_guidance_weight", type=float, default=0.8,
                       help="Weight for direct logits guidance (0.0-1.0)")

    parser.add_argument("--enable_terminal_filtering", action="store_true",
                       help="Enable terminal filtering and two-stage optimization (only effective when guidance_scale > 0)")
    parser.add_argument("--disable_terminal_filtering", action="store_true",
                       help="Disable terminal filtering and two-stage optimization (even when guidance_scale > 0)")

    parser.add_argument("--guidance_start_ratio", type=float, default=0.0,
                       help="Guidance start ratio (0.0-1.0). 0.0=start from beginning, 0.8=start from 80%% of sampling steps")
    parser.add_argument("--guidance_end_ratio", type=float, default=1.0,
                       help="Guidance end ratio (0.0-1.0). 1.0=end at final step, 0.5=end at 50%% of sampling steps")

    _args = parser.parse_args()


    required_condition_params = [
        'condition_aware', 'condition_dim', 'condition_use_prob', 'condition_noise_std',
        'target_qed', 'target_sa'  # 只保留QED和SA
    ]

    required_guidance_params = [
        'guidance_scale', 'adaptive_guidance', 'guidance_model_path'
    ]

    missing_params = []
    for param in required_condition_params + required_guidance_params:
        if not hasattr(_args, param):
            missing_params.append(param)

    if missing_params:
        raise ValueError(f"Missing required parameters: {missing_params}")


    # SOTA: Advanced GPU device management and CUDA optimization
    def setup_optimal_gpu_environment(gpu_device=None, force_cpu=False):
        """Setup optimal GPU environment with SOTA performance optimizations."""
        import subprocess
        import time

        if force_cpu:
            os.environ['CUDA_VISIBLE_DEVICES'] = ''
            return 'cpu'

        # Check GPU availability and memory
        if torch.cuda.is_available():
            num_gpus = torch.cuda.device_count()

            # Get GPU memory info
            gpu_memory_info = []
            for i in range(num_gpus):
                try:
                    # Clear cache first
                    torch.cuda.empty_cache()

                    # Get memory info
                    total_memory = torch.cuda.get_device_properties(i).total_memory
                    allocated_memory = torch.cuda.memory_allocated(i)
                    cached_memory = torch.cuda.memory_reserved(i)
                    free_memory = total_memory - allocated_memory - cached_memory

                    gpu_memory_info.append({
                        'device': i,
                        'name': torch.cuda.get_device_properties(i).name,
                        'total': total_memory / 1024**3,  # GB
                        'free': free_memory / 1024**3,    # GB
                        'allocated': allocated_memory / 1024**3,  # GB
                        'utilization': (allocated_memory + cached_memory) / total_memory
                    })



                except Exception as e:
                    gpu_memory_info.append({'device': i, 'free': 0, 'utilization': 1.0})

            # Select optimal GPU
            if gpu_device is not None:
                if 0 <= gpu_device < num_gpus:
                    selected_gpu = gpu_device
                else:
                    selected_gpu = max(gpu_memory_info, key=lambda x: x.get('free', 0))['device']
            else:
                # Auto-select GPU with most free memory
                selected_gpu = max(gpu_memory_info, key=lambda x: x.get('free', 0))['device']

            # Set CUDA device
            device_name = f'cuda:{selected_gpu}'
            torch.cuda.set_device(selected_gpu)

            # SOTA: Optimize CUDA settings for RTX 4090/3090
            if 'RTX 4090' in gpu_memory_info[selected_gpu].get('name', '') or 'RTX 3090' in gpu_memory_info[selected_gpu].get('name', ''):
                torch.set_float32_matmul_precision('high')  # Use Tensor Cores
                torch.backends.cudnn.benchmark = True       # Optimize for consistent input sizes
                torch.backends.cuda.matmul.allow_tf32 = True  # Enable TF32 for performance
                torch.backends.cudnn.allow_tf32 = True
            else:
                torch.set_float32_matmul_precision('medium')

            # Clear any existing CUDA cache
            torch.cuda.empty_cache()

            return device_name

        else:
            return 'cpu'

    # Setup optimal GPU environment
    optimal_device = setup_optimal_gpu_environment(_args.gpu_device, _args.force_cpu)

    # SOTA: Proper Config initialization with config_file as first argument
    args_dict = _args.__dict__.copy()
    config_file = args_dict.pop('config_file')  # Remove config_file from kwargs

    # Remove GPU management args from config
    args_dict.pop('gpu_device', None)
    args_dict.pop('force_cpu', None)

    cfg = Config(config_file, **args_dict)

    cfg.test_only = _args.test_only

    if _args.condition_aware:
        if not hasattr(cfg.dynamics, 'net_config'):
            cfg.dynamics.net_config = Struct()

        cfg.dynamics.net_config.condition_aware = True
        cfg.dynamics.net_config.condition_dim = _args.condition_dim

        if not hasattr(cfg.data, 'condition_aware'):
            cfg.data.condition_aware = Struct()

        cfg.data.condition_aware.enabled = True
        cfg.data.condition_aware.use_probability = _args.condition_use_prob
        cfg.data.condition_aware.noise_std = _args.condition_noise_std
        cfg.data.condition_aware.target_conditions = Struct()
        cfg.data.condition_aware.target_conditions.qed = [0.60, 0.75]
        cfg.data.condition_aware.target_conditions.sa = [0.0, 0.35]

        if not hasattr(cfg.evaluation, 'target_conditions'):
            cfg.evaluation.target_conditions = Struct()
        cfg.evaluation.target_conditions.qed = _args.target_qed
        cfg.evaluation.target_conditions.sa = _args.target_sa

        if _args.disable_terminal_filtering:
            cfg.evaluation.terminal_filtering = False
        elif _args.enable_terminal_filtering:
            cfg.evaluation.terminal_filtering = True
        elif not hasattr(cfg.evaluation, 'terminal_filtering'):
            cfg.evaluation.terminal_filtering = True

        if not hasattr(cfg.evaluation, 'guidance_timing'):
            cfg.evaluation.guidance_timing = Struct()
        cfg.evaluation.guidance_timing.start_ratio = _args.guidance_start_ratio
        cfg.evaluation.guidance_timing.end_ratio = _args.guidance_end_ratio

        if not hasattr(cfg.evaluation, 'condition_guidance'):
            cfg.evaluation.condition_guidance = Struct()

        cfg.evaluation.condition_guidance.enabled = _args.guidance_model_path is not None
        cfg.evaluation.condition_guidance.guidance_scale = _args.guidance_scale
        cfg.evaluation.condition_guidance.adaptive_guidance = _args.adaptive_guidance
        cfg.evaluation.condition_guidance.model_path = _args.guidance_model_path  

        if _args.guidance_model_path:
            cfg.evaluation.condition_guidance.vocab_size = 263 
            cfg.evaluation.condition_guidance.seq_len = 64     
            cfg.evaluation.condition_guidance.condition_dim = _args.condition_dim 
            if not hasattr(cfg.evaluation.condition_guidance, 'target_conditions'):
                cfg.evaluation.condition_guidance.target_conditions = Struct()
            cfg.evaluation.condition_guidance.target_conditions.qed = _args.target_qed
            cfg.evaluation.condition_guidance.target_conditions.sa = _args.target_sa

            assert cfg.evaluation.condition_guidance.enabled == True
            assert cfg.evaluation.condition_guidance.model_path == _args.guidance_model_path
            assert cfg.evaluation.condition_guidance.guidance_scale == _args.guidance_scale
            assert cfg.evaluation.condition_guidance.adaptive_guidance == _args.adaptive_guidance

    seed_everything(cfg.seed)

    logging_level = {
        "info": logging.INFO,
        "debug": logging.DEBUG,
        "warning": logging.WARNING,
        "error": logging.ERROR,
        "fatal": logging.FATAL,
    }
    logging.set_verbosity(logging_level[cfg.logging_level])

    if cfg.empty_folder:
        shutil.rmtree(cfg.accounting.logdir)

    wandb_logger = get_logger(cfg)
        
    if cfg.test_only:
        config_path = cfg.accounting.dump_config_path

        if os.path.exists(config_path):
            tr_cfg = Config(config_path)
            if _args.ckpt_path and os.path.exists(_args.ckpt_path):
                ckpt_dir = os.path.dirname(_args.ckpt_path)
                potential_config_paths = [
                    os.path.join(ckpt_dir, 'config.yaml'),
                    os.path.join(ckpt_dir, '..', 'config.yaml'),
                    os.path.join(ckpt_dir, '..', '..', 'config.yaml')
                ]

                found_config = None
                for potential_path in potential_config_paths:
                    if os.path.exists(potential_path):
                        found_config = potential_path
                        break

                if found_config:
                    tr_cfg = Config(found_config)
                else:
                    found_config = None
            else:
                found_config = None

            if not found_config:

                os.makedirs(os.path.dirname(config_path), exist_ok=True)

                cfg.save2yaml(config_path)

                try:
                    tr_cfg = Config(config_path)
                except Exception as e:
                    tr_cfg = cfg

        if hasattr(cfg.dynamics, 'net_config') and hasattr(cfg.dynamics.net_config, 'num_blocks'):
            if not hasattr(tr_cfg.dynamics, 'net_config'):
                tr_cfg.dynamics.net_config = Struct()
            tr_cfg.dynamics.net_config.num_blocks = cfg.dynamics.net_config.num_blocks


        if _args.condition_aware and hasattr(cfg.dynamics.net_config, 'condition_aware'):
            if not hasattr(tr_cfg.dynamics, 'net_config'):
                tr_cfg.dynamics.net_config = Struct()
            tr_cfg.dynamics.net_config.condition_aware = cfg.dynamics.net_config.condition_aware
            tr_cfg.dynamics.net_config.condition_dim = cfg.dynamics.net_config.condition_dim

        if hasattr(cfg.evaluation, 'condition_guidance') and cfg.evaluation.condition_guidance.enabled:
            if not hasattr(tr_cfg.evaluation, 'condition_guidance'):
                tr_cfg.evaluation.condition_guidance = Struct()
            tr_cfg.evaluation.condition_guidance = cfg.evaluation.condition_guidance
            

        # Preserve critical training parameters for test_only mode
        original_eval_batch_size = getattr(cfg.evaluation, 'batch_size', 16)
        original_num_samples = getattr(cfg.evaluation, 'num_samples', 10)
        original_sample_steps = getattr(cfg.evaluation, 'sample_steps', 100)
        original_time_scheduler_path = getattr(cfg.evaluation, 'time_scheduler_path', None)

        cfg.dynamics = tr_cfg.dynamics

        # Merge train configuration to preserve batch_size and other critical parameters
        if hasattr(tr_cfg, 'train'):
            cfg.train = tr_cfg.train
            # Override with command line eval_batch_size if provided
            if hasattr(cfg, 'eval_batch_size') and cfg.eval_batch_size > 0:
                cfg.train.batch_size = cfg.eval_batch_size
            elif cfg.train.batch_size <= 0:
                cfg.train.batch_size = 16  # Safe default
        else:
            # Create minimal train config if missing
            if not hasattr(cfg, 'train'):
                cfg.train = Struct()
            cfg.train.batch_size = getattr(cfg, 'eval_batch_size', 16)
            cfg.train.num_workers = 0  # Safe for test mode

        # Merge data configuration
        tr_cfg.data.name = cfg.data.name
        tr_cfg.data.path = cfg.data.path
        if hasattr(cfg.data, 'split'):
            tr_cfg.data.split = cfg.data.split
        if hasattr(cfg.data, 'version'):
            tr_cfg.data.version = cfg.data.version
        elif hasattr(tr_cfg.data, 'version'):
            del tr_cfg.data.version
        tr_cfg.data.smiles_path = cfg.data.smiles_path
        tr_cfg.data.with_split = cfg.data.with_split
        cfg.data = tr_cfg.data

        cmd_condition_guidance = None
        cmd_target_conditions = None
        cmd_terminal_filtering = None
        cmd_condition_aware = None

        if hasattr(cfg.evaluation, 'condition_guidance'):
            cmd_condition_guidance = cfg.evaluation.condition_guidance

        if hasattr(cfg.evaluation, 'target_conditions'):
            cmd_target_conditions = cfg.evaluation.target_conditions

        if hasattr(cfg.evaluation, 'terminal_filtering'):
            cmd_terminal_filtering = cfg.evaluation.terminal_filtering

        if hasattr(cfg.evaluation, 'condition_aware'):
            cmd_condition_aware = cfg.evaluation.condition_aware

        # SOTA: Comprehensive evaluation configuration merge
        if hasattr(tr_cfg, 'evaluation'):
            cfg.evaluation = tr_cfg.evaluation
        else:
            # Create evaluation config if missing
            if not hasattr(cfg, 'evaluation'):
                cfg.evaluation = Struct()
            cfg.evaluation.batch_size = 100  # Default from config

        if cmd_condition_guidance is not None:
            cfg.evaluation.condition_guidance = cmd_condition_guidance

        else:
            if hasattr(cfg.evaluation, 'condition_guidance'):
                cfg.evaluation.condition_guidance.enabled = False
                cfg.evaluation.condition_guidance.model_path = None


        if cmd_target_conditions is not None:
            cfg.evaluation.target_conditions = cmd_target_conditions


        if cmd_terminal_filtering is not None:
            cfg.evaluation.terminal_filtering = cmd_terminal_filtering


        if cmd_condition_aware is not None:
            cfg.evaluation.condition_aware = cmd_condition_aware


        # Restore and validate command line evaluation parameters
        cfg.evaluation.batch_size = max(1, original_eval_batch_size)  # Ensure positive
        cfg.evaluation.num_samples = max(1, original_num_samples)
        cfg.evaluation.sample_steps = max(1, original_sample_steps)
        if original_time_scheduler_path:
            cfg.evaluation.time_scheduler_path = original_time_scheduler_path

        # SOTA: Ensure train.batch_size is properly set for DataLoader
        # In test_only mode, use eval_batch_size for both train and eval loaders
        if hasattr(cfg, 'eval_batch_size') and cfg.eval_batch_size > 0:
            cfg.train.batch_size = cfg.eval_batch_size
            cfg.evaluation.batch_size = cfg.eval_batch_size
        elif cfg.train.batch_size <= 0:
            cfg.train.batch_size = cfg.evaluation.batch_size

        # Ensure both batch sizes are positive
        if cfg.train.batch_size <= 0:
            cfg.train.batch_size = 16
        if cfg.evaluation.batch_size <= 0:
            cfg.evaluation.batch_size = 16

        # Merge other configurations
        if hasattr(tr_cfg, 'time_decoupled'):
            cfg.time_decoupled = tr_cfg.time_decoupled
        if hasattr(tr_cfg, 'decouple_mode'):
            cfg.decouple_mode = tr_cfg.decouple_mode


        if _args.guidance_model_path:

            try:
                use_direct_logits = getattr(_args, 'use_direct_logits_guidance', False)
                enable_enhanced = getattr(_args, 'enable_enhanced_guidance', False)

                if use_direct_logits:
                    from core.models.direct_logits_guidance import create_hybrid_guidance_integrator

                    direct_guidance_weight = getattr(_args, 'direct_guidance_weight', 0.8)

                    guidance_integrator = create_hybrid_guidance_integrator(
                        guidance_model_path=_args.guidance_model_path,
                        device=optimal_device,
                        use_direct_guidance=True,
                        direct_guidance_weight=direct_guidance_weight,
                        ablation_no_multiplicative=getattr(_args, 'ablation_no_multiplicative', False),
                        ablation_no_geometric=getattr(_args, 'ablation_no_geometric', False),
                        ablation_no_time_consistency=getattr(_args, 'ablation_no_time_consistency', False)
                    )

                elif enable_enhanced:
                    from core.models.guidance_integration_wrapper import (
                        create_wrapped_guidance_integrator,
                        create_booster_config
                    )

                    amplification_level = getattr(_args, 'amplification_level', 'moderate')
                    booster_config = create_booster_config(
                        amplification_level=amplification_level,
                        enable_early_intervention=True,
                        enable_atom_preference=True,
                        enable_synergy=True
                    )


                    guidance_integrator = create_wrapped_guidance_integrator(
                        guidance_model_path=_args.guidance_model_path,
                        device=optimal_device,
                        enable_booster=True,
                        booster_config=booster_config,
                        ablation_no_multiplicative=getattr(_args, 'ablation_no_multiplicative', False),
                        ablation_no_geometric=getattr(_args, 'ablation_no_geometric', False),
                        ablation_no_time_consistency=getattr(_args, 'ablation_no_time_consistency', False)
                    )
                else:
                    from core.models.geometric_guidance_integration import create_geometric_guidance_integrator


                    guidance_integrator = create_geometric_guidance_integrator(
                        guidance_model_path=_args.guidance_model_path,
                        device=optimal_device,
                        model_config=None,
                        ablation_no_multiplicative=getattr(_args, 'ablation_no_multiplicative', False),
                        ablation_no_geometric=getattr(_args, 'ablation_no_geometric', False),
                        ablation_no_time_consistency=getattr(_args, 'ablation_no_time_consistency', False)
                    )

 
            except Exception as e:
                import traceback
                traceback.print_exc()
                raise RuntimeError(f"{e}")
        else:
            guidance_integrator = None


    elif hasattr(cfg, 'ckpt_path') and cfg.ckpt_path is not None:
        config_path = os.path.join('/'.join(cfg.ckpt_path.split('/')[:-2]), "config.yaml")
        assert os.path.exists(config_path), f"config file {config_path} not found"
        model_config = Config(config_path)
        cfg.dynamics = model_config.dynamics
        model_config.data.name = cfg.data.name
        if hasattr(cfg.data, 'version'):
            model_config.data.version = cfg.data.version
        model_config.data.path = cfg.data.path
        model_config.data.smiles_path = cfg.data.smiles_path
        model_config.data.with_split = cfg.data.with_split
        model_config.data.split = cfg.data.split
        cfg.data = model_config.data
        cfg.save2yaml(cfg.accounting.dump_config_path)
    else:
        # SOTA: Ensure evaluation section has the required parameters for loss grid generation
        if not hasattr(cfg, 'evaluation'):
            cfg.evaluation = Struct()

        # Set default values if not provided
        if not hasattr(cfg.evaluation, 'mode'):
            cfg.evaluation.mode = getattr(cfg, 'mode', 'val')
        if not hasattr(cfg.evaluation, 'sample_steps'):
            cfg.evaluation.sample_steps = getattr(cfg, 'sample_steps', 100)
        if not hasattr(cfg.evaluation, 'batch_size'):
            cfg.evaluation.batch_size = getattr(cfg, 'eval_batch_size', 100)
        if not hasattr(cfg.evaluation, 'num_samples'):
            cfg.evaluation.num_samples = getattr(cfg, 'num_samples', 10)

        # SOTA: Ensure time_scheduler_path is properly set
        if hasattr(cfg, 'time_scheduler_path') and cfg.time_scheduler_path:
            cfg.evaluation.time_scheduler_path = cfg.time_scheduler_path

        cfg.save2yaml(cfg.accounting.dump_config_path)

    train_loader, val_loader, dataset_smiles_set = get_dataloader(cfg)

    wandb_logger.log_hyperparams(cfg.todict())

    model = SBDD4Train(config=cfg)

    
    if cfg.train.resume:
        cfg.ckpt_path = os.path.join(cfg.accounting.checkpoint_dir, "last.ckpt")

    if hasattr(cfg, 'ckpt_path') and cfg.ckpt_path is not None:
        checkpoint = torch.load(cfg.ckpt_path, map_location=optimal_device)
        model.load_state_dict(checkpoint["state_dict"])

        # Move model to optimal device
        model = model.to(optimal_device)
       
    if hasattr(cfg.evaluation, "time_scheduler_path") and cfg.evaluation.time_scheduler_path is not None:
        time_scheduler = torch.load(cfg.evaluation.time_scheduler_path, map_location='cpu')

        # Handle different time scheduler formats
        if isinstance(time_scheduler, np.ndarray):
            time_scheduler = torch.from_numpy(time_scheduler).float()
        elif not isinstance(time_scheduler, torch.Tensor):
            time_scheduler = torch.tensor(time_scheduler).float()

        # Move time scheduler to optimal device
        time_scheduler = time_scheduler.to(optimal_device)

        model.configure_time_scheduler(time_scheduler)

    callbacks = [
            RecoverCallback(
                latest_ckpt=os.path.join(cfg.accounting.checkpoint_dir, "last.ckpt"),
                resume=cfg.train.resume,
                recover_trigger_loss=1e7,
            ),
            # TODO: this seems a dynamic clip, turn to static?
            GradientClip(max_grad_norm=cfg.train.max_grad_norm),  # time consuming
            # TODO: add data normalizing back?
            NormalizerCallback(normalizer_dict=cfg.data.normalizer_dict),
            MolVisualizationCallback(
                # dataset=train_loader.loader.ds,
                # atom_enc_mode=cfg.data.transform.ligand_atom_mode,
                atom_decoder=cfg.data.atom_decoder,
                colors_dic=cfg.data.colors_dic,
                radius_dic=cfg.data.radius_dic,
            ),
            TwistedReconValidationCallback(
                val_freq=min(cfg.train.val_freq, len(train_loader)),
                enable_loss_grid=getattr(cfg, 'enable_loss_grid', False),
            ),
            CondMolGenValidationCallback(
                dataset=None,  # TODO: implement CrossDockGen & NewBenchmark
                # atom_decoder={1: 'H', 6: 'C', 7: 'N', 8: 'O', 9: 'F', 15: 'P', 16: 'S', 17: 'Cl'},
                atom_decoder=cfg.data.atom_decoder,
                atom_enc_mode=cfg.data.transform.ligand_atom_mode,
                atom_type_one_hot=False,
                single_bond=False,
                docking_config=None if 'zinc' in cfg.data.path else cfg.evaluation.docking_config,
                dataset_smiles_set=dataset_smiles_set,
                # single_bond=cfg.evaluation.single_bond,  # TODO: check compatibility
            ),
            DockingTestCallback(
                dataset=None,  # TODO: implement CrossDockGen & NewBenchmark
                atom_decoder=cfg.data.atom_decoder,
                atom_enc_mode=cfg.data.transform.ligand_atom_mode,
                atom_type_one_hot=False,
                single_bond=False,
                docking_config=None if 'zinc' in cfg.data.path else cfg.evaluation.docking_config,
                dataset_smiles_set=dataset_smiles_set,
                docking_rmsd=getattr(cfg.evaluation, 'docking_rmsd', False),
            ),
            ModelCheckpoint(
                monitor="val/recon_loss",
                mode="min",
                # monitor="val/mol_stable",
                # mode="max",
                every_n_epochs=cfg.train.ckpt_freq,
                # every_n_train_steps=cfg.train.val_freq,
                dirpath=cfg.accounting.checkpoint_dir,
                filename="epoch{epoch:02d}-val_loss{val/recon_loss:.2f}-mol_stable{val/mol_stable:.2f}-complete{val/completeness:.2f}",
                save_top_k=3,
                auto_insert_metric_name=False,
                save_last=True,
            ),
            EMACallback(decay=cfg.train.ema_decay),
            EarlyStopping(monitor='val/recon_loss', mode='min', patience=cfg.train.scheduler.patience * 2),
            # EMACallback(decay=cfg.train.ema_decay, ema_device="cuda" if torch.cuda.is_available() else "cpu"),
            # DebugCallback(),
            # LearningRateMonitor(logging_interval='step'),
        ]
    

    # SOTA: Configure trainer with optimal device settings
    if optimal_device.startswith('cuda'):
        # Extract GPU number from device string (e.g., 'cuda:1' -> 1)
        gpu_id = int(optimal_device.split(':')[1]) if ':' in optimal_device else 0
        trainer_devices = [gpu_id]
        accelerator = "gpu"
    else:
        trainer_devices = 1
        accelerator = "cpu"

    trainer = pl.Trainer(
        default_root_dir=cfg.accounting.logdir,
        max_epochs=cfg.train.epochs,
        check_val_every_n_epoch=cfg.train.ckpt_freq,
        devices=trainer_devices,
        accelerator=accelerator,
        strategy="auto",
        # accumulate_grad_batches=2,
        # overfit_batches=10,
        logger=wandb_logger,
        num_sanity_val_steps=0,
        # overfit_batches=10,
        # gradient_clip_val=1.0,
        callbacks=callbacks,
    )
    # num_sanity_val_steps=2, overfit_batches=10, devices=1
    # trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=eval_loader)

    if not cfg.test_only:
        model.dynamics.sampling_strategy = 'vanilla'
        # if cfg.train.resume:
        #     trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=val_loader, ckpt_path="last")
        # else:
        # wandb.watch(model.dynamics, log='all')
        # Enhanced training with SOTA error handling
        try:
            trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=val_loader)

        except RuntimeError as e:
            if "DataLoader worker" in str(e) and "killed by signal" in str(e):

                # Attempt to continue with testing if checkpoints exist
                import glob
                ckpts = glob.glob(os.path.join(cfg.accounting.checkpoint_dir, "*.ckpt"))

            else:
                
                raise

        except Exception as e:
            raise
        model.dynamics.sampling_strategy = cfg.dynamics.sampling_strategy
        if torch.cuda.device_count() > 1:
            trainer.devices = [0]
            callbacks.append(
                DockingTestCallback(
                    dataset=None,  # TODO: implement CrossDockGen & NewBenchmark
                    atom_decoder=cfg.data.atom_decoder,
                    atom_enc_mode=cfg.data.transform.ligand_atom_mode,
                    atom_type_one_hot=False,
                    single_bond=False,
                    docking_config=None if 'zinc' in cfg.data.path else cfg.evaluation.docking_config,
                    dataset_smiles_set=dataset_smiles_set,
                    docking_rmsd=getattr(cfg.evaluation, 'docking_rmsd', False),
                )
            )
            trainer.callbacks = callbacks
        # Enhanced testing with error handling
        try:
            trainer.test(model, dataloaders=val_loader, ckpt_path="best")
   

        finally:
            from core.evaluation.utils.vina_resource_manager import cleanup_vina_resources, print_vina_resource_stats
            print_vina_resource_stats()
            cleanup_vina_resources()
           
            for i in range(3):
                collected = gc.collect()

            # Clear CUDA cache
            if torch.cuda.is_available():
                torch.cuda.empty_cache()


    else:

        if not os.path.exists(cfg.accounting.checkpoint_dir):
            if _args.ckpt_path and os.path.exists(_args.ckpt_path):
                best_ckpt = _args.ckpt_path
            else:
                raise FileNotFoundError(f"not found")
        else:
            if _args.ckpt_path and os.path.exists(_args.ckpt_path):
                best_ckpt = _args.ckpt_path
            else:
                ckpts = glob.glob(os.path.join(cfg.accounting.checkpoint_dir, "*val_loss*.ckpt"))

                if len(ckpts) == 0:
                    alternative_patterns = [
                        "*complete*.ckpt",
                        "*mol_stable*.ckpt",
                        "*.ckpt",
                        "*epoch*.ckpt"
                    ]

                    for pattern in alternative_patterns:
                        ckpts = glob.glob(os.path.join(cfg.accounting.checkpoint_dir, pattern))
                        if ckpts:
                            break

                    if len(ckpts) == 0:
                        if _args.ckpt_path and os.path.exists(_args.ckpt_path):
                            best_ckpt = _args.ckpt_path
                        else:
                            raise FileNotFoundError(f"not found")

                if len(ckpts) > 0:
                    try:
                        def safe_parse_metric(filename, metric_name):
                            """安全地从文件名中解析指标值"""
                            try:
                                if metric_name in filename:
                                    after_metric = filename.split(metric_name)[-1]
                                    match = re.search(r'(\d+\.?\d*)', after_metric)
                                    if match:
                                        return float(match.group(1))
                                return None
                            except Exception:
                                return None

                        if not hasattr(cfg, "best_ckpt") or cfg.best_ckpt == "val_loss":
                            valid_ckpts = []
                            for ckpt in ckpts:
                                val_loss = safe_parse_metric(os.path.basename(ckpt), "val_loss")
                                if val_loss is not None:
                                    valid_ckpts.append((ckpt, val_loss))

                            if valid_ckpts:
                                best_ckpt = sorted(valid_ckpts, key=lambda x: x[1])[0][0]
                            else:
                                best_ckpt = ckpts[0]

                        elif cfg.best_ckpt == "complete":
                            complete_ckpts = glob.glob(os.path.join(cfg.accounting.checkpoint_dir, "*complete*"))
                            if len(complete_ckpts) == 0:
                                complete_ckpts = ckpts

                            valid_ckpts = []
                            for ckpt in complete_ckpts:
                                complete_score = safe_parse_metric(os.path.basename(ckpt), "complete")
                                if complete_score is not None:
                                    valid_ckpts.append((ckpt, complete_score))

                            if valid_ckpts:
                                best_ckpt = sorted(valid_ckpts, key=lambda x: x[1])[-1][0]
                            else:
                                best_ckpt = complete_ckpts[0]
                        else:
                            valid_ckpts = []
                            for ckpt in ckpts:
                                mol_stable = safe_parse_metric(os.path.basename(ckpt), "mol_stable")
                                if mol_stable is not None:
                                    valid_ckpts.append((ckpt, mol_stable))

                            if valid_ckpts:
                                best_ckpt = sorted(valid_ckpts, key=lambda x: x[1])[-1][0] 
                            else:
                                best_ckpt = ckpts[0]

                    except Exception as e:
                        best_ckpt = ckpts[0]

        import torch
        checkpoint = torch.load(best_ckpt, map_location='cpu')
            

        if hasattr(_args, 'use_gradient_guidance') and _args.use_gradient_guidance:
            model.dynamics.use_gradient_guidance = True
        else:
            model.dynamics.use_gradient_guidance = False


        if hasattr(_args, 'ablation_no_multiplicative') and _args.ablation_no_multiplicative:
            print(f"   (W/O Multiplicative)")
            model.dynamics.ablation_no_multiplicative = True
        else:
            model.dynamics.ablation_no_multiplicative = False

        if hasattr(_args, 'ablation_no_geometric') and _args.ablation_no_geometric:
            print(f"   (W/O Geometric)")
            model.dynamics.ablation_no_geometric = True
        else:
            model.dynamics.ablation_no_geometric = False

        if hasattr(_args, 'ablation_no_time_consistency') and _args.ablation_no_time_consistency:
            print(f"   (W/O Time Consistency)")
            model.dynamics.ablation_no_time_consistency = True
        else:
            model.dynamics.ablation_no_time_consistency = False

        any_ablation = (model.dynamics.ablation_no_multiplicative or
                       model.dynamics.ablation_no_geometric or
                       model.dynamics.ablation_no_time_consistency)
    

        if hasattr(_args, 'guidance_start_ratio') and hasattr(_args, 'guidance_end_ratio'):

            if not hasattr(model.cfg.evaluation, 'guidance_timing'):
                from core.config.config import Struct
                model.cfg.evaluation.guidance_timing = Struct()

            model.cfg.evaluation.guidance_timing.start_ratio = _args.guidance_start_ratio
            model.cfg.evaluation.guidance_timing.end_ratio = _args.guidance_end_ratio

        trainer.test(model, dataloaders=val_loader, ckpt_path=best_ckpt)
        # trainer.test(model, dataloaders=val_loader, ckpt_path="last")
