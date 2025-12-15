#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import torch
import pickle
import numpy as np
from tqdm import tqdm
from rdkit import Chem
from rdkit.Chem import QED, Descriptors
from rdkit.Contrib.SA_Score import sascorer
import argparse
from pathlib import Path

# 导入数据集相关模块
from core.datasets import get_dataset
from core.config.config import Config


def sa_norm_from_rdkit(sa_raw: float) -> float:
    """
    将 RDKit 的原始 SA 分数 (≈1~10; 低=易合成) 映射为 [0,1] 的高优分数。
    1 -> 1.0（最易合成），10 -> 0.0（最难合成）
    """
    v = float(sa_raw)
    # 裁剪到 RDKit 常见范围，避免异常值影响
    if v < 1.0:
        v = 1.0
    elif v > 10.0:
        v = 10.0
    return (10.0 - v) / 9.0


def calculate_molecular_properties_from_data(data):
    try:
        if hasattr(data, 'ligand_smiles') and data.ligand_smiles is not None:
            smiles = data.ligand_smiles
            if isinstance(smiles, (list, tuple)) and len(smiles) > 0:
                smiles = smiles[0]

            mol = Chem.MolFromSmiles(smiles)
            if mol is not None:
                qed_value = float(QED.qed(mol))

                sa_raw = float(sascorer.calculateScore(mol))
                sa_normalized = sa_norm_from_rdkit(sa_raw)

                return qed_value, sa_normalized

        return None, None

    except Exception as e:
        return None, None


def process_dataset_split(dataset, split_name, atom_decoder, ligand_atom_mode, output_dir):

    processed_data = []
    valid_count = 0
    total_count = len(dataset)

    for idx in tqdm(range(total_count), desc=f"处理{split_name}"):
        try:
            data = dataset[idx]

            qed, sa = calculate_molecular_properties_from_data(data)

            if qed is not None and sa is not None:
                mol_data = {
                    'molecule': data,              
                    'conditions': [qed, sa],      
                    'original_index': idx
                }
                processed_data.append(mol_data)
                valid_count += 1

        except Exception as e:
            continue

    # 保存处理后的数据
    output_file = output_dir / f"{split_name}_condition_data.pkl"
    with open(output_file, 'wb') as f:
        pickle.dump(processed_data, f)


    return processed_data


def generate_condition_guidance_dataset(config_path, output_dir):

    cfg = Config(config_path)

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    dataset, subsets = get_dataset(config=cfg.data)

    atom_decoder = cfg.data.atom_decoder
    ligand_atom_mode = cfg.data.transform.ligand_atom_mode

    all_processed_data = {}

    for split_name, split_dataset in subsets.items():
        if len(split_dataset) > 0:
            processed_data = process_dataset_split(
                split_dataset, split_name, atom_decoder, ligand_atom_mode, output_dir
            )
            all_processed_data[split_name] = processed_data

    if 'train' in all_processed_data and len(all_processed_data['train']) > 0:
        train_data = all_processed_data['train']
        total_train = len(train_data)

        test_size = max(1, total_train // 10)
        val_size = max(1, total_train // 10)

        import random
        random.shuffle(train_data)

        test_data = train_data[:test_size]
        val_data = train_data[test_size:test_size + val_size]
        final_train_data = train_data[test_size + val_size:]

        all_processed_data['train'] = final_train_data
        all_processed_data['val'] = val_data
        all_processed_data['test'] = test_data


    total_molecules = sum(len(data) for data in all_processed_data.values())

    all_qed = []
    all_sa = []

    for split_data in all_processed_data.values():
        for mol_data in split_data:
            qed, sa = mol_data['conditions']
            all_qed.append(float(qed))
            all_sa.append(float(sa))

    if all_qed and all_sa:
        stats = {
            'total_molecules': total_molecules,
            'qed_stats': {
                'mean': float(np.mean(all_qed)),
                'std': float(np.std(all_qed)),
                'min': float(np.min(all_qed)),
                'max': float(np.max(all_qed))
            },
            'sa_stats': {
                'mean': float(np.mean(all_sa)),
                'std': float(np.std(all_sa)),
                'min': float(np.min(all_sa)),
                'max': float(np.max(all_sa))
            }
        }


        stats_file = output_dir / "dataset_statistics.pkl"
        with open(stats_file, 'wb') as f:
            pickle.dump(stats, f)


    #  保存处理后的数据（包含新的测试集）
    for split_name, split_data in all_processed_data.items():
        output_file = output_dir / f"{split_name}_condition_data.pkl"

        with open(output_file, 'wb') as f:
            pickle.dump(split_data, f)



    return all_processed_data


def main():
    parser = argparse.ArgumentParser(description="生成条件引导模型数据集")
    parser.add_argument(
        "--config", 
        type=str, 
        default="configs/crossdock_train_test_condition_aware.yaml",
        help="配置文件路径"
    )
    parser.add_argument(
        "--output_dir", 
        type=str, 
        default="data/condition_guidance_dataset",
        help="输出目录"
    )
    
    args = parser.parse_args()
    
    # 生成数据集
    generate_condition_guidance_dataset(args.config, args.output_dir)


if __name__ == "__main__":
    main()
