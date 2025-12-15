#!/usr/bin/env python
# -*- coding: utf-8 -*-

import pickle
import torch
from pathlib import Path

# 1) 读取 posebusters_rmsd1.0_pocket10/index.pkl
root = Path("/data-extend/wangzilin/project/molpilot/MolPilot/data/posebusters_benchmark_set_pocket10")
index_path = root / "index.pkl"

with open(index_path, "rb") as f:
    index = pickle.load(f)

N = len(index)
print("Total processed PoseBusters pockets:", N)

# 2) 简单策略：全部样本都作为 test
all_idx = torch.arange(N, dtype=torch.long)

split = {
    "train": all_idx,   # 虽然不会在 PoseBusters 上训练，但为了兼容性先填上
    "val":   all_idx,   # 有的 DataModule 可能会访问 val，这里也填上
    "test":  all_idx,   # 真正用来做 OOD 评估的是这一项
}

out_path = Path("/data-extend/wangzilin/project/molpilot/MolPilot/data/posebusters_benchmark_alltest_split.pt")
torch.save(split, out_path)

print("Saved split file to:", out_path)
print(f"train/val/test size = {len(split['train'])}/{len(split['val'])}/{len(split['test'])}")
