#!/usr/bin/env python
# -*- coding: utf-8 -*-
import sys
from pathlib import Path

import numpy as np

if not hasattr(np, "int"):
    np.int = int
if not hasattr(np, "long"):
    np.long = int
if not hasattr(np, "bool"):
    np.bool = bool        # 或 np.bool_ 也可以

script_dir = Path(__file__).resolve().parent
project_root = script_dir.parent.parent  # MolPilot/scripts/data_preparation -> MolPilot
sys.path.insert(0, str(project_root))
import os
from core.datasets.pl_pair_dataset import PocketLigandPairDataset

if __name__ == "__main__":
    raw_path = "/data-extend/wangzilin/project/molpilot/MolPilot/data/posebusters_benchmark_set_pocket10"

    # transform 对预处理不重要，可以传 None
    dataset = PocketLigandPairDataset(
        raw_path=raw_path,
        transform=None,
        version="kekulize",
    )

    print("Done. LMDB path:")
    print(dataset.processed_path)
    print("Number of entries:", len(dataset))
