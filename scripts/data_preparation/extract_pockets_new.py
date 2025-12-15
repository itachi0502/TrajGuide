import os
import sys
import argparse
import multiprocessing as mp
import pickle
import shutil
from functools import partial
from pathlib import Path

import numpy as np
# ==== 兼容补丁：为老代码补上 np.int ====
if not hasattr(np, "int"):
    np.int = int
if not hasattr(np, "long"):
    np.long = int
# =====================================

# 添加项目根目录
script_dir = Path(__file__).resolve().parent
project_root = script_dir.parent.parent
sys.path.insert(0, str(project_root))

from tqdm.auto import tqdm
from utils.data import PDBProtein, parse_sdf_file


def process_item(item, args):
    """
    item: (protein_rel, ligand_rel, rmsd)
    """
    protein_rel, ligand_rel, rmsd = item
    pdb_path = os.path.join(args.source, protein_rel)
    sdf_path = os.path.join(args.source, ligand_rel)

    try:
        # **关键改动 1：直接用路径初始化 PDBProtein 和 parse_sdf_file**
        protein = PDBProtein(pdb_path)              # 让 PDBProtein 自己读文件
        ligand = parse_sdf_file(sdf_path)           # 直接传路径，和 TagMol / MolPilot 原始代码一致

        # 以 ligand 为中心，截取 10 Å 口袋
        pocket_block = protein.residues_to_pdb_block(
            protein.query_residues_ligand(ligand, args.radius)
        )

        # 目标文件名 & 路径
        ligand_fn = ligand_rel
        pocket_fn = ligand_rel[:-4] + f'_pocket{args.radius}.pdb'
        ligand_dest = os.path.join(args.dest, ligand_fn)
        pocket_dest = os.path.join(args.dest, pocket_fn)

        # 创建子目录，例如 dest/5S8I_2LY/
        os.makedirs(os.path.dirname(ligand_dest), exist_ok=True)

        # 复制 ligand.sdf
        shutil.copyfile(sdf_path, ligand_dest)

        # 写 pocket pdb
        with open(pocket_dest, 'w') as f:
            f.write(pocket_block)

        return pocket_fn, ligand_fn, protein_rel, rmsd

    except Exception as e:
        # **关键改动 2：打印具体异常信息，方便你定位问题**
        print(f'[ERROR] Failed on protein={protein_rel}, ligand={ligand_rel}: {e}')
        return None  # 用 None 表示失败


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', type=str, required=True)
    parser.add_argument('--dest', type=str, required=True)
    parser.add_argument('--radius', type=int, default=10)
    parser.add_argument('--num_workers', type=int, default=16)
    parser.add_argument('--rmsd_thr', type=float, default=None)
    parser.add_argument('--overwrite', action='store_true')
    args = parser.parse_args()

    # 处理目标目录
    if os.path.exists(args.dest):
        if args.overwrite:
            print(f'⚠️  目标目录已存在，将在其上继续写入: {args.dest}')
        else:
            print(f'❌ 错误：目标目录已存在: {args.dest}')
            print('   使用 --overwrite 允许覆盖，或手动删除该目录后重试')
            sys.exit(1)

    os.makedirs(args.dest, exist_ok=True)

    # 读取 source/index.pkl（你之前 build_posebuster.py 生成的）
    index_path = os.path.join(args.source, 'index.pkl')
    with open(index_path, 'rb') as f:
        index = pickle.load(f)

    print('源样本数:', len(index))

    # PoseBusters 的 rmsd 你设成 0.0，本质上没用，这里允许可选过滤
    if args.rmsd_thr is not None:
        index = [it for it in index if len(it) > 2 and it[2] <= args.rmsd_thr]
    print('RMSD 过滤后样本数:', len(index))

    if len(index) == 0:
        print('❌ 没有可处理的样本')
        sys.exit(1)

    # 多进程处理
    pool = mp.Pool(args.num_workers)
    results = []
    for ret in tqdm(pool.imap_unordered(partial(process_item, args=args), index), total=len(index)):
        results.append(ret)
    pool.close()
    pool.join()

    # **关键改动 3：只保留 process_item 成功的条目**
    index_pocket = [r for r in results if r is not None and r[0] is not None]
    print('成功生成 pocket 的样本数:', len(index_pocket))

    out_index_path = os.path.join(args.dest, 'index.pkl')
    with open(out_index_path, 'wb') as f:
        pickle.dump(index_pocket, f)

    print('Done. %d protein-ligand pairs in total.' % len(index_pocket))
