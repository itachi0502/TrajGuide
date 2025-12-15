import os
import sys
import argparse
import multiprocessing as mp
import pickle
import shutil
from functools import partial
from pathlib import Path

# 添加项目根目录到 Python 路径
script_dir = Path(__file__).resolve().parent
project_root = script_dir.parent.parent  # MolPilot/scripts/data_preparation -> MolPilot
sys.path.insert(0, str(project_root))

from tqdm.auto import tqdm

from utils.data import PDBProtein, parse_sdf_file


def process_item(item, args):
    try:
        # item 格式: (protein_file, ligand_file, rmsd)
        protein_fn = item[0]
        ligand_fn = item[1]
        rmsd = item[2] if len(item) > 2 else 0.0

        # 读取蛋白质和配体文件
        protein_path = os.path.join(args.source, protein_fn)
        ligand_path = os.path.join(args.source, ligand_fn)

        if not os.path.exists(protein_path):
            print(f'❌ Protein file not found: {protein_path}')
            return None, ligand_fn, protein_fn, rmsd

        if not os.path.exists(ligand_path):
            print(f'❌ Ligand file not found: {ligand_path}')
            return None, ligand_fn, protein_fn, rmsd

        # 解析蛋白质
        with open(protein_path, 'r') as f:
            pdb_block = f.read()
        protein = PDBProtein(pdb_block)

        # 解析配体
        ligand = parse_sdf_file(ligand_path)

        # 提取 pocket
        pocket_residues = protein.query_residues_ligand(ligand, args.radius)

        if len(pocket_residues) == 0:
            print(f'⚠️  No pocket residues found for {ligand_fn} (radius={args.radius}Å)')
            return None, ligand_fn, protein_fn, rmsd

        pdb_block_pocket = protein.residues_to_pdb_block(pocket_residues)

        # 生成输出文件名
        pocket_fn = ligand_fn[:-4] + '_pocket%d.pdb' % args.radius
        ligand_dest = os.path.join(args.dest, ligand_fn)
        pocket_dest = os.path.join(args.dest, pocket_fn)
        os.makedirs(os.path.dirname(ligand_dest), exist_ok=True)

        # 复制配体文件
        shutil.copyfile(src=ligand_path, dst=ligand_dest)

        # 保存 pocket 文件
        with open(pocket_dest, 'w') as f:
            f.write(pdb_block_pocket)

        return pocket_fn, ligand_fn, protein_fn, rmsd

    except Exception as e:
        print(f'❌ Exception occurred for {item}: {type(e).__name__}: {str(e)}')
        import traceback
        traceback.print_exc()
        return None, item[1] if len(item) > 1 else None, item[0] if len(item) > 0 else None, item[2] if len(item) > 2 else 0.0


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', type=str, default='./data/crossdocked_subset')
    parser.add_argument('--dest', type=str, required=True)
    parser.add_argument('--radius', type=int, default=10)
    parser.add_argument('--num_workers', type=int, default=16)
    parser.add_argument('--rmsd_thr', type=float, default=None,
                        help='RMSD threshold for filtering (only process items with RMSD <= threshold)')
    parser.add_argument('--overwrite', action='store_true',
                        help='Allow overwriting existing destination directory')
    args = parser.parse_args()

    # 检查目标目录是否存在
    if os.path.exists(args.dest):
        if args.overwrite:
            print(f'⚠️  目标目录已存在，将覆盖: {args.dest}')
            # 不删除目录，允许增量更新
        else:
            print(f'❌ 错误：目标目录已存在: {args.dest}')
            print(f'   请使用 --overwrite 参数允许覆盖，或手动删除该目录')
            print(f'   删除命令: rm -rf {args.dest}')
            sys.exit(1)

    os.makedirs(args.dest, exist_ok=True)
    with open(os.path.join(args.source, 'index.pkl'), 'rb') as f:
        index = pickle.load(f)

    # 如果指定了 RMSD 阈值，进行过滤
    if args.rmsd_thr is not None:
        original_count = len(index)
        # index 格式: (protein_file, ligand_file, rmsd)
        index = [item for item in index if len(item) > 2 and item[2] <= args.rmsd_thr]
        filtered_count = len(index)
        print(f'RMSD filtering: {original_count} -> {filtered_count} items (threshold: {args.rmsd_thr})')

    if len(index) == 0:
        print('No items to process after filtering!')
        sys.exit(1)

    pool = mp.Pool(args.num_workers)
    index_pocket = []
    failed_samples = []
    success_count = 0
    fail_count = 0

    print(f'\n🚀 开始提取 pocket，共 {len(index)} 个样本...\n')

    for item_pocket in tqdm(pool.imap_unordered(partial(process_item, args=args), index), total=len(index)):
        if item_pocket[0] is not None:  # pocket_fn
            index_pocket.append(item_pocket)
            success_count += 1
        else:
            failed_samples.append(item_pocket)
            fail_count += 1

    pool.close()

    # 只保存成功的样本到 index.pkl
    index_path = os.path.join(args.dest, 'index.pkl')
    with open(index_path, 'wb') as f:
        pickle.dump(index_pocket, f)

    # 保存失败的样本到单独的文件
    if len(failed_samples) > 0:
        failed_path = os.path.join(args.dest, 'failed_samples.pkl')
        with open(failed_path, 'wb') as f:
            pickle.dump(failed_samples, f)
        print(f'\n⚠️  失败样本已保存到: {failed_path}')

    print(f'\n✅ 完成！')
    print(f'   总样本数: {len(index)}')
    print(f'   成功提取: {success_count} ({success_count/len(index)*100:.1f}%)')
    print(f'   提取失败: {fail_count} ({fail_count/len(index)*100:.1f}%)')
    print(f'   索引文件: {index_path}')
    print(f'   索引中的样本数: {len(index_pocket)}')
    