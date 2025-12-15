"""
修复 Pocket Index 文件

功能：
1. 从 index.pkl 中移除失败的样本（pocket_fn 为 None）
2. 保存失败的样本到单独的文件
3. 更新 index.pkl 只包含成功的样本
"""

import os
import sys
import pickle
import argparse
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description='Fix pocket index file by removing failed samples')
    parser.add_argument('--pocket_dir', type=str, required=True,
                        help='Pocket directory containing index.pkl')
    args = parser.parse_args()
    
    print("=" * 80)
    print("修复 Pocket Index 文件")
    print("=" * 80)
    
    # 检查目录是否存在
    if not os.path.exists(args.pocket_dir):
        print(f"❌ 目录不存在: {args.pocket_dir}")
        sys.exit(1)
    
    print(f"\n📁 Pocket 目录: {args.pocket_dir}")
    
    # 加载原始 index.pkl
    index_path = os.path.join(args.pocket_dir, 'index.pkl')
    if not os.path.exists(index_path):
        print(f"❌ 索引文件不存在: {index_path}")
        sys.exit(1)
    
    print(f"\n📋 加载索引文件: {index_path}")
    
    with open(index_path, 'rb') as f:
        index = pickle.load(f)
    
    print(f"   原始样本数: {len(index)}")
    
    # 分离成功和失败的样本
    success_samples = []
    failed_samples = []
    
    for item in index:
        pocket_fn = item[0]
        if pocket_fn is not None:
            success_samples.append(item)
        else:
            failed_samples.append(item)
    
    print(f"\n📊 统计:")
    print(f"   成功样本: {len(success_samples)} ({len(success_samples)/len(index)*100:.1f}%)")
    print(f"   失败样本: {len(failed_samples)} ({len(failed_samples)/len(index)*100:.1f}%)")
    
    # 如果没有失败的样本，直接退出
    if len(failed_samples) == 0:
        print(f"\n✅ 索引文件已经是正确的，无需修复！")
        sys.exit(0)
    
    # 显示失败的样本
    print(f"\n⚠️  失败样本列表:")
    for i, item in enumerate(failed_samples[:10]):  # 只显示前10个
        ligand_fn = item[1] if len(item) > 1 else 'Unknown'
        protein_fn = item[2] if len(item) > 2 else 'Unknown'
        print(f"   {i+1}. {ligand_fn}")
    if len(failed_samples) > 10:
        print(f"   ... 还有 {len(failed_samples) - 10} 个失败样本")
    
    # 备份原始 index.pkl
    backup_path = os.path.join(args.pocket_dir, 'index.pkl.backup')
    print(f"\n💾 备份原始索引文件到: {backup_path}")
    
    import shutil
    shutil.copy(index_path, backup_path)
    
    # 保存修复后的 index.pkl（只包含成功的样本）
    print(f"\n✅ 保存修复后的索引文件: {index_path}")
    
    with open(index_path, 'wb') as f:
        pickle.dump(success_samples, f)
    
    print(f"   新索引样本数: {len(success_samples)}")
    
    # 保存失败的样本到单独的文件
    failed_path = os.path.join(args.pocket_dir, 'failed_samples.pkl')
    print(f"\n⚠️  保存失败样本到: {failed_path}")
    
    with open(failed_path, 'wb') as f:
        pickle.dump(failed_samples, f)
    
    print(f"\n" + "=" * 80)
    print("修复完成！")
    print("=" * 80)
    
    print(f"\n📊 修复总结:")
    print(f"   原始样本数: {len(index)}")
    print(f"   成功样本数: {len(success_samples)}")
    print(f"   失败样本数: {len(failed_samples)}")
    print(f"   修复后索引: {index_path}")
    print(f"   备份文件: {backup_path}")
    print(f"   失败样本: {failed_path}")
    
    print(f"\n✅ 现在可以重新运行测试了！")


if __name__ == '__main__':
    main()

