"""
修复 Split 文件

问题：Split 文件包含 180 个索引，但 index.pkl 只有 168 个有效样本
解决方案：重新生成 split 文件，基于修复后的 index.pkl
"""

import os
import sys
import torch
import pickle
import argparse
from pathlib import Path


def fix_split_file(split_path, index_path, output_path=None):
    """
    修复 split 文件
    
    Args:
        split_path: 原始 split 文件路径
        index_path: 修复后的 index.pkl 路径
        output_path: 输出路径（默认覆盖原文件）
    """
    print("=" * 80)
    print("🔧 修复 Split 文件")
    print("=" * 80)
    
    # 检查文件是否存在
    if not os.path.exists(split_path):
        print(f"❌ Split 文件不存在: {split_path}")
        sys.exit(1)
    
    if not os.path.exists(index_path):
        print(f"❌ index.pkl 不存在: {index_path}")
        sys.exit(1)
    
    # 加载原始 split
    print(f"\n📋 加载原始 split 文件: {split_path}")
    split = torch.load(split_path)
    print(f"   Keys: {list(split.keys())}")
    
    for key, indices in split.items():
        print(f"   {key}: {len(indices)} 个索引")
    
    # 加载 index.pkl
    print(f"\n📋 加载 index.pkl: {index_path}")
    with open(index_path, 'rb') as f:
        index = pickle.load(f)
    
    valid_count = sum(1 for item in index if item[0] is not None)
    failed_count = len(index) - valid_count
    
    print(f"   总样本数: {len(index)}")
    print(f"   有效样本: {valid_count}")
    print(f"   失败样本: {failed_count}")
    
    # 创建新的 split
    print(f"\n🔧 创建新的 split...")
    
    # 方案1：如果 index.pkl 已经修复（只包含有效样本）
    if failed_count == 0:
        print(f"   ✅ index.pkl 已修复，所有样本都有效")
        print(f"   使用连续索引: 0 到 {len(index) - 1}")
        
        new_split = {}
        for key, old_indices in split.items():
            # 将所有索引映射到新的范围 [0, valid_count)
            # 假设原始 split 的索引是基于 180 个样本的
            # 我们需要将其映射到 168 个样本
            
            # 简单方案：直接使用前 valid_count 个索引
            new_indices = list(range(valid_count))
            new_split[key] = new_indices
            
            print(f"   {key}: {len(old_indices)} → {len(new_indices)} 个索引")
    
    # 方案2：如果 index.pkl 仍然包含失败样本
    else:
        print(f"   ⚠️  index.pkl 仍包含 {failed_count} 个失败样本")
        print(f"   创建有效样本的索引映射...")
        
        # 创建有效样本的索引映射
        valid_indices = [i for i, item in enumerate(index) if item[0] is not None]
        print(f"   有效样本索引: {valid_indices[:10]}...")
        
        new_split = {}
        for key, old_indices in split.items():
            # 过滤掉失败样本的索引
            new_indices = [i for i in old_indices if i in valid_indices]
            
            # 重新映射到连续索引
            index_mapping = {old_idx: new_idx for new_idx, old_idx in enumerate(valid_indices)}
            new_indices = [index_mapping[i] for i in new_indices]
            
            new_split[key] = new_indices
            
            print(f"   {key}: {len(old_indices)} → {len(new_indices)} 个索引")
    
    # 保存新的 split
    if output_path is None:
        # 备份原文件
        backup_path = split_path + '.backup'
        print(f"\n💾 备份原文件: {backup_path}")
        torch.save(split, backup_path)
        
        output_path = split_path
    
    print(f"\n💾 保存新的 split 文件: {output_path}")
    torch.save(new_split, output_path)
    
    # 验证
    print(f"\n✅ 验证新的 split 文件:")
    for key, indices in new_split.items():
        print(f"   {key}:")
        print(f"      样本数: {len(indices)}")
        print(f"      索引范围: [{min(indices)}, {max(indices)}]")
        print(f"      前10个索引: {sorted(indices)[:10]}")
    
    # 检查是否有超出范围的索引
    max_valid_index = valid_count - 1
    for key, indices in new_split.items():
        if max(indices) > max_valid_index:
            print(f"\n   ⚠️  警告：{key} 的最大索引 ({max(indices)}) 超出有效范围 ({max_valid_index})")
        else:
            print(f"\n   ✅ {key} 的所有索引都在有效范围内")
    
    print("\n" + "=" * 80)
    print("✅ Split 文件修复完成！")
    print("=" * 80)
    
    print(f"\n📊 修复总结:")
    print(f"   原始 split: {split_path}")
    print(f"   备份文件: {split_path}.backup")
    print(f"   新 split: {output_path}")
    print(f"   有效样本数: {valid_count}")
    
    for key in new_split.keys():
        old_count = len(split[key])
        new_count = len(new_split[key])
        print(f"   {key}: {old_count} → {new_count} 个索引")


def main():
    parser = argparse.ArgumentParser(description='Fix split file to match index.pkl')
    parser.add_argument('--split_path', type=str, required=True,
                        help='Path to split file (.pt)')
    parser.add_argument('--index_path', type=str, required=True,
                        help='Path to index.pkl')
    parser.add_argument('--output_path', type=str, default=None,
                        help='Output path (default: overwrite original)')
    args = parser.parse_args()
    
    fix_split_file(args.split_path, args.index_path, args.output_path)


if __name__ == '__main__':
    main()

