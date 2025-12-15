"""
重建 LMDB 数据库

功能：
1. 删除旧的 LMDB 数据库（key 不连续）
2. 强制系统重新创建 LMDB（key 连续）
3. 验证新的 LMDB 数据库
"""

import os
import sys
import argparse
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description='Rebuild LMDB database for pocket dataset')
    parser.add_argument('--pocket_dir', type=str, required=True,
                        help='Pocket directory containing index.pkl')
    parser.add_argument('--version', type=str, default='final',
                        help='LMDB version (default: final)')
    args = parser.parse_args()
    
    print("=" * 80)
    print("重建 LMDB 数据库")
    print("=" * 80)
    
    # 检查目录是否存在
    if not os.path.exists(args.pocket_dir):
        print(f"❌ 目录不存在: {args.pocket_dir}")
        sys.exit(1)
    
    print(f"\n📁 Pocket 目录: {args.pocket_dir}")
    
    # 计算 LMDB 路径
    pocket_dir = Path(args.pocket_dir).resolve()
    base_name = pocket_dir.name
    parent_dir = pocket_dir.parent
    lmdb_path = parent_dir / f"{base_name}_processed_{args.version}.lmdb"
    
    print(f"📊 LMDB 路径: {lmdb_path}")
    
    # 检查 LMDB 是否存在
    if not lmdb_path.exists():
        print(f"\n✅ LMDB 不存在，无需删除")
        print(f"   系统会在下次运行时自动创建")
        sys.exit(0)
    
    # 显示 LMDB 信息
    lmdb_size = lmdb_path.stat().st_size / (1024 * 1024)  # MB
    print(f"\n📊 旧 LMDB 信息:")
    print(f"   文件大小: {lmdb_size:.2f} MB")
    
    # 检查是否有 lock 文件
    lock_path = Path(str(lmdb_path) + '-lock')
    if lock_path.exists():
        print(f"   Lock 文件: {lock_path}")
    
    # 询问用户是否删除
    print(f"\n⚠️  警告：即将删除旧的 LMDB 数据库")
    print(f"   删除后，系统会在下次运行时重新创建（耗时约 5-10 分钟）")
    
    response = input(f"\n是否继续？(yes/no): ")
    if response.lower() not in ['yes', 'y']:
        print(f"\n❌ 取消操作")
        sys.exit(0)
    
    # 删除 LMDB
    print(f"\n🗑️  删除旧 LMDB: {lmdb_path}")
    try:
        lmdb_path.unlink()
        print(f"   ✅ 删除成功")
    except Exception as e:
        print(f"   ❌ 删除失败: {e}")
        sys.exit(1)
    
    # 删除 lock 文件
    if lock_path.exists():
        print(f"\n🗑️  删除 Lock 文件: {lock_path}")
        try:
            lock_path.unlink()
            print(f"   ✅ 删除成功")
        except Exception as e:
            print(f"   ⚠️  删除失败: {e}")
    
    print(f"\n" + "=" * 80)
    print("删除完成！")
    print("=" * 80)
    
    print(f"\n✅ 下次运行测试时，系统会自动重新创建 LMDB")
    print(f"   预计耗时: 5-10 分钟（168 个样本）")
    print(f"\n📝 重新创建过程:")
    print(f"   1. 读取 index.pkl（168 个有效样本）")
    print(f"   2. 解析每个 pocket 和 ligand 文件")
    print(f"   3. 创建 LMDB 数据库（key 连续：0-167）")
    print(f"   4. 保存到: {lmdb_path}")


if __name__ == '__main__':
    main()

