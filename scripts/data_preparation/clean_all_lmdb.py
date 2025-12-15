"""
清理所有相关的 LMDB 数据库

功能：
1. 查找指定 pocket 目录的所有 LMDB 文件
2. 显示每个 LMDB 的详细信息
3. 删除所有 LMDB 文件（或指定版本）
4. 强制系统重新创建 LMDB
"""

import os
import sys
import argparse
from pathlib import Path


def find_all_lmdb_files(pocket_dir):
    """查找所有相关的 LMDB 文件"""
    pocket_dir = Path(pocket_dir).resolve()
    base_name = pocket_dir.name
    parent_dir = pocket_dir.parent
    
    # 查找所有匹配的 LMDB 文件
    pattern = f"{base_name}_processed_*.lmdb"
    lmdb_files = list(parent_dir.glob(pattern))
    
    # 同时查找 lock 文件
    lmdb_info = []
    for lmdb_file in lmdb_files:
        lock_file = Path(str(lmdb_file) + '-lock')
        lmdb_info.append({
            'lmdb': lmdb_file,
            'lock': lock_file if lock_file.exists() else None,
            'size': lmdb_file.stat().st_size / (1024 * 1024),  # MB
            'version': lmdb_file.stem.split('_processed_')[-1]
        })
    
    return lmdb_info


def main():
    parser = argparse.ArgumentParser(description='Clean all LMDB databases for pocket dataset')
    parser.add_argument('--pocket_dir', type=str, required=True,
                        help='Pocket directory containing index.pkl')
    parser.add_argument('--version', type=str, default=None,
                        help='Only clean specific version (e.g., kekulize, final). If not specified, clean all versions.')
    parser.add_argument('--dry_run', action='store_true',
                        help='Dry run mode: show what would be deleted without actually deleting')
    args = parser.parse_args()
    
    print("=" * 80)
    print("🧹 清理 LMDB 数据库")
    print("=" * 80)
    
    # 检查目录是否存在
    if not os.path.exists(args.pocket_dir):
        print(f"❌ 目录不存在: {args.pocket_dir}")
        sys.exit(1)
    
    print(f"\n📁 Pocket 目录: {args.pocket_dir}")
    
    # 查找所有 LMDB 文件
    lmdb_info = find_all_lmdb_files(args.pocket_dir)
    
    if not lmdb_info:
        print(f"\n✅ 未找到任何 LMDB 文件，无需清理")
        sys.exit(0)
    
    # 过滤指定版本
    if args.version:
        lmdb_info = [info for info in lmdb_info if info['version'] == args.version]
        if not lmdb_info:
            print(f"\n✅ 未找到版本 '{args.version}' 的 LMDB 文件，无需清理")
            sys.exit(0)
    
    # 显示找到的 LMDB 文件
    print(f"\n📊 找到 {len(lmdb_info)} 个 LMDB 文件:")
    print("")
    
    total_size = 0
    for i, info in enumerate(lmdb_info, 1):
        print(f"{i}. {info['lmdb'].name}")
        print(f"   版本: {info['version']}")
        print(f"   大小: {info['size']:.2f} MB")
        if info['lock']:
            print(f"   Lock: {info['lock'].name}")
        print("")
        total_size += info['size']
    
    print(f"总大小: {total_size:.2f} MB")
    
    # Dry run 模式
    if args.dry_run:
        print(f"\n🔍 Dry Run 模式：以下文件将被删除（实际未删除）:")
        for info in lmdb_info:
            print(f"   - {info['lmdb']}")
            if info['lock']:
                print(f"   - {info['lock']}")
        print(f"\n✅ Dry Run 完成！使用 --dry_run=false 执行实际删除")
        sys.exit(0)
    
    # 询问用户确认
    print(f"\n⚠️  警告：即将删除以上 LMDB 文件")
    print(f"   删除后，系统会在下次运行时重新创建（耗时约 5-10 分钟）")
    
    response = input(f"\n是否继续？(yes/no): ")
    if response.lower() not in ['yes', 'y']:
        print(f"\n❌ 取消操作")
        sys.exit(0)
    
    # 删除 LMDB 文件
    print(f"\n🗑️  开始删除...")
    print("")
    
    deleted_count = 0
    failed_count = 0
    
    for info in lmdb_info:
        # 删除 LMDB
        try:
            print(f"🗑️  删除: {info['lmdb'].name}")
            info['lmdb'].unlink()
            print(f"   ✅ 删除成功")
            deleted_count += 1
        except Exception as e:
            print(f"   ❌ 删除失败: {e}")
            failed_count += 1
        
        # 删除 lock 文件
        if info['lock']:
            try:
                print(f"🗑️  删除: {info['lock'].name}")
                info['lock'].unlink()
                print(f"   ✅ 删除成功")
            except Exception as e:
                print(f"   ⚠️  删除失败: {e}")
        
        print("")
    
    # 总结
    print("=" * 80)
    print("清理完成！")
    print("=" * 80)
    
    print(f"\n📊 清理总结:")
    print(f"   成功删除: {deleted_count} 个 LMDB 文件")
    if failed_count > 0:
        print(f"   删除失败: {failed_count} 个 LMDB 文件")
    print(f"   释放空间: {total_size:.2f} MB")
    
    print(f"\n✅ 下次运行测试时，系统会自动重新创建 LMDB")
    print(f"   预计耗时: 5-10 分钟（基于样本数量）")
    
    print(f"\n📝 重新创建过程:")
    print(f"   1. 读取 index.pkl")
    print(f"   2. 解析每个 pocket 和 ligand 文件")
    print(f"   3. 创建 LMDB 数据库（key 连续）")
    print(f"   4. 保存到: {args.pocket_dir}_processed_<version>.lmdb")


if __name__ == '__main__':
    main()

