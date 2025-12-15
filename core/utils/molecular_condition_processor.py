"""
SOTA级分子条件数据预处理系统
==================================

核心功能：
1. 为每个训练分子计算QED、SA条件特征
2. 统计均值/方差并实现标准化
3. 保存到标注文件，支持快速加载
4. 与现有MolPilot数据流无缝集成

设计原则：
- 高效：支持批量处理和缓存机制
- 鲁棒：处理异常分子和边界情况
- 可扩展：支持新增条件类型
- 兼容：与现有数据集格式完全兼容
"""

import os
import json
import pickle
import numpy as np
import torch
from typing import Dict, List, Optional, Tuple, Union
from pathlib import Path
from tqdm import tqdm
import logging
from dataclasses import dataclass, asdict
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing as mp

try:
    # SOTA: 抑制RDKit的调试输出
    import warnings
    warnings.filterwarnings('ignore')

    from rdkit import Chem
    from rdkit.Chem import Descriptors, QED, Crippen
    from rdkit.Contrib.SA_Score import sascorer
    from rdkit import RDLogger
    RDLogger.DisableLog('rdApp.*')  # 抑制RDKit日志

    RDKIT_AVAILABLE = True
except ImportError:
    print("⚠️  RDKit未安装，将使用默认条件值")
    RDKIT_AVAILABLE = False


@dataclass
class MolecularConditions:
    """分子条件数据结构"""
    qed: float          # Drug-likeness (0-1)
    sa: float           # Synthetic Accessibility (0-1, 标准化)

    def to_tensor(self) -> torch.Tensor:
        """转换为张量格式"""
        return torch.tensor([self.qed, self.sa], dtype=torch.float32)

    @classmethod
    def from_tensor(cls, tensor: torch.Tensor) -> 'MolecularConditions':
        """从张量创建条件对象"""
        return cls(
            qed=tensor[0].item(),
            sa=tensor[1].item()
        )


@dataclass
class ConditionStatistics:
    """条件统计信息"""
    mean: Dict[str, float]
    std: Dict[str, float]
    min_val: Dict[str, float]
    max_val: Dict[str, float]
    count: int
    
    def normalize_conditions(self, conditions: MolecularConditions) -> MolecularConditions:
        """标准化条件"""
        return MolecularConditions(
            qed=conditions.qed,  # QED已经在0-1范围内
            sa=min(conditions.sa, 10.0) / 10.0  # SA标准化到0-1
        )


class MolecularConditionProcessor:
    """SOTA级分子条件处理器"""
    
    def __init__(self, cache_dir: str = "data/condition_cache",
                 enable_multiprocessing: bool = True,
                 max_workers: Optional[int] = None,
                 dataset_name: str = "default"):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        self.enable_multiprocessing = enable_multiprocessing
        self.max_workers = max_workers or min(8, mp.cpu_count())
        self.dataset_name = dataset_name

        # 统计信息
        self.statistics: Optional[ConditionStatistics] = None
        self.stats_file = self.cache_dir / f"{dataset_name}_condition_statistics.json"

        # SOTA: 持久化缓存文件（基于数据集名称）
        self.condition_cache_file = self.cache_dir / f"{dataset_name}_molecular_conditions.pkl"
        self.condition_cache: Dict[str, MolecularConditions] = {}

        # SOTA: 数据集级别的条件映射文件
        self.dataset_condition_file = self.cache_dir / f"{dataset_name}_dataset_conditions.json"
        self.dataset_conditions: Dict[str, List[float]] = {}  # {sample_id: [qed, sa]}

        # 加载已有缓存
        self._load_cache()
        self._load_statistics()
        self._load_dataset_conditions()

        print(f"✅ SOTA分子条件处理器初始化完成")
        print(f"   数据集: {dataset_name}")
        print(f"   缓存目录: {self.cache_dir}")
        print(f"   多进程: {self.enable_multiprocessing} (workers: {self.max_workers})")
        print(f"   已缓存SMILES条件: {len(self.condition_cache)}")
        print(f"   已缓存数据集条件: {len(self.dataset_conditions)}")
    
    def _load_cache(self):
        """加载条件缓存"""
        if self.condition_cache_file.exists():
            try:
                with open(self.condition_cache_file, 'rb') as f:
                    self.condition_cache = pickle.load(f)
                print(f"📊 加载条件缓存: {len(self.condition_cache)} 个分子")
            except Exception as e:
                print(f"⚠️  条件缓存加载失败: {e}")
                self.condition_cache = {}
    
    def _save_cache(self):
        """保存条件缓存"""
        try:
            with open(self.condition_cache_file, 'wb') as f:
                pickle.dump(self.condition_cache, f)
            print(f"💾 条件缓存已保存: {len(self.condition_cache)} 个分子")
        except Exception as e:
            print(f"⚠️  条件缓存保存失败: {e}")
    
    def _load_statistics(self):
        """加载统计信息"""
        if self.stats_file.exists():
            try:
                with open(self.stats_file, 'r') as f:
                    stats_dict = json.load(f)
                    self.statistics = ConditionStatistics(**stats_dict)
                print(f"📈 加载条件统计信息: {self.statistics.count} 个样本")
            except Exception as e:
                print(f"⚠️  统计信息加载失败: {e}")
                self.statistics = None
    
    def _save_statistics(self):
        """保存统计信息"""
        if self.statistics is not None:
            try:
                with open(self.stats_file, 'w') as f:
                    json.dump(asdict(self.statistics), f, indent=2)
                print(f"📊 统计信息已保存: {self.statistics.count} 个样本")
            except Exception as e:
                print(f"⚠️  统计信息保存失败: {e}")
    
    @staticmethod
    def _calculate_single_condition(smiles: str) -> Optional[MolecularConditions]:
        """计算单个分子的条件（静态方法，支持多进程）"""
        if not RDKIT_AVAILABLE:
            return MolecularConditions(qed=0.5, sa=0.5)

        try:
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                return None

            # 计算QED和SA
            qed_value = QED.qed(mol)
            sa_value = sascorer.calculateScore(mol)

            return MolecularConditions(
                qed=qed_value,
                sa=sa_value
            )

        except Exception as e:
            print(f"⚠️  分子条件计算失败 {smiles}: {e}")
            return None
    
    def calculate_conditions(self, smiles: Union[str, List[str]], 
                           use_cache: bool = True) -> Union[MolecularConditions, List[MolecularConditions]]:
        """计算分子条件"""
        is_single = isinstance(smiles, str)
        smiles_list = [smiles] if is_single else smiles
        
        results = []
        to_calculate = []
        
        # 检查缓存
        for smi in smiles_list:
            if use_cache and smi in self.condition_cache:
                results.append(self.condition_cache[smi])
            else:
                results.append(None)
                to_calculate.append((len(results) - 1, smi))
        
        # 计算未缓存的条件
        if to_calculate:
            print(f"📊 需要计算的分子数: {len(to_calculate)}")
            print(f"📊 结果列表长度: {len(results)}")

            if self.enable_multiprocessing and len(to_calculate) > 1:
                # SOTA: 修复多进程计算的索引错误
                print(f"🔄 使用多进程计算 (workers: {self.max_workers})")
                with ProcessPoolExecutor(max_workers=self.max_workers) as executor:
                    # 创建future到(结果索引, SMILES)的映射
                    future_to_data = {
                        executor.submit(self._calculate_single_condition, smi): (idx, smi)
                        for idx, smi in to_calculate
                    }
                    print(f"📊 提交的任务数: {len(future_to_data)}")

                    for future in tqdm(as_completed(future_to_data),
                                     total=len(to_calculate),
                                     desc="计算分子条件"):
                        try:
                            # SOTA: 安全获取映射数据
                            if future not in future_to_data:
                                print(f"⚠️  Future映射丢失，跳过")
                                continue

                            idx, smi = future_to_data[future]

                            # 验证索引有效性
                            if idx >= len(results):
                                print(f"⚠️  索引越界 {idx} >= {len(results)}，跳过")
                                continue

                            condition = future.result()
                            if condition is not None:
                                results[idx] = condition
                                if use_cache:
                                    self.condition_cache[smi] = condition
                            else:
                                # 使用默认值
                                default_condition = MolecularConditions(qed=0.5, sa=0.5)
                                results[idx] = default_condition
                                if use_cache:
                                    self.condition_cache[smi] = default_condition

                        except Exception as e:
                            print(f"⚠️  条件计算异常: {e}")
                            # 尝试获取索引，如果失败则跳过
                            try:
                                idx, smi = future_to_data.get(future, (None, None))
                                if idx is not None and idx < len(results):
                                    default_condition = MolecularConditions(qed=0.5, sa=0.5)
                                    results[idx] = default_condition
                                    if use_cache:
                                        self.condition_cache[smi] = default_condition
                            except:
                                pass  # 如果无法恢复，跳过这个样本
            else:
                # SOTA: 单进程计算（带安全检查）
                for idx, smi in tqdm(to_calculate, desc="计算分子条件"):
                    try:
                        # 验证索引有效性
                        if idx >= len(results):
                            print(f"⚠️  单进程索引越界 {idx} >= {len(results)}，跳过")
                            continue

                        condition = self._calculate_single_condition(smi)
                        if condition is not None:
                            results[idx] = condition
                            if use_cache:
                                self.condition_cache[smi] = condition
                        else:
                            # 使用默认值
                            default_condition = MolecularConditions(qed=0.5, sa=0.5)
                            results[idx] = default_condition
                            if use_cache:
                                self.condition_cache[smi] = default_condition
                    except Exception as e:
                        print(f"⚠️  单进程条件计算异常 (idx={idx}): {e}")
                        if idx < len(results):
                            default_condition = MolecularConditions(qed=0.5, sa=0.5)
                            results[idx] = default_condition
                            if use_cache:
                                self.condition_cache[smi] = default_condition
        
        # 保存缓存
        if use_cache and to_calculate:
            self._save_cache()
        
        return results[0] if is_single else results

    def _load_dataset_conditions(self):
        """加载数据集级别的条件映射"""
        if self.dataset_condition_file.exists():
            try:
                with open(self.dataset_condition_file, 'r') as f:
                    self.dataset_conditions = json.load(f)
                print(f"📊 加载数据集条件映射: {len(self.dataset_conditions)} 个样本")
            except Exception as e:
                print(f"⚠️  数据集条件映射加载失败: {e}")
                self.dataset_conditions = {}

    def _save_dataset_conditions(self):
        """保存数据集级别的条件映射"""
        try:
            with open(self.dataset_condition_file, 'w') as f:
                json.dump(self.dataset_conditions, f, indent=2)
            print(f"💾 数据集条件映射已保存: {len(self.dataset_conditions)} 个样本")
        except Exception as e:
            print(f"⚠️  数据集条件映射保存失败: {e}")

    def precompute_dataset_conditions(self, dataset, sample_id_key: str = 'ligand_filename',
                                    smiles_key: str = 'ligand_smiles', force_recompute: bool = False):
        """SOTA: 预计算整个数据集的条件，建立sample_id到条件的映射"""
        print(f"🔄 预计算数据集条件: {len(dataset)} 个样本")

        if not force_recompute and len(self.dataset_conditions) >= len(dataset) * 0.9:
            print(f"📊 数据集条件已预计算，跳过")
            return

        # 批量提取SMILES
        smiles_list = []
        sample_ids = []
        failed_count = 0

        print(f"📊 开始提取SMILES和样本ID...")

        # SOTA: 抑制可能的调试输出
        import sys
        import os

        # 临时重定向stderr来抑制可能的调试输出
        original_stderr = sys.stderr
        try:
            # 创建一个空的文件对象来抑制输出
            devnull = open(os.devnull, 'w')

            for i, data in enumerate(tqdm(dataset, desc="提取SMILES")):
                try:
                    # 临时抑制stderr输出
                    sys.stderr = devnull

                    # 获取样本ID
                    sample_id = None
                    if hasattr(data, sample_id_key):
                        sample_id = getattr(data, sample_id_key)
                    elif hasattr(data, 'ligand_filename'):
                        sample_id = data.ligand_filename
                    elif hasattr(data, 'filename'):
                        sample_id = data.filename
                    else:
                        sample_id = f"sample_{i}"

                    # 获取SMILES
                    smiles = None
                    if hasattr(data, smiles_key):
                        smiles = getattr(data, smiles_key)
                    elif hasattr(data, 'ligand_smiles'):
                        smiles = data.ligand_smiles
                    elif hasattr(data, 'smiles'):
                        smiles = data.smiles

                    # 恢复stderr
                    sys.stderr = original_stderr

                    if smiles and smiles.strip():
                        smiles_list.append(smiles.strip())
                        sample_ids.append(sample_id)
                    else:
                        # 如果没有SMILES，使用默认条件
                        self.dataset_conditions[sample_id] = [0.5, 0.5]
                        failed_count += 1

                except Exception as e:
                    # 恢复stderr
                    sys.stderr = original_stderr
                    # 只在真正需要时输出错误信息
                    if i % 1000 == 0:  # 每1000个样本输出一次错误
                        print(f"⚠️  样本 {i} 数据提取失败: {e}")
                    sample_id = f"sample_{i}"
                    self.dataset_conditions[sample_id] = [0.5, 0.5]
                    failed_count += 1

        finally:
            # 确保恢复stderr
            sys.stderr = original_stderr
            devnull.close()

        print(f"📊 SMILES提取完成:")
        print(f"   成功提取: {len(smiles_list)} 个")
        print(f"   使用默认: {failed_count} 个")

        # 批量计算条件
        if smiles_list:
            print(f"🔄 批量计算 {len(smiles_list)} 个分子的条件")
            conditions_list = self.calculate_conditions(smiles_list, use_cache=True)

            # 建立映射
            success_count = 0
            for sample_id, conditions in zip(sample_ids, conditions_list):
                if conditions is not None:
                    normalized = self.normalize_conditions(conditions)
                    self.dataset_conditions[sample_id] = [normalized.qed, normalized.sa]
                    success_count += 1
                else:
                    self.dataset_conditions[sample_id] = [0.5, 0.5]

            print(f"📊 条件计算完成:")
            print(f"   成功计算: {success_count} 个")
            print(f"   使用默认: {len(sample_ids) - success_count} 个")

        # 保存映射
        self._save_dataset_conditions()
        print(f"✅ 数据集条件预计算完成: {len(self.dataset_conditions)} 个样本")

    def get_conditions_by_sample_id(self, sample_id: str) -> torch.Tensor:
        """根据样本ID获取预计算的条件"""
        if sample_id in self.dataset_conditions:
            return torch.tensor(self.dataset_conditions[sample_id], dtype=torch.float32)
        else:
            # SOTA: 如果没有预计算条件，记录并返回默认条件
            # 这种情况下Transform应该回退到SMILES计算
            return torch.tensor([0.5, 0.5], dtype=torch.float32)
    
    def normalize_conditions(self, conditions: MolecularConditions) -> MolecularConditions:
        """🔥 统一的条件标准化方案，与训练数据生成保持一致"""
        # 🚨 关键修复：使用与数据生成一致的SA反向归一化
        # 训练数据使用: (10-SA)/9，高值=易合成
        # 这里也必须使用相同的逻辑
        sa_raw = conditions.sa
        if sa_raw < 1.0:
            sa_raw = 1.0
        elif sa_raw > 10.0:
            sa_raw = 10.0
        sa_normalized = (10.0 - sa_raw) / 9.0  # 反向归一化，与训练数据一致

        return MolecularConditions(
            qed=conditions.qed,  # QED已经在0-1范围内
            sa=sa_normalized  # 🚨 修复：使用反向归一化
        )

    @staticmethod
    def denormalize_sa(sa_normalized: float) -> float:
        """🚨 修复：反标准化SA值，使用与训练数据一致的反向映射"""
        # 训练数据使用: sa_normalized = (10 - sa_raw) / 9
        # 反向计算: sa_raw = 10 - sa_normalized * 9
        return 10.0 - sa_normalized * 9.0

    @staticmethod
    def validate_target_conditions(target_qed: float, target_sa: float) -> tuple:
        """🔥 统一的目标条件验证和标准化，与训练数据保持一致"""
        # QED验证：0-1范围
        validated_qed = max(0.0, min(1.0, target_qed))

        # 🚨 SA验证：使用与训练数据一致的反向归一化
        if target_sa > 1.0:
            # 假设输入的是原始SA值（如3.5），需要反向标准化
            sa_raw = min(target_sa, 10.0)
            validated_sa = (10.0 - sa_raw) / 9.0  # 反向归一化
            print(f"🔄 SA值反向标准化: 原始SA={target_sa} -> 标准化SA={validated_sa:.3f}")
            print(f"   解释: SA={target_sa}(较难合成) -> 标准化值={validated_sa:.3f}(低值=难合成)")
        else:
            # 假设输入的已经是标准化值，但需要验证是否符合反向归一化逻辑
            validated_sa = max(0.0, min(1.0, target_sa))
            implied_sa_raw = 10.0 - validated_sa * 9.0
            print(f"🔄 SA值验证: 标准化SA={target_sa} -> 对应原始SA≈{implied_sa_raw:.1f}")

        return validated_qed, validated_sa
    
    def get_normalized_conditions(self, smiles: Union[str, List[str]]) -> Union[torch.Tensor, List[torch.Tensor]]:
        """获取标准化的条件张量"""
        conditions = self.calculate_conditions(smiles, use_cache=True)
        
        if isinstance(conditions, list):
            normalized = [self.normalize_conditions(c) for c in conditions]
            return [c.to_tensor() for c in normalized]
        else:
            normalized = self.normalize_conditions(conditions)
            return normalized.to_tensor()
    
    def get_default_conditions(self) -> MolecularConditions:
        """获取默认条件"""
        return MolecularConditions(qed=0.5, sa=0.5)
    
    def get_default_normalized_tensor(self) -> torch.Tensor:
        """获取默认标准化条件张量"""
        default_conditions = self.get_default_conditions()
        normalized = self.normalize_conditions(default_conditions)
        return normalized.to_tensor()


def create_condition_processor(cache_dir: str = "data/condition_cache",
                             enable_multiprocessing: bool = True,
                             max_workers: Optional[int] = None,
                             dataset_name: str = "default") -> MolecularConditionProcessor:
    """创建条件处理器的工厂函数"""
    return MolecularConditionProcessor(
        cache_dir=cache_dir,
        enable_multiprocessing=enable_multiprocessing,
        max_workers=max_workers,
        dataset_name=dataset_name
    )


# 全局条件处理器实例（单例模式，支持多数据集）
_global_processors: Dict[str, MolecularConditionProcessor] = {}

def get_global_condition_processor(dataset_name: str = "default") -> MolecularConditionProcessor:
    """获取全局条件处理器实例"""
    global _global_processors
    if dataset_name not in _global_processors:
        _global_processors[dataset_name] = create_condition_processor(dataset_name=dataset_name)
    return _global_processors[dataset_name]
