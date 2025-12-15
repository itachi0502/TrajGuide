#!/usr/bin/env python3
"""
增强的VinaScore计算模块
实现本地精修、盒子校准、柔性残基等高级对接选项
"""

import os
import tempfile
import numpy as np
from pathlib import Path
from typing import Optional, Dict, List, Tuple, Union

try:
    from vina import Vina
    VINA_AVAILABLE = True
except ImportError:
    VINA_AVAILABLE = False

from .docking_vina import VinaDock
from .utils.vina_compat import create_compatible_vina


class EnhancedVinaScoring:
    """
    增强的VinaScore计算，支持多种优化策略
    """
    
    def __init__(self, receptor_path: str, ligand_path: str, 
                 pocket_center: Optional[List[float]] = None,
                 box_size: Optional[List[float]] = None,
                 ref_ligand_path: Optional[str] = None):
        """
        初始化增强的Vina评分器
        
        Args:
            receptor_path: 受体蛋白路径
            ligand_path: 配体分子路径
            pocket_center: 口袋中心坐标 [x, y, z]
            box_size: 盒子大小 [size_x, size_y, size_z]
            ref_ligand_path: 参考配体路径（用于自动定盒）
        """
        self.receptor_path = receptor_path
        self.ligand_path = ligand_path
        self.ref_ligand_path = ref_ligand_path
        
        # 初始化基础对接器
        self.base_docker = VinaDock(ligand_path, receptor_path)
        
        # 设置盒子参数
        if pocket_center and box_size:
            self.pocket_center = pocket_center
            self.box_size = box_size
        elif ref_ligand_path:
            self._auto_define_box_from_reference()
        else:
            # 使用默认盒子
            self.base_docker.get_box()
            self.pocket_center = self.base_docker.pocket_center
            self.box_size = self.base_docker.box_size
    
    def _auto_define_box_from_reference(self, buffer: float = 2.0):

        if not self.ref_ligand_path or not os.path.exists(self.ref_ligand_path):
            self.base_docker.get_box()
            self.pocket_center = self.base_docker.pocket_center
            self.box_size = self.base_docker.box_size
            return
        
        try:
            self.base_docker.get_box(ref=self.ref_ligand_path, buffer=buffer)
            self.pocket_center = self.base_docker.pocket_center
            self.box_size = self.base_docker.box_size
        except Exception as e:
            self.base_docker.get_box()
            self.pocket_center = self.base_docker.pocket_center
            self.box_size = self.base_docker.box_size
    
    def standard_docking(self, exhaustiveness: int = 8, **kwargs) -> float:

        try:
            score = self.base_docker.dock(
                mode='dock', 
                exhaustiveness=exhaustiveness, 
                **kwargs
            )
            return score
        except Exception as e:
            return float('nan')
    
    def local_refinement(self, exhaustiveness: int = 8, **kwargs) -> float:

        try:
            scores = {}

            try:
                score_only = self.base_docker.dock(mode='score_only', **kwargs)
                scores['score_only'] = score_only
            except Exception as e:
                scores['score_only'] = float('nan')

            try:
                minimize_score = self.base_docker.dock(mode='minimize', **kwargs)
                scores['minimize'] = minimize_score
            except Exception as e:
                scores['minimize'] = float('nan')

            try:
                dock_score = self.base_docker.dock(
                    mode='dock',
                    exhaustiveness=exhaustiveness,
                    **kwargs
                )
                scores['dock'] = dock_score
            except Exception as e:
                scores['dock'] = float('nan')

            refined_score = self._select_best_score(scores)

            return refined_score

        except Exception as e:
            return self.standard_docking(exhaustiveness, **kwargs)

    def _select_best_score(self, scores: Dict[str, float]) -> float:

        valid_scores = {k: v for k, v in scores.items() if not np.isnan(v)}

        if not valid_scores:
            return float('nan')

        score_only = valid_scores.get('score_only', float('nan'))
        minimize = valid_scores.get('minimize', float('nan'))
        dock = valid_scores.get('dock', float('nan'))

        if not np.isnan(score_only) and score_only > 10:
            valid_scores.pop('score_only', None)

        if not np.isnan(dock) and not np.isnan(minimize):
            if abs(dock - minimize) < 2.0:
                best_score = dock
                method = "dock (一致性验证)"
            else:
                best_score = min(dock, minimize)
                method = "dock/minimize (最优)"
        elif not np.isnan(dock):
            best_score = dock
            method = "dock (单一)"
        elif not np.isnan(minimize):
            best_score = minimize
            method = "minimize (单一)"
        elif not np.isnan(score_only) and score_only <= 10:
            best_score = score_only
            method = "score_only (备用)"
        else:
            return float('nan')

        return best_score
    
    def calibrated_box_docking(self, box_expansion: float = 1.0, 
                             exhaustiveness: int = 8, **kwargs) -> float:
        try:
            original_center = self.pocket_center.copy()
            original_size = self.box_size.copy()
            
            best_score = float('inf')
            best_config = None
            
            size_factors = [0.9, 1.0, box_expansion]
            
            for factor in size_factors:
                try:
                    adjusted_size = [s * factor for s in original_size]
                    
                    temp_docker = VinaDock(self.ligand_path, self.receptor_path)
                    temp_docker.pocket_center = original_center
                    temp_docker.box_size = adjusted_size
                    
                    score = temp_docker.dock(
                        mode='dock',
                        exhaustiveness=exhaustiveness,
                        **kwargs
                    )
                    
                    if score < best_score:
                        best_score = score
                        best_config = (original_center, adjusted_size)

                    
                except Exception as e:
                    continue
            
            if best_config:
                self.pocket_center, self.box_size = best_config
                return best_score
            else:
                return self.standard_docking(exhaustiveness, **kwargs)
                
        except Exception as e:
            return self.standard_docking(exhaustiveness, **kwargs)
    
    def multi_strategy_scoring(self, strategies: List[str] = None, 
                             exhaustiveness: int = 8, **kwargs) -> Dict[str, float]:

        if strategies is None:
            strategies = ['standard', 'local_refine', 'calibrated_box']
        
        results = {}
        
        for strategy in strategies:
            try:
                if strategy == 'standard':
                    score = self.standard_docking(exhaustiveness, **kwargs)
                elif strategy == 'local_refine':
                    score = self.local_refinement(exhaustiveness, **kwargs)
                elif strategy == 'calibrated_box':
                    score = self.calibrated_box_docking(1.1, exhaustiveness, **kwargs)
                else:
                    continue
                
                results[strategy] = score
                
            except Exception as e:
                results[strategy] = float('nan')
        
        valid_results = {k: v for k, v in results.items() if not np.isnan(v)}
        if valid_results:
            best_strategy = min(valid_results.keys(), key=lambda k: valid_results[k])
            best_score = valid_results[best_strategy]
            results['best_strategy'] = best_strategy
            results['best_score'] = best_score
        
        return results
    
    def enhanced_scoring(self, use_local_refine: bool = True,
                        use_box_calibration: bool = True,
                        exhaustiveness: int = 8, **kwargs) -> float:
        strategies = ['standard']
        
        if use_local_refine:
            strategies.append('local_refine')
        
        if use_box_calibration:
            strategies.append('calibrated_box')
        
        results = self.multi_strategy_scoring(strategies, exhaustiveness, **kwargs)
        
        return results.get('best_score', float('nan'))


def enhance_vina_score(ligand_path: str, receptor_path: str,
                      pocket_center: Optional[List[float]] = None,
                      box_size: Optional[List[float]] = None,
                      ref_ligand_path: Optional[str] = None,
                      enhancement_level: str = 'moderate') -> float:

    if not VINA_AVAILABLE:
        return float('nan')

    try:
        scorer = EnhancedVinaScoring(
            receptor_path=receptor_path,
            ligand_path=ligand_path,
            pocket_center=pocket_center,
            box_size=box_size,
            ref_ligand_path=ref_ligand_path
        )

        if enhancement_level == 'basic':
            return scorer.local_refinement(exhaustiveness=8)
        elif enhancement_level == 'moderate':
            return scorer.local_refinement(exhaustiveness=12)
        elif enhancement_level == 'aggressive':
            refined_score = scorer.local_refinement(exhaustiveness=16)

            if not np.isnan(refined_score) and -12 < refined_score < -3:
                calibrated_score = scorer.calibrated_box_docking(
                    box_expansion=1.1,
                    exhaustiveness=16
                )
                if not np.isnan(calibrated_score) and calibrated_score < refined_score:
                    return calibrated_score


            return refined_score
        else:
            return scorer.local_refinement()

    except Exception as e:
        return float('nan')
