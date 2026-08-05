
from typing import List, Tuple, Dict, Set
from dataclasses import dataclass
from enum import Enum
import numpy as np
from scipy.optimize import linear_sum_assignment


class MatchStrategy(Enum):
    """匹配策略枚举"""
    GREEDY_TWO_ROUND = "greedy_two_round"  # 两轮贪心匹配
    HUNGARIAN = "hungarian"  # 匈牙利算法匹配


@dataclass
class HNMRPeak:
    """1H NMR峰数据结构"""
    shift_min: float
    shift_max: float
    multiplicity: str
    integration: str  # 可以是数字或表达式如"3H"
    coupling_constants: List[float]
    
    @property
    def shift_center(self) -> float:
        """计算化学位移中心值"""
        return (self.shift_min + self.shift_max) / 2
    
    @property
    def shift_range(self) -> float:
        """计算化学位移范围"""
        return abs(self.shift_max - self.shift_min)
    
    @property
    def n_hydrogen(self) -> float:
        """解析积分值,提取氢数"""
        if isinstance(self.integration, (int, float)):
            return float(self.integration)
        # 处理"3H", "2.5H"等格式
        integration_str = str(self.integration).upper().replace('H', '').strip()
        try:
            return float(integration_str) if integration_str else 1.0
        except:
            return 1.0


def distance_penalty(distance: float, scale: float) -> float:
    """
    基于距离计算相似度惩罚
    使用高斯衰减函数
    
    Args:
        distance: 化学位移距离
        scale: 容忍距离
        
    Returns:
        相似度分数 [0, 1]
    """
    # sigma = scale / 2
    # return np.exp(-(distance ** 2) / (2 * sigma ** 2))
    return np.exp(-distance / scale)


class PeakMatcher:
    """
    光谱峰匹配器
    支持两种匹配策略:
    1. 两轮贪心匹配 (GREEDY_TWO_ROUND)
    2. 匈牙利算法匹配 (HUNGARIAN)
    """
    
    def __init__(self, scale: float, tolerance: float, strategy: MatchStrategy = MatchStrategy.HUNGARIAN, cost_sim: bool = False):
        """
        Args:
            tolerance: 容忍距离,超过此距离的匹配视为无效
            strategy: 匹配策略
        """
        self.scale = scale
        self.tolerance = tolerance
        self.strategy = strategy
        self.cost_sim = cost_sim  
    
    def match_peaks(self, shifts_A: List[float], shifts_B: List[float]) -> List[Tuple[int, int, float, bool]]:
        """
        基于化学位移进行峰匹配
        
        Args:
            shifts_A: 光谱A的化学位移列表
            shifts_B: 光谱B的化学位移列表
            
        Returns:
            匹配列表,每个元素为(idx_A, idx_B, distance, is_valid)
        """
        if len(shifts_A) == 0 or len(shifts_B) == 0:
            return []
        
        if self.strategy == MatchStrategy.GREEDY_TWO_ROUND:
            return self._greedy_two_round_match(shifts_A, shifts_B)
        elif self.strategy == MatchStrategy.HUNGARIAN:
            return self._hungarian_match(shifts_A, shifts_B)
        else:
            raise ValueError(f"Unknown matching strategy: {self.strategy}")
    
    def _greedy_two_round_match(self, shifts_A: List[float], shifts_B: List[float]) -> List[Tuple[int, int, float, bool]]:
        """
        两轮贪心匹配策略
        
        第1轮: A中每个峰找B中最近的峰进行匹配
        第2轮: B中未被匹配的峰找A中最近的峰进行匹配
        """
        matches = []
        matched_B = set()
        
        # 第1轮: A找B
        for idx_a, shift_a in enumerate(shifts_A):
            distances = [abs(shift_a - shift_b) for shift_b in shifts_B]
            idx_b = int(np.argmin(distances))
            distance = distances[idx_b]
            is_valid = distance <= self.tolerance
            sim = distance_penalty(distance, self.scale)
            
            matches.append((1, idx_a, idx_b, shifts_A[idx_a], shifts_B[idx_b], sim, distance, is_valid))
            matched_B.add(idx_b)
        
        # 第2轮: 未匹配的B找A
        unmatched_B = set(range(len(shifts_B))) - matched_B
        for idx_b in unmatched_B:
            shift_b = shifts_B[idx_b]
            distances = [abs(shift_b - shift_a) for shift_a in shifts_A]
            idx_a = int(np.argmin(distances))
            distance = distances[idx_a]
            is_valid = distance <= self.tolerance
            sim = distance_penalty(distance, self.scale)
            
            matches.append((2, idx_a, idx_b, shifts_A[idx_a], shifts_B[idx_b], sim, distance, is_valid))
        
        return matches
    
    def _hungarian_match(self, shifts_A: List[float], shifts_B: List[float]) -> List[Tuple[int, int, float, bool]]:
        """
        匈牙利算法匹配策略
        
        第1轮: 少的一方与多的一方进行1v1最优匹配(匈牙利算法)
        第2轮: 多的一方未匹配的峰与少的一方进行贪心匹配
        """
        n_A = len(shifts_A)
        n_B = len(shifts_B)
        
        # 确定哪边少,哪边多
        if n_A <= n_B:
            # A少B多: A与B的子集进行1v1匹配
            shorter_shifts = shifts_A
            longer_shifts = shifts_B
            is_A_shorter = True
        else:
            # B少A多: B与A的子集进行1v1匹配
            shorter_shifts = shifts_B
            longer_shifts = shifts_A
            is_A_shorter = False
        
        n_short = len(shorter_shifts)
        n_long = len(longer_shifts)
        
        # 第1轮: 匈牙利算法 - 少的一方与多的一方进行1v1匹配
        # 构建距离矩阵 (n_short × n_long)
        similarity_matrix = np.zeros((n_short, n_long))
        cost_matrix = np.zeros((n_short, n_long))
        for i, shift_short in enumerate(shorter_shifts):
            for j, shift_long in enumerate(longer_shifts):
                cost_matrix[i, j] = abs(shift_short - shift_long) ** 2
                similarity_matrix[i, j] = distance_penalty(abs(shift_short - shift_long), self.scale)
        # 使用匈牙利算法求解最优分配
        if self.cost_sim:
            cost_matrix = 1 - similarity_matrix

        row_indices, col_indices = linear_sum_assignment(cost_matrix)
        
        matches = []
        matched_long = set()
        
        # 记录第1轮匹配
        for i, j in zip(row_indices, col_indices):
            sim = similarity_matrix[i, j]
            distance = abs(shorter_shifts[i] - longer_shifts[j])
            is_valid = distance <= self.tolerance
            if is_A_shorter:
                matches.append((1, i, j, shifts_A[i], shifts_B[j], sim, distance, is_valid))
                matched_long.add(j)
            else:
                matches.append((1, j, i, shifts_A[j], shifts_B[i], sim, distance, is_valid))
                matched_long.add(j)
        
        # 第2轮: 多的一方未匹配的峰找少的一方最近的峰(贪心)
        unmatched_long = set(range(n_long)) - matched_long
        
        for j in unmatched_long:
            shift_long = longer_shifts[j]
            sims = [distance_penalty(abs(shift_short - shift_long), self.scale) for shift_short in shorter_shifts]
            i = int(np.argmax(sims))
            sim = sims[i]
            distance = abs(shorter_shifts[i]-shift_long)
            is_valid = distance <= self.tolerance
            
            if is_A_shorter:
                matches.append((2, i, j, shifts_A[i], shifts_B[j], sim, distance, is_valid))
            else:
                matches.append((2, j, i, shifts_A[j], shifts_B[i], sim, distance, is_valid))
        
        return matches


class ExpandedPeakMatcher:
    """
    展开式峰匹配器 - 专门用于1H NMR
    将峰按照nH展开成多个"虚拟峰"进行匹配
    """
    
    def __init__(self, scale: float, tolerance: float, strategy: MatchStrategy = MatchStrategy.HUNGARIAN):
        """
        Args:
            tolerance: 容忍距离
            strategy: 匹配策略
        """
        # self.scale = scale
        # self.tolerance = tolerance
        # self.strategy = strategy
        self.base_matcher = PeakMatcher(scale, tolerance, strategy)
    
    def match_peaks_expanded(self, peaks_A: List[HNMRPeak], peaks_B: List[HNMRPeak]) -> Dict:
        """
        按nH展开后进行峰匹配
        
        策略:
        1. 将每个峰按其nH值展开成多个相同shift的"虚拟峰"
        2. 对展开后的峰列表进行匹配
        3. 将匹配结果映射回原始峰
        
        Args:
            peaks_A: 光谱A的峰列表
            peaks_B: 光谱B的峰列表
            
        Returns:
            包含匹配信息的字典
        """
        # 展开峰列表
        expanded_shifts_A, peak_map_A = self._expand_peaks(peaks_A)
        expanded_shifts_B, peak_map_B = self._expand_peaks(peaks_B)
        
        # 对展开后的峰进行匹配
        expanded_matches = self.base_matcher.match_peaks(expanded_shifts_A, expanded_shifts_B)
        
        # 统计原始峰的匹配次数和相似度
        peak_match_stats_A = {i: {'count': 0, 'similarities': []} for i in range(len(peaks_A))}
        peak_match_stats_B = {i: {'count': 0, 'similarities': []} for i in range(len(peaks_B))}
        for _round, exp_idx_a, exp_idx_b, shift_a, shift_b, sim, distance, is_valid in expanded_matches:
            orig_idx_a = peak_map_A[exp_idx_a]
            orig_idx_b = peak_map_B[exp_idx_b]
            
            peak_match_stats_A[orig_idx_a]['count'] += 1
            peak_match_stats_B[orig_idx_b]['count'] += 1
            
            if is_valid:
                # sim = distance_penalty(distance, self.scale)
                peak_match_stats_A[orig_idx_a]['similarities'].append(sim)
                peak_match_stats_B[orig_idx_b]['similarities'].append(sim)
        
        return {
            'expanded_matches': expanded_matches,
            'peak_map_A': peak_map_A,
            'peak_map_B': peak_map_B,
            'peak_match_stats_A': peak_match_stats_A,
            'peak_match_stats_B': peak_match_stats_B,
            'n_expanded_A': len(expanded_shifts_A),
            'n_expanded_B': len(expanded_shifts_B)
        }
    
    def _expand_peaks(self, peaks: List[HNMRPeak]) -> Tuple[List[float], List[int]]:
        """
        将峰列表按nH展开
        
        例: 一个3H的峰(shift=7.2)会被展开成3个shift=7.2的虚拟峰
        
        Args:
            peaks: 峰列表
            
        Returns:
            (展开后的shift列表, 映射关系: 展开峰索引->原始峰索引)
        """
        expanded_shifts = []
        peak_map = []  # 记录每个展开峰对应的原始峰索引
        
        for orig_idx, peak in enumerate(peaks):
            n_h = int(round(peak.n_hydrogen))  # 取整
            n_h = max(1, n_h)  # 至少1个氢
            
            for _ in range(n_h):
                expanded_shifts.append(peak.shift_center)
                peak_map.append(orig_idx)
        
        return expanded_shifts, peak_map


class CNMRSimilarityScorer:
    """13C NMR光谱相似度评估"""
    
    def __init__(self, scale: float = 5.0, tolerance: float =20.0, strategy: MatchStrategy = MatchStrategy.HUNGARIAN):
        """
        Args:
            scale: 13C NMR化学位移容忍距离(ppm),默认10.0
            strategy: 匹配策略
        """
        self.scale = scale
        self.tolerance = tolerance
        self.strategy = strategy
        self.matcher = PeakMatcher(scale, tolerance, strategy, cost_sim=True)

    def calculate_similarity_score(self, shifts_A: List[float], shifts_B: List[float]) -> Dict:
        """
        计算两个13C NMR光谱的相似度
        
        Args:
            shifts_A: 光谱A的化学位移列表
            shifts_B: 光谱B的化学位移列表
            
        Returns:
            包含相似度和详细信息的字典
        """
        if len(shifts_A) == 0 or len(shifts_B) == 0:
            return {
                'similarity': 0.0,
                'avg_peak_similarity': 0.0,
                'penalty_coefficient': 0.0,
                'n_matches': 0,
                'n_valid_matches': 0,
                'n_invalid_matches': 0,
                'n_delta_peak': abs(len(shifts_A) - len(shifts_B)),
                'strategy': self.strategy.value
            }
        
        # 峰匹配
        matches = self.matcher.match_peaks(shifts_A, shifts_B)
        
        # 计算匹配统计
        n_match = len(matches)
        n_valid = sum(1 for m in matches if m[7])
        n_invalid = n_match - n_valid
        n_delta_peak = abs(len(shifts_A) - len(shifts_B))
        n_match_min = max(len(shifts_A), len(shifts_B))
        
        # 计算平均峰相似度
        peak_similarities = []
        valid_matches = 0
        forced_matches = 0
        for _round, idx_a, idx_b, shift_a, shift_b, sim, distance, is_valid in matches:
            if is_valid:
                # sim = distance_penalty(distance, self.scale)
                if _round == 1:
                    valid_matches+=1
                    peak_similarities.append(sim)
                else:
                    peak_similarities.append(0.8*sim) 
            else:
                peak_similarities.append(0.0)

            if _round > 1:
                forced_matches+=1
        
        avg_peak_similarity = np.mean(peak_similarities) if peak_similarities else 0.0
        
        # 计算惩罚系数
        penalty_coefficient = n_match_min / (n_match + n_delta_peak + (n_match - valid_matches) + 1e-8)
        # penalty_coefficient = n_match_min / (n_match + n_delta_peak + n_invalid + 1e-8)
        # penalty_coefficient = n_match_min / (n_match + n_delta_peak + 0*n_invalid + 1e-8)
        # penalty_coefficient = (valid_matches/min(len(shifts_A), len(shifts_B))) * (1 - forced_matches/max(len(shifts_A), len(shifts_B)))
        
        # 最终相似度
        # final_similarity = avg_peak_similarity * penalty_coefficient
        final_similarity = 0.8*avg_peak_similarity + 0.2*penalty_coefficient
        
        return {
            'similarity': final_similarity,
            'avg_peak_similarity': avg_peak_similarity,
            'penalty_coefficient': penalty_coefficient,
            'n_matches': n_match,
            'n_valid_matches': n_valid,
            'n_invalid_matches': n_invalid,
            'n_delta_peak': n_delta_peak,
            'n_match_min': n_match_min,
            'matches': matches,
            'strategy': self.strategy.value
        }


class HNMRSimilarityScorer:
    """1H NMR光谱相似度评估"""
    
    def __init__(self, 
                 scale: float = 1.0, 
                 tolerance: float = 2.0, 
                 strategy: MatchStrategy = MatchStrategy.HUNGARIAN,
                 use_expanded_matching: bool = True):
        """
        Args:
            scale: 1H NMR化学位移容忍距离(ppm),默认2.0
            strategy: 匹配策略
            use_expanded_matching: 是否使用按nH展开的匹配
        """
        self.scale = scale
        self.tolerance = tolerance
        self.strategy = strategy
        self.use_expanded_matching = use_expanded_matching
        if use_expanded_matching:
            self.matcher = ExpandedPeakMatcher(scale, tolerance, strategy)
        else:
            self.matcher = PeakMatcher(scale, tolerance, strategy)


    def calculate_shift_min_similarity(self, peak1: HNMRPeak, peak2: HNMRPeak) -> float:
        """计算δ_min相似度 """
        diff = abs(peak1.shift_min - peak2.shift_min)
        return distance_penalty(diff, self.scale)
    
    def calculate_shift_max_similarity(self, peak1: HNMRPeak, peak2: HNMRPeak) -> float:
        """计算δ_max相似度 """
        diff = abs(peak1.shift_max - peak2.shift_max)
        return distance_penalty(diff, self.scale)
    
    def calculate_multiplicity_similarity(self, peak1: HNMRPeak, peak2: HNMRPeak) -> float:
        """计算多重性相似度 - 完全匹配才得分"""
        if peak1.multiplicity.lower() == peak2.multiplicity.lower():
            return 1
        else:
            return 0.0
    
    def calculate_integration_similarity(self, peak1: HNMRPeak, peak2: HNMRPeak) -> float:
        """计算积分相似度 - 完全匹配才得分"""
        if peak1.integration == peak2.integration:
            return 1
        else:
            return 0.0
    
    def calculate_coupling_similarity(self, list1: List[float], list2: List[float]) -> float:
        """
        计算两个相同长度耦合常数列表的相似度
        
        Args:
            list1, list2: 长度相同的耦合常数列表
            
        Returns:
            相似度得分 (0-1)
        """
        if len(list1) != len(list2):
            return 0.0
        
        if len(list1) == 0:
            return 1.0  # 两个都是空列表，完全相似
        
        # 对两个列表进行排序后比较
        sorted1 = sorted(list1)
        sorted2 = sorted(list2)
        
        # 计算对应位置的差异
        total_similarity = 0.0
        for j1, j2 in zip(sorted1, sorted2):
            # 耦合常数的容差可以稍大一些，使用0.5 Hz作为scale
            diff = abs(j1 - j2)
            similarity = distance_penalty(diff, 10)  # 0.5 Hz的scale参数
            total_similarity += similarity
        
        return total_similarity / len(list1)
    
    def calculate_coupling_constants_similarity(self, peak1: HNMRPeak, peak2: HNMRPeak) -> float:
        """计算耦合常数相似度 """
        # 首先检查长度是否匹配
        if len(peak1.coupling_constants) != len(peak2.coupling_constants):
            return 0.0
        
        # 计算相似度
        coupling_sim = self.calculate_coupling_similarity(
            peak1.coupling_constants, peak2.coupling_constants
        )
        
        return coupling_sim 
    
    def calculate_peak_similarity(self, peak1: HNMRPeak, peak2: HNMRPeak) -> Dict[str, float]:
        """计算两个峰的总相似度 (五元组各0.2分)"""
        similarities = {
            'shift_min': self.calculate_shift_min_similarity(peak1, peak2) * 0.4,
            'shift_max': self.calculate_shift_max_similarity(peak1, peak2) * 0.4,
            'multiplicity': self.calculate_multiplicity_similarity(peak1, peak2) * 0.05,
            'integration': self.calculate_integration_similarity(peak1, peak2) * 0.1,
            'coupling': self.calculate_coupling_constants_similarity(peak1, peak2) * 0.05
        }
        
        similarities['total'] = sum(similarities.values())
        return similarities
    

    def calculate_similarity_score(self, peaks_A: List[HNMRPeak], peaks_B: List[HNMRPeak]) -> Dict:
        """
        计算两个1H NMR光谱的相似度
        
        Args:
            peaks_A: 光谱A的峰列表
            peaks_B: 光谱B的峰列表
            
        Returns:
            包含相似度和详细信息的字典
        """
        if len(peaks_A) == 0 or len(peaks_B) == 0:
            return {
                'similarity': 0.0,
                'avg_peak_similarity': 0.0,
                'penalty_coefficient': 0.0,
                'nh_penalty': 0.0,
                'n_matches': 0,
                'n_valid_matches': 0,
                'n_invalid_matches': 0,
                'n_delta_peak': abs(len(peaks_A) - len(peaks_B)),
                'strategy': self.strategy.value,
                'use_expanded_matching': self.use_expanded_matching
            }
        
        if self.use_expanded_matching:
            return self._calculate_with_expansion(peaks_A, peaks_B)
        else:
            return self._calculate_without_expansion(peaks_A, peaks_B)
    
    def _calculate_without_expansion(self, peaks_A: List[HNMRPeak], peaks_B: List[HNMRPeak]) -> Dict:
        """不展开nH的计算方式(原始方式)"""
        shifts_A = [peak.shift_center for peak in peaks_A]
        shifts_B = [peak.shift_center for peak in peaks_B]
        
        matches = self.matcher.match_peaks(shifts_A, shifts_B)
        
        n_match = len(matches)
        n_valid = sum(1 for m in matches if m[7])
        n_invalid = n_match - n_valid
        n_delta_peak = abs(len(peaks_A) - len(peaks_B))
        n_match_min = max(len(peaks_A), len(peaks_B))
        
        peak_similarities = []
        valid_matches = 0
        forced_matches = 0
        for _round, idx_a, idx_b, shift_a, shift_b, sim, distance, is_valid in matches:
            if is_valid:
                # sim = distance_penalty(distance, self.scale)
                sim = self.calculate_peak_similarity(peaks_A[idx_a], peaks_B[idx_b])
                sim = sim['total']
                if _round == 1:
                    valid_matches+=1
                    peak_similarities.append(sim)
                else:
                    peak_similarities.append(0.8*sim)
            else:
                peak_similarities.append(0.0)

            if _round > 1:
                forced_matches+=1
        
        avg_peak_similarity = np.mean(peak_similarities) if peak_similarities else 0.0
        # penalty_coefficient = n_match_min / (n_match + n_delta_peak + n_invalid + 1e-8)
        penalty_coefficient = n_match_min / (n_match + n_delta_peak + (n_match - valid_matches) + 1e-8)
        
        n_nh_A = sum(peak.n_hydrogen for peak in peaks_A)
        n_nh_B = sum(peak.n_hydrogen for peak in peaks_B)
        nh_penalty = min(n_nh_A, n_nh_B) / (max(n_nh_A, n_nh_B) + 1e-8)
        
        final_similarity = ((0.8*avg_peak_similarity) + (0.2*penalty_coefficient)) * nh_penalty
        
        return {
            'similarity': final_similarity,
            'avg_peak_similarity': avg_peak_similarity,
            'penalty_coefficient': penalty_coefficient,
            'nh_penalty': nh_penalty,
            'n_matches': n_match,
            'n_valid_matches': n_valid,
            'n_invalid_matches': n_invalid,
            'n_delta_peak': n_delta_peak,
            'n_match_min': n_match_min,
            'n_nh_A': n_nh_A,
            'n_nh_B': n_nh_B,
            'matches': matches,
            'strategy': self.strategy.value,
            'use_expanded_matching': False
        }
    
    def _calculate_with_expansion(self, peaks_A: List[HNMRPeak], peaks_B: List[HNMRPeak]) -> Dict:
        """按nH展开的计算方式"""
        match_result = self.matcher.match_peaks_expanded(peaks_A, peaks_B)
        
        expanded_matches = match_result['expanded_matches']
        peak_map_A = match_result['peak_map_A']
        peak_map_B = match_result['peak_map_B']
        n_expanded_A = match_result['n_expanded_A']
        n_expanded_B = match_result['n_expanded_B']
        # 计算展开后的匹配统计
        n_match = len(expanded_matches)
        n_valid = sum(1 for m in expanded_matches if m[7])
        n_invalid = n_match - n_valid
        n_delta_peak = abs(n_expanded_A - n_expanded_B)
        n_match_min = max(n_expanded_A, n_expanded_B)
        
        # 计算平均峰相似度
        peak_similarities = []
        valid_matches = 0
        forced_matches = 0
        for _round, exp_idx_a, exp_idx_b, shift_a, shift_b, sim, distance, is_valid in expanded_matches:
            if is_valid:
                # sim = distance_penalty(distance, self.scale)
                # sim = calculate_peak_similarity(peaks_A[idx_a], peaks_B[idx_b])
                sim = self.calculate_peak_similarity(peaks_A[peak_map_A[exp_idx_a]], peaks_B[peak_map_B[exp_idx_b]])  
                sim = sim['total']
                if _round == 1:
                    valid_matches+=1
                    peak_similarities.append(sim)
                else:
                    peak_similarities.append(0.8*sim)
            else:
                peak_similarities.append(0.0)

            
            if _round > 1:
                forced_matches+=1
        
        avg_peak_similarity = np.mean(peak_similarities) if peak_similarities else 0.0
        
        # 计算惩罚系数
        # penalty_coefficient = n_match_min / (n_match + n_delta_peak + n_invalid + 1e-8)
        penalty_coefficient = n_match_min / (n_match + n_delta_peak + (n_match - valid_matches) + 1e-8)
        
        # 氢总数惩罚(基于展开后的数量,其实就是总氢数)
        nh_penalty = min(n_expanded_A, n_expanded_B) / (max(n_expanded_A, n_expanded_B) + 1e-8)
        
        # 最终相似度
        # final_similarity = avg_peak_similarity * penalty_coefficient * nh_penalty
        final_similarity = ((0.8*avg_peak_similarity) + (0.2*penalty_coefficient)) * nh_penalty
        
        # 统计原始峰的匹配信息
        n_nh_A = sum(peak.n_hydrogen for peak in peaks_A)
        n_nh_B = sum(peak.n_hydrogen for peak in peaks_B)
        
        return {
            'similarity': final_similarity,
            'avg_peak_similarity': avg_peak_similarity,
            'penalty_coefficient': penalty_coefficient,
            'nh_penalty': nh_penalty,
            'n_matches': n_match,
            'n_valid_matches': n_valid,
            'n_invalid_matches': n_invalid,
            'n_delta_peak': n_delta_peak,
            'n_match_min': n_match_min,
            'n_nh_A': n_nh_A,
            'n_nh_B': n_nh_B,
            'n_expanded_A': n_expanded_A,
            'n_expanded_B': n_expanded_B,
            'matches': expanded_matches,
            'peak_map_A': match_result['peak_map_A'],
            'peak_map_B': match_result['peak_map_B'],
            'peak_match_stats_A': match_result['peak_match_stats_A'],
            'peak_match_stats_B': match_result['peak_match_stats_B'],
            'strategy': self.strategy.value,
            'use_expanded_matching': True
        }
