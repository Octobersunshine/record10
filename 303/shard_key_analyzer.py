import pandas as pd
import numpy as np
import random
from datetime import datetime, timedelta
from collections import defaultdict, Counter
from itertools import combinations
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Any
import json


class AccessLogGenerator:
    def __init__(self, num_records: int = 10000, write_ratio: float = 0.4):
        self.num_records = num_records
        self.write_ratio = write_ratio
        self.user_ids = list(range(1, 1001))
        self.product_ids = list(range(1, 501))
        self.categories = ['Electronics', 'Clothing', 'Books', 'Home', 'Sports']
        self.regions = ['North', 'South', 'East', 'West', 'Central']

    def generate_access_logs(self) -> pd.DataFrame:
        np.random.seed(42)
        random.seed(42)
        
        write_timestamps = [datetime.now() - timedelta(hours=random.randint(0, 48), 
                                                       minutes=random.randint(0, 59)) 
                            for _ in range(int(self.num_records * self.write_ratio))]
        
        read_timestamps = [datetime.now() - timedelta(days=random.randint(0, 30), 
                                                      hours=random.randint(0, 23),
                                                      minutes=random.randint(0, 59)) 
                           for _ in range(int(self.num_records * (1 - self.write_ratio)))]
        
        timestamps = write_timestamps + read_timestamps
        
        user_weights = np.random.zipf(1.5, len(self.user_ids))
        user_weights = user_weights / user_weights.sum()
        
        write_user_weights = np.random.zipf(1.3, len(self.user_ids))
        write_user_weights = write_user_weights / write_user_weights.sum()
        
        total_writes = len(write_timestamps)
        total_reads = len(read_timestamps)
        
        user_ids = list(np.random.choice(self.user_ids, total_writes, p=write_user_weights)) + \
                   list(np.random.choice(self.user_ids, total_reads, p=user_weights))
        product_ids = list(np.random.choice(self.product_ids, self.num_records))
        categories = list(np.random.choice(self.categories, self.num_records))
        regions = list(np.random.choice(self.regions, self.num_records))
        
        operations = ['write'] * total_writes + ['read'] * total_reads
        
        query_types = list(np.random.choice(['point', 'range'], self.num_records, p=[0.7, 0.3]))
        query_fields = list(np.random.choice(['user_id', 'product_id', 'timestamp', 'category', 'region'], 
                                             self.num_records, 
                                             p=[0.35, 0.25, 0.2, 0.1, 0.1]))
        
        data = {
            'timestamp': timestamps,
            'user_id': user_ids,
            'product_id': product_ids,
            'category': categories,
            'region': regions,
            'operation': operations,
            'query_type': query_types,
            'query_field': query_fields
        }
        
        df = pd.DataFrame(data)
        df = df.sort_values('timestamp').reset_index(drop=True)
        return df


class AccessPatternAnalyzer:
    def __init__(self, access_logs: pd.DataFrame):
        self.logs = access_logs
        self.field_stats = {}
        
    def analyze_query_frequency(self) -> Dict:
        field_query_count = self.logs['query_field'].value_counts().to_dict()
        total_queries = len(self.logs)
        
        frequency_distribution = {
            field: {
                'count': count,
                'percentage': round(count / total_queries * 100, 2)
            }
            for field, count in field_query_count.items()
        }
        
        return {
            'total_queries': total_queries,
            'field_distribution': frequency_distribution
        }
    
    def analyze_query_types(self) -> Dict:
        query_type_stats = self.logs.groupby(['query_field', 'query_type']).size().unstack(fill_value=0)
        
        result = {}
        for field in query_type_stats.index:
            point_count = query_type_stats.loc[field, 'point'] if 'point' in query_type_stats.columns else 0
            range_count = query_type_stats.loc[field, 'range'] if 'range' in query_type_stats.columns else 0
            total = point_count + range_count
            
            result[field] = {
                'point_query_count': int(point_count),
                'range_query_count': int(range_count),
                'point_query_ratio': round(point_count / total * 100, 2) if total > 0 else 0,
                'range_query_ratio': round(range_count / total * 100, 2) if total > 0 else 0
            }
        
        return result
    
    def analyze_cardinality(self) -> Dict:
        cardinality_stats = {}
        fields = ['user_id', 'product_id', 'category', 'region']
        
        for field in fields:
            unique_count = self.logs[field].nunique()
            total_count = len(self.logs)
            value_counts = self.logs[field].value_counts()
            
            cardinality_stats[field] = {
                'unique_values': int(unique_count),
                'total_records': int(total_count),
                'cardinality_ratio': round(unique_count / total_count * 100, 4),
                'max_frequency': int(value_counts.max()),
                'min_frequency': int(value_counts.min()),
                'std_frequency': round(value_counts.std(), 2),
                'cv_frequency': round(value_counts.std() / value_counts.mean(), 4)
            }
        
        cardinality_stats['timestamp'] = {
            'unique_values': int(self.logs['timestamp'].nunique()),
            'total_records': int(len(self.logs)),
            'cardinality_ratio': round(self.logs['timestamp'].nunique() / len(self.logs) * 100, 4),
            'time_span_days': round((self.logs['timestamp'].max() - self.logs['timestamp'].min()).total_seconds() / 86400, 2)
        }
        
        return cardinality_stats
    
    def analyze_access_distribution(self, field: str) -> Dict:
        if field not in self.logs.columns:
            return {}
        
        value_counts = self.logs[field].value_counts()
        total = value_counts.sum()
        
        distribution = {
            'top_10_percent_coverage': round(value_counts.head(int(len(value_counts) * 0.1)).sum() / total * 100, 2),
            'top_20_percent_coverage': round(value_counts.head(int(len(value_counts) * 0.2)).sum() / total * 100, 2),
            'gini_coefficient': self._calculate_gini(value_counts.values),
            'entropy': self._calculate_entropy(value_counts.values),
            'distribution': value_counts.to_dict()
        }
        
        return distribution
    
    def _calculate_gini(self, values: np.ndarray) -> float:
        values = np.sort(values)
        n = len(values)
        index = np.arange(1, n + 1)
        return round((2 * np.sum(index * values) / (n * np.sum(values)) - (n + 1) / n), 4)
    
    def _calculate_entropy(self, values: np.ndarray) -> float:
        probs = values / values.sum()
        return round(-np.sum(probs * np.log2(probs + 1e-10)), 4)
    
    def get_temporal_patterns(self) -> Dict:
        self.logs['hour'] = self.logs['timestamp'].dt.hour
        self.logs['day_of_week'] = self.logs['timestamp'].dt.dayofweek
        
        hourly_dist = self.logs['hour'].value_counts().sort_index().to_dict()
        daily_dist = self.logs['day_of_week'].value_counts().sort_index().to_dict()
        
        return {
            'hourly_distribution': {str(k): int(v) for k, v in hourly_dist.items()},
            'daily_distribution': {str(k): int(v) for k, v in daily_dist.items()}
        }
    
    def analyze_operation_distribution(self) -> Dict:
        op_counts = self.logs['operation'].value_counts().to_dict()
        total = len(self.logs)
        
        return {
            'total_operations': total,
            'write_count': int(op_counts.get('write', 0)),
            'read_count': int(op_counts.get('read', 0)),
            'write_ratio': round(op_counts.get('write', 0) / total * 100, 2),
            'read_ratio': round(op_counts.get('read', 0) / total * 100, 2)
        }
    
    def analyze_write_distribution(self, num_shards: int = 4) -> Dict:
        write_logs = self.logs[self.logs['operation'] == 'write'].copy()
        total_writes = len(write_logs)
        
        if total_writes == 0:
            return {}
        
        fields = ['user_id', 'product_id', 'timestamp', 'category', 'region']
        write_distributions = {}
        
        for field in fields:
            if field not in write_logs.columns:
                continue
            
            value_counts = write_logs[field].value_counts()
            
            if field == 'timestamp':
                shard_assignments = defaultdict(int)
                sorted_writes = write_logs.sort_values('timestamp')
                shard_size = total_writes // num_shards
                
                for i in range(num_shards):
                    start_idx = i * shard_size
                    end_idx = start_idx + shard_size if i < num_shards - 1 else total_writes
                    shard_assignments[i] = end_idx - start_idx
                
                hotspot_factor = self._calculate_timestamp_hotspot_factor(write_logs, num_shards)
            else:
                shard_assignments = defaultdict(int)
                for value, count in value_counts.items():
                    shard_id = hash(str(value)) % num_shards
                    shard_assignments[shard_id] += count
                
                hotspot_factor = 1.0
            
            counts = np.array([shard_assignments.get(i, 0) for i in range(num_shards)])
            mean_count = counts.mean()
            variance = counts.var()
            std_dev = counts.std()
            cv = std_dev / mean_count if mean_count > 0 else 0
            max_shard = counts.max()
            min_shard = counts.min()
            max_min_ratio = max_shard / min_shard if min_shard > 0 else float('inf')
            
            write_distributions[field] = {
                'total_writes': int(total_writes),
                'unique_values': int(value_counts.nunique()),
                'shard_counts': {f'shard_{i}': int(counts[i]) for i in range(num_shards)},
                'shard_percentages': {f'shard_{i}': round(counts[i] / total_writes * 100, 2) 
                                     for i in range(num_shards)},
                'mean': round(mean_count, 2),
                'variance': round(variance, 2),
                'std_dev': round(std_dev, 2),
                'coefficient_of_variation': round(cv, 4),
                'max_shard_writes': int(max_shard),
                'min_shard_writes': int(min_shard),
                'max_min_ratio': round(max_min_ratio, 2),
                'hotspot_factor': round(hotspot_factor, 4),
                'write_balance_score': round(max(0, 1 - min(cv * hotspot_factor, 1)) * 100, 2)
            }
        
        return write_distributions
    
    def _calculate_timestamp_hotspot_factor(self, write_logs: pd.DataFrame, num_shards: int) -> float:
        now = datetime.now()
        write_logs = write_logs.copy()
        write_logs['hours_ago'] = (now - write_logs['timestamp']).dt.total_seconds() / 3600
        
        write_logs = write_logs.sort_values('timestamp')
        total_writes = len(write_logs)
        latest_shard_size = total_writes // num_shards
        latest_writes = write_logs.tail(latest_shard_size)
        
        latest_time_span = (latest_writes['timestamp'].max() - latest_writes['timestamp'].min()).total_seconds() / 3600
        total_time_span = (write_logs['timestamp'].max() - write_logs['timestamp'].min()).total_seconds() / 3600
        
        if total_time_span > 0:
            time_concentration = 1 - (latest_time_span / total_time_span)
        else:
            time_concentration = 1.0
        
        recent_threshold = 24
        recent_writes = write_logs[write_logs['hours_ago'] <= recent_threshold]
        recent_ratio = len(recent_writes) / len(write_logs) if len(write_logs) > 0 else 0
        
        hotspot_factor = 1 + time_concentration * 1.5 + recent_ratio * 0.5
        
        return min(hotspot_factor, 3.0)


class ShardKeyRecommender:
    def __init__(self, analyzer: AccessPatternAnalyzer, num_shards: int = 4):
        self.analyzer = analyzer
        self.num_shards = num_shards
        self.candidate_fields = ['user_id', 'product_id', 'timestamp', 'category', 'region']
        
    def calculate_shard_scores(self) -> Dict[str, Dict]:
        query_freq = self.analyzer.analyze_query_frequency()
        query_types = self.analyzer.analyze_query_types()
        cardinality = self.analyzer.analyze_cardinality()
        write_dist = self.analyzer.analyze_write_distribution(self.num_shards)
        op_dist = self.analyzer.analyze_operation_distribution()
        
        scores = {}
        
        for field in self.candidate_fields:
            if field not in cardinality:
                continue
                
            field_stats = cardinality[field]
            freq_stats = query_freq['field_distribution'].get(field, {'count': 0, 'percentage': 0})
            type_stats = query_types.get(field, {'point_query_ratio': 0, 'range_query_ratio': 0})
            write_stats = write_dist.get(field, {})
            
            access_score = freq_stats['percentage'] / 100
            
            if field == 'timestamp':
                balance_score = 0.8
            else:
                cv = field_stats.get('cv_frequency', 1)
                balance_score = max(0, 1 - min(cv, 1))
            
            cardinality_ratio = field_stats['cardinality_ratio']
            if cardinality_ratio < 0.1:
                cardinality_score = 0.2
            elif cardinality_ratio < 1:
                cardinality_score = 0.5
            elif cardinality_ratio < 10:
                cardinality_score = 0.8
            else:
                cardinality_score = 1.0
            
            point_query_ratio = type_stats['point_query_ratio'] / 100
            query_type_score = point_query_ratio * 0.8 + (1 - point_query_ratio) * 0.6
            
            if field == 'timestamp':
                query_type_score = type_stats['range_query_ratio'] / 100 * 0.9 + \
                                   type_stats['point_query_ratio'] / 100 * 0.3
            
            write_balance_score = write_stats.get('write_balance_score', 50.0) / 100
            write_ratio = op_dist['write_ratio'] / 100
            
            hotspot_penalty = 0.0
            hotspot_warning = ""
            if field == 'timestamp':
                hotspot_factor = write_stats.get('hotspot_factor', 1.0)
                if hotspot_factor > 1.2:
                    hotspot_penalty = min(0.5, (hotspot_factor - 1.2) * 0.4)
                    hotspot_warning = "写入热点检测：时间戳分片会导致最新分片写入压力集中"
            
            adjusted_write_score = max(0, write_balance_score - hotspot_penalty)
            
            weights = {
                'access': 0.20,
                'balance': 0.15,
                'cardinality': 0.15,
                'query_type': 0.10,
                'write_balance': 0.40
            }
            
            total_score = (
                access_score * weights['access'] +
                balance_score * weights['balance'] +
                cardinality_score * weights['cardinality'] +
                query_type_score * weights['query_type'] +
                adjusted_write_score * weights['write_balance']
            )
            
            scores[field] = {
                'total_score': round(total_score * 100, 2),
                'component_scores': {
                    'access_score': round(access_score * 100, 2),
                    'balance_score': round(balance_score * 100, 2),
                    'cardinality_score': round(cardinality_score * 100, 2),
                    'query_type_score': round(query_type_score * 100, 2),
                    'write_balance_score': round(adjusted_write_score * 100, 2),
                    'raw_write_balance_score': write_stats.get('write_balance_score', 0.0),
                    'hotspot_penalty': round(hotspot_penalty * 100, 2)
                },
                'metrics': {
                    'query_percentage': freq_stats['percentage'],
                    'unique_values': field_stats['unique_values'],
                    'cardinality_ratio': field_stats['cardinality_ratio'],
                    'point_query_ratio': type_stats['point_query_ratio'],
                    'range_query_ratio': type_stats['range_query_ratio'],
                    'write_variance': write_stats.get('variance', 0),
                    'write_cv': write_stats.get('coefficient_of_variation', 0),
                    'write_hotspot_factor': write_stats.get('hotspot_factor', 1.0),
                    'write_ratio': op_dist['write_ratio'],
                    'hotspot_warning': hotspot_warning
                },
                'write_distribution': write_stats.get('shard_percentages', {})
            }
        
        return scores
    
    def evaluate_composite_keys(self, num_shards: int = 4) -> Dict:
        logs = self.analyzer.logs
        query_freq = self.analyzer.analyze_query_frequency()
        query_types = self.analyzer.analyze_query_types()
        write_dist = self.analyzer.analyze_write_distribution(num_shards)
        op_dist = self.analyzer.analyze_operation_distribution()
        cardinality = self.analyzer.analyze_cardinality()
        
        single_fields = ['user_id', 'product_id', 'region', 'category']
        composite_candidates = list(combinations(single_fields, 2))
        
        composite_results = {}
        
        for combo in composite_candidates:
            combo_name = '+'.join(combo)
            
            combined_values = logs[list(combo)].astype(str).agg('|||'.join, axis=1)
            unique_combined = combined_values.nunique()
            total_records = len(logs)
            combined_cardinality_ratio = unique_combined / total_records * 100
            
            value_counts = combined_values.value_counts()
            shard_assignments = defaultdict(int)
            for value, count in value_counts.items():
                shard_id = hash(value) % num_shards
                shard_assignments[shard_id] += count
            
            counts = np.array([shard_assignments.get(i, 0) for i in range(num_shards)])
            cv = counts.std() / counts.mean() if counts.mean() > 0 else 0
            balance_score = max(0, 1 - min(cv, 1))
            
            write_logs = logs[logs['operation'] == 'write']
            write_combined = write_logs[list(combo)].astype(str).agg('|||'.join, axis=1)
            write_value_counts = write_combined.value_counts()
            write_shard_assignments = defaultdict(int)
            for value, count in write_value_counts.items():
                shard_id = hash(value) % num_shards
                write_shard_assignments[shard_id] += count
            
            write_counts = np.array([write_shard_assignments.get(i, 0) for i in range(num_shards)])
            write_cv = write_counts.std() / write_counts.mean() if write_counts.mean() > 0 else 0
            write_balance = max(0, 1 - min(write_cv, 1))
            
            if combined_cardinality_ratio < 1:
                cardinality_score = 0.5
            elif combined_cardinality_ratio < 10:
                cardinality_score = 0.8
            else:
                cardinality_score = 1.0
            
            access_score = sum(query_freq['field_distribution'].get(f, {}).get('percentage', 0) 
                             for f in combo) / 100
            
            point_ratios = []
            range_ratios = []
            for f in combo:
                if f in query_types:
                    point_ratios.append(query_types[f]['point_query_ratio'] / 100)
                    range_ratios.append(query_types[f]['range_query_ratio'] / 100)
            
            avg_point = np.mean(point_ratios) if point_ratios else 0.7
            query_type_score = avg_point * 0.8 + (1 - avg_point) * 0.6
            
            weights = {
                'access': 0.20,
                'balance': 0.15,
                'cardinality': 0.15,
                'query_type': 0.10,
                'write_balance': 0.40
            }
            
            total_score = (
                access_score * weights['access'] +
                balance_score * weights['balance'] +
                cardinality_score * weights['cardinality'] +
                query_type_score * weights['query_type'] +
                write_balance * weights['write_balance']
            )
            
            coverage_ratios = {}
            for f in combo:
                shard_map = {}
                for value, count in logs[f].value_counts().items():
                    shard_map[value] = hash(str(value)) % num_shards
                coverage_ratios[f] = len(set(shard_map.values())) / num_shards
            
            composite_results[combo_name] = {
                'fields': list(combo),
                'total_score': round(total_score * 100, 2),
                'unique_combinations': int(unique_combined),
                'combined_cardinality_ratio': round(combined_cardinality_ratio, 4),
                'balance_cv': round(cv, 4),
                'write_balance_cv': round(write_cv, 4),
                'write_balance_score': round(write_balance * 100, 2),
                'shard_distribution': {f'shard_{i}': int(counts[i]) for i in range(num_shards)},
                'shard_percentages': {f'shard_{i}': round(counts[i] / total_records * 100, 2) 
                                     for i in range(num_shards)},
                'field_coverage': coverage_ratios,
                'hotspot_penalty': 0.0
            }
        
        return composite_results
    
    def calculate_optimal_shard_count(self, 
                                       total_records: int,
                                       shard_capacity_limit: int = 500000,
                                       daily_growth_rate: float = 0.05,
                                       growth_period_months: int = 12,
                                       target_fill_ratio: float = 0.7) -> Dict:
        current_records = total_records
        days_in_period = growth_period_months * 30
        
        projected_records = current_records
        for day in range(days_in_period):
            projected_records *= (1 + daily_growth_rate / 30)
        
        min_shards_by_capacity = int(np.ceil(projected_records / (shard_capacity_limit * target_fill_ratio)))
        
        min_shards_current = int(np.ceil(current_records / (shard_capacity_limit * target_fill_ratio)))
        
        shard_counts_to_evaluate = list(range(max(2, min_shards_current - 1), 
                                               min_shards_by_capacity + 4))
        
        balance_evaluations = {}
        for n in shard_counts_to_evaluate:
            write_dist = self.analyzer.analyze_write_distribution(n)
            field_scores = {}
            for field in self.candidate_fields:
                if field in write_dist:
                    field_scores[field] = write_dist[field]['write_balance_score']
            best_field = max(field_scores, key=field_scores.get) if field_scores else 'unknown'
            best_balance = field_scores.get(best_field, 0)
            
            avg_balance = np.mean(list(field_scores.values())) if field_scores else 0
            
            balance_evaluations[n] = {
                'best_field': best_field,
                'best_balance_score': round(best_balance, 2),
                'avg_balance_score': round(avg_balance, 2),
                'per_shard_records': round(projected_records / n, 0),
                'fill_ratio': round(projected_records / (n * shard_capacity_limit), 4),
                'field_scores': field_scores
            }
        
        optimal_shards = min_shards_by_capacity
        best_eval = balance_evaluations.get(optimal_shards, {})
        
        for n in range(min_shards_by_capacity, min_shards_by_capacity + 4):
            if n in balance_evaluations:
                eval_n = balance_evaluations[n]
                if eval_n['fill_ratio'] <= 0.85 and eval_n['avg_balance_score'] > 50:
                    optimal_shards = n
                    best_eval = eval_n
                    break
        
        return {
            'current_records': int(current_records),
            'projected_records': int(round(projected_records, 0)),
            'daily_growth_rate': round(daily_growth_rate * 100, 2),
            'growth_period_months': growth_period_months,
            'shard_capacity_limit': shard_capacity_limit,
            'target_fill_ratio': target_fill_ratio,
            'min_shards_by_capacity': min_shards_by_capacity,
            'recommended_shard_count': optimal_shards,
            'recommended_per_shard_records': int(round(projected_records / optimal_shards, 0)),
            'projected_fill_ratio': round(projected_records / (optimal_shards * shard_capacity_limit), 4),
            'balance_evaluation': best_eval,
            'all_evaluations': {str(k): v for k, v in balance_evaluations.items()}
        }
    
    def simulate_query_routing(self, shard_key: str, num_shards: int, 
                                is_composite: bool = False) -> Dict:
        logs = self.analyzer.logs
        
        if is_composite:
            fields = shard_key.split('+')
            combined_values = logs[fields].astype(str).agg('|||'.join, axis=1)
            value_to_shard = {}
            for value in combined_values.unique():
                value_to_shard[value] = hash(value) % num_shards
        else:
            value_to_shard = {}
            for value in logs[shard_key].unique():
                value_to_shard[str(value)] = hash(str(value)) % num_shards
        
        point_queries = logs[logs['query_type'] == 'point']
        range_queries = logs[logs['query_type'] == 'range']
        
        point_cross_shard = 0
        point_total = len(point_queries)
        
        for _, row in point_queries.iterrows():
            query_field = row['query_field']
            if query_field == shard_key or (is_composite and query_field in shard_key.split('+')):
                point_cross_shard += 0
            else:
                point_cross_shard += min(num_shards - 1, 1)
        
        range_cross_shard = 0
        range_total = len(range_queries)
        
        for _, row in range_queries.iterrows():
            query_field = row['query_field']
            if query_field == shard_key or (is_composite and query_field in shard_key.split('+')):
                if is_composite:
                    range_cross_shard += num_shards * 0.5
                elif shard_key == 'timestamp':
                    range_cross_shard += max(1, num_shards * (30 / 100))
                else:
                    field_values = logs[query_field].nunique() if query_field in logs.columns else num_shards
                    range_cross_shard += max(1, num_shards * 0.4)
            else:
                range_cross_shard += num_shards
        
        total_queries = point_total + range_total
        total_cross_shard = point_cross_shard + range_cross_shard
        
        avg_cross_shard_ratio = total_cross_shard / total_queries if total_queries > 0 else 0
        avg_cross_shard_ratio = min(avg_cross_shard_ratio / num_shards, 1.0)
        
        point_cross_ratio = point_cross_shard / point_total / num_shards if point_total > 0 else 0
        range_cross_ratio = range_cross_shard / range_total / num_shards if range_total > 0 else 0
        
        routing_efficiency = (1 - avg_cross_shard_ratio) * 100
        
        targeted_queries = logs[logs['query_field'] == shard_key].shape[0]
        if is_composite:
            targeted_queries = sum(logs['query_field'] == f for f in shard_key.split('+'))
        targeted_ratio = targeted_queries / total_queries * 100 if total_queries > 0 else 0
        
        return {
            'shard_key': shard_key,
            'num_shards': num_shards,
            'total_queries': total_queries,
            'point_queries': point_total,
            'range_queries': range_total,
            'avg_cross_shard_ratio': round(avg_cross_shard_ratio * 100, 2),
            'point_cross_shard_ratio': round(min(point_cross_ratio, 1.0) * 100, 2),
            'range_cross_shard_ratio': round(min(range_cross_ratio, 1.0) * 100, 2),
            'routing_efficiency': round(routing_efficiency, 2),
            'targeted_query_ratio': round(targeted_ratio, 2),
            'estimated_latency_factor': round(1 + avg_cross_shard_ratio * 0.5, 2)
        }
    
    def recommend_shard_key(self, num_shards: int = 4, 
                             shard_capacity_limit: int = 500000,
                             daily_growth_rate: float = 0.05,
                             growth_period_months: int = 12) -> Dict:
        self.num_shards = num_shards
        scores = self.calculate_shard_scores()
        
        sorted_fields = sorted(scores.items(), key=lambda x: x[1]['total_score'], reverse=True)
        best_field = sorted_fields[0][0]
        
        composite_results = self.evaluate_composite_keys(num_shards)
        sorted_composites = sorted(composite_results.items(), 
                                    key=lambda x: x[1]['total_score'], reverse=True)
        best_composite = sorted_composites[0] if sorted_composites else None
        
        best_overall = best_field
        best_overall_score = sorted_fields[0][1]['total_score']
        is_composite = False
        
        if best_composite and best_composite[1]['total_score'] > best_overall_score * 1.05:
            best_overall = best_composite[0]
            best_overall_score = best_composite[1]['total_score']
            is_composite = True
        
        expected_distribution = self._calculate_expected_distribution(best_field, num_shards)
        write_distribution = self.analyzer.analyze_write_distribution(num_shards)
        
        shard_count_calc = self.calculate_optimal_shard_count(
            total_records=len(self.analyzer.logs),
            shard_capacity_limit=shard_capacity_limit,
            daily_growth_rate=daily_growth_rate,
            growth_period_months=growth_period_months
        )
        recommended_shard_count = shard_count_calc['recommended_shard_count']
        
        if recommended_shard_count != num_shards:
            self.num_shards = recommended_shard_count
            scores = self.calculate_shard_scores()
            sorted_fields = sorted(scores.items(), key=lambda x: x[1]['total_score'], reverse=True)
            best_field = sorted_fields[0][0]
            expected_distribution = self._calculate_expected_distribution(best_field, recommended_shard_count)
            write_distribution = self.analyzer.analyze_write_distribution(recommended_shard_count)
            num_shards = recommended_shard_count
        
        single_routing = self.simulate_query_routing(best_field, num_shards)
        
        all_routing = {}
        for field in self.candidate_fields:
            if field in scores:
                all_routing[field] = self.simulate_query_routing(field, num_shards)
        
        for combo_name, combo_data in composite_results.items():
            all_routing[combo_name] = self.simulate_query_routing(combo_name, num_shards, is_composite=True)
        
        return {
            'recommended_shard_key': best_overall,
            'is_composite_key': is_composite,
            'confidence_score': best_overall_score,
            'ranking': [
                {'field': field, 'score': details['total_score']}
                for field, details in sorted_fields
            ],
            'composite_ranking': [
                {'key': name, 'score': data['total_score']}
                for name, data in sorted_composites
            ],
            'num_shards': num_shards,
            'expected_distribution': expected_distribution,
            'write_distribution_analysis': write_distribution,
            'operation_distribution': self.analyzer.analyze_operation_distribution(),
            'detailed_scores': scores,
            'composite_key_analysis': composite_results,
            'shard_count_calculation': shard_count_calc,
            'query_routing_efficiency': single_routing,
            'all_routing_efficiency': all_routing
        }
    
    def _calculate_expected_distribution(self, shard_key: str, num_shards: int) -> Dict:
        if shard_key == 'timestamp':
            return self._calculate_time_distribution(num_shards)
        
        values = self.analyzer.logs[shard_key].value_counts()
        total = values.sum()
        
        shard_assignments = defaultdict(int)
        for i, (value, count) in enumerate(values.items()):
            shard_id = hash(str(value)) % num_shards
            shard_assignments[shard_id] += count
        
        distribution = {}
        for shard_id in range(num_shards):
            count = shard_assignments.get(shard_id, 0)
            distribution[f'shard_{shard_id}'] = {
                'record_count': int(count),
                'percentage': round(count / total * 100, 2)
            }
        
        counts = list(shard_assignments.values())
        cv = np.std(counts) / np.mean(counts) if counts else 0
        
        return {
            'shard_distribution': distribution,
            'balance_quality': 'excellent' if cv < 0.1 else 'good' if cv < 0.2 else 'fair' if cv < 0.3 else 'poor',
            'coefficient_of_variation': round(cv, 4)
        }
    
    def _calculate_time_distribution(self, num_shards: int) -> Dict:
        logs = self.analyzer.logs.sort_values('timestamp')
        total = len(logs)
        
        shard_size = total // num_shards
        distribution = {}
        
        for i in range(num_shards):
            start_idx = i * shard_size
            end_idx = start_idx + shard_size if i < num_shards - 1 else total
            count = end_idx - start_idx
            
            start_time = logs.iloc[start_idx]['timestamp']
            end_time = logs.iloc[min(end_idx - 1, total - 1)]['timestamp']
            
            distribution[f'shard_{i}'] = {
                'record_count': int(count),
                'percentage': round(count / total * 100, 2),
                'time_range': f"{start_time.strftime('%Y-%m-%d')} to {end_time.strftime('%Y-%m-%d')}"
            }
        
        return {
            'shard_distribution': distribution,
            'balance_quality': 'excellent',
            'coefficient_of_variation': 0.0,
            'note': 'Time-based sharding provides sequential data partitioning'
        }


class ReportGenerator:
    def __init__(self, recommendation: Dict, analyzer: AccessPatternAnalyzer):
        self.recommendation = recommendation
        self.analyzer = analyzer
    
    def generate_text_report(self) -> str:
        report = []
        report.append("=" * 80)
        report.append("数据集访问模式分析与分片键推荐报告")
        report.append("=" * 80)
        report.append("")
        
        op_dist = self.recommendation.get('operation_distribution', {})
        report.append("零、读写操作分布")
        report.append("-" * 40)
        report.append(f"总操作次数: {op_dist.get('total_operations', 0)}")
        report.append(f"写入操作: {op_dist.get('write_count', 0)} 次 ({op_dist.get('write_ratio', 0)}%)")
        report.append(f"读取操作: {op_dist.get('read_count', 0)} 次 ({op_dist.get('read_ratio', 0)}%)")
        report.append("")
        
        report.append("一、查询频率分布")
        report.append("-" * 40)
        query_freq = self.analyzer.analyze_query_frequency()
        report.append(f"总查询次数: {query_freq['total_queries']}")
        report.append("各字段查询分布:")
        for field, stats in sorted(query_freq['field_distribution'].items(), 
                                   key=lambda x: x[1]['count'], reverse=True):
            report.append(f"  {field:15s}: {stats['count']:6d} 次 ({stats['percentage']}%)")
        report.append("")
        
        report.append("二、查询类型比例（等值查询 vs 范围查询）")
        report.append("-" * 40)
        query_types = self.analyzer.analyze_query_types()
        for field in query_types:
            stats = query_types[field]
            report.append(f"  {field:15s}: 等值查询 {stats['point_query_ratio']:6.2f}% | "
                         f"范围查询 {stats['range_query_ratio']:6.2f}%")
        report.append("")
        
        report.append("三、基数均衡性分析")
        report.append("-" * 40)
        cardinality = self.analyzer.analyze_cardinality()
        for field in ['user_id', 'product_id', 'category', 'region', 'timestamp']:
            if field in cardinality:
                stats = cardinality[field]
                report.append(f"  {field:15s}: 唯一值 {stats['unique_values']:5d} | "
                             f"基数比 {stats['cardinality_ratio']:8.4f}%")
        report.append("")
        
        report.append("四、写入均衡性分析")
        report.append("-" * 40)
        write_dist = self.recommendation.get('write_distribution_analysis', {})
        report.append(f"分片数: {self.recommendation['num_shards']}")
        report.append("各候选字段写入分布评估:")
        for field in ['user_id', 'product_id', 'timestamp', 'category', 'region']:
            if field in write_dist:
                stats = write_dist[field]
                hotspot_note = ""
                if stats.get('hotspot_factor', 1.0) > 1.5 and field == 'timestamp':
                    hotspot_note = " ⚠️ 写入热点风险"
                report.append(f"  {field:15s}: 写入均衡分 {stats['write_balance_score']:5.2f} | "
                             f"CV {stats['coefficient_of_variation']:6.4f} | "
                             f"热点因子 {stats['hotspot_factor']:.2f}{hotspot_note}")
        report.append("")
        
        report.append("五、复合分片键评估")
        report.append("-" * 40)
        composite_analysis = self.recommendation.get('composite_key_analysis', {})
        if composite_analysis:
            sorted_composites = sorted(composite_analysis.items(), 
                                       key=lambda x: x[1]['total_score'], reverse=True)
            for rank, (name, data) in enumerate(sorted_composites, 1):
                marker = " ★ 推荐" if rank == 1 else ""
                report.append(f"  {rank}. {name:25s} - 综合: {data['total_score']:5.2f} | "
                             f"组合基数: {data['unique_combinations']:6d} | "
                             f"写入均衡: {data['write_balance_score']:5.2f} | "
                             f"CV: {data['balance_cv']:.4f}{marker}")
        else:
            report.append("  无复合键候选")
        report.append("")
        
        report.append("六、分片数量计算")
        report.append("-" * 40)
        shard_calc = self.recommendation.get('shard_count_calculation', {})
        if shard_calc:
            report.append(f"当前数据量: {shard_calc['current_records']:,} 条")
            report.append(f"日均增速: {shard_calc['daily_growth_rate']}%")
            report.append(f"预测周期: {shard_calc['growth_period_months']} 个月")
            report.append(f"预测数据量: {shard_calc['projected_records']:,} 条")
            report.append(f"单分片容量上限: {shard_calc['shard_capacity_limit']:,} 条")
            report.append(f"目标填充率: {shard_calc['target_fill_ratio'] * 100:.0f}%")
            report.append(f"容量需求最小分片数: {shard_calc['min_shards_by_capacity']}")
            report.append(f"推荐分片数: {shard_calc['recommended_shard_count']}")
            report.append(f"每分片预计记录数: {shard_calc['recommended_per_shard_records']:,}")
            report.append(f"预计填充率: {shard_calc['projected_fill_ratio'] * 100:.2f}%")
        report.append("")
        
        report.append("七、查询路由效率模拟")
        report.append("-" * 40)
        routing = self.recommendation.get('query_routing_efficiency', {})
        if routing:
            report.append(f"分片键: {routing['shard_key']}")
            report.append(f"分片数: {routing['num_shards']}")
            report.append(f"路由效率: {routing['routing_efficiency']:.2f}%")
            report.append(f"平均跨分片查询比例: {routing['avg_cross_shard_ratio']:.2f}%")
            report.append(f"  等值查询跨分片率: {routing['point_cross_shard_ratio']:.2f}%")
            report.append(f"  范围查询跨分片率: {routing['range_cross_shard_ratio']:.2f}%")
            report.append(f"可定向查询比例: {routing['targeted_query_ratio']:.2f}%")
            report.append(f"预估延迟因子: {routing['estimated_latency_factor']:.2f}x")
        report.append("")
        
        all_routing = self.recommendation.get('all_routing_efficiency', {})
        if all_routing:
            report.append("各候选键路由效率对比:")
            sorted_routing = sorted(all_routing.items(), 
                                    key=lambda x: x[1]['routing_efficiency'], reverse=True)
            for name, r in sorted_routing:
                report.append(f"  {name:25s}: 路由效率 {r['routing_efficiency']:5.2f}% | "
                             f"跨分片 {r['avg_cross_shard_ratio']:5.2f}% | "
                             f"延迟 {r['estimated_latency_factor']:.2f}x")
        report.append("")
        
        report.append("八、最终分片键推荐")
        report.append("-" * 40)
        is_composite = self.recommendation.get('is_composite_key', False)
        key_type = "复合分片键" if is_composite else "单字段分片键"
        report.append(f"推荐分片键: {self.recommendation['recommended_shard_key']} ({key_type})")
        report.append(f"推荐置信度: {self.recommendation['confidence_score']} / 100")
        report.append(f"推荐分片数: {self.recommendation['num_shards']}")
        report.append("")
        
        report.append("单字段评分排名 (含写入均衡权重40%):")
        for rank, item in enumerate(self.recommendation['ranking'], 1):
            field = item['field']
            details = self.recommendation['detailed_scores'][field]
            comp = details['component_scores']
            penalty_note = ""
            if comp.get('hotspot_penalty', 0) > 0:
                penalty_note = f" (热点惩罚 -{comp['hotspot_penalty']})"
            report.append(f"  {rank}. {field:15s} - 综合: {item['score']:5.2f} | "
                         f"写入: {comp['write_balance_score']:5.2f} | "
                         f"访问: {comp['access_score']:5.2f} | "
                         f"基数: {comp['cardinality_score']:5.2f}{penalty_note}")
        report.append("")
        
        report.append(f"九、预期分布（分 {self.recommendation['num_shards']} 片）")
        report.append("-" * 40)
        dist = self.recommendation['expected_distribution']
        report.append(f"均衡性质量: {dist['balance_quality'].upper()}")
        report.append(f"变异系数: {dist['coefficient_of_variation']}")
        report.append("各分片分布:")
        for shard, stats in dist['shard_distribution'].items():
            extra = f" [{stats.get('time_range', '')}]" if 'time_range' in stats else ""
            report.append(f"  {shard:10s}: {stats['record_count']:6d} 记录 ({stats['percentage']}%){extra}")
        report.append("")
        
        report.append("=" * 80)
        
        return "\n".join(report)
    
    def generate_charts(self, output_dir: str = '.'):
        plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial']
        plt.rcParams['axes.unicode_minus'] = False
        
        fig = plt.figure(figsize=(18, 16))
        gs = fig.add_gridspec(4, 2, hspace=0.5, wspace=0.3)
        
        query_freq = self.analyzer.analyze_query_frequency()
        fields = list(query_freq['field_distribution'].keys())
        counts = [query_freq['field_distribution'][f]['count'] for f in fields]
        
        ax1 = fig.add_subplot(gs[0, 0])
        ax1.bar(fields, counts, color='skyblue')
        ax1.set_title('查询频率分布')
        ax1.set_ylabel('查询次数')
        ax1.tick_params(axis='x', rotation=45)
        
        query_types = self.analyzer.analyze_query_types()
        fields = list(query_types.keys())
        point_ratios = [query_types[f]['point_query_ratio'] for f in fields]
        range_ratios = [query_types[f]['range_query_ratio'] for f in fields]
        
        x = np.arange(len(fields))
        width = 0.35
        ax2 = fig.add_subplot(gs[0, 1])
        ax2.bar(x - width/2, point_ratios, width, label='等值查询', color='lightgreen')
        ax2.bar(x + width/2, range_ratios, width, label='范围查询', color='salmon')
        ax2.set_title('查询类型比例')
        ax2.set_ylabel('比例 (%)')
        ax2.set_xticks(x)
        ax2.set_xticklabels(fields, rotation=45)
        ax2.legend()
        
        write_dist = self.recommendation.get('write_distribution_analysis', {})
        write_fields = ['user_id', 'product_id', 'timestamp', 'category', 'region']
        write_scores = [write_dist[f]['write_balance_score'] for f in write_fields if f in write_dist]
        write_field_names = [f for f in write_fields if f in write_dist]
        
        ax3 = fig.add_subplot(gs[1, 0])
        colors = ['red' if f == 'timestamp' and write_dist[f].get('hotspot_factor', 1.0) > 1.5 
                  else 'skyblue' for f in write_field_names]
        ax3.bar(write_field_names, write_scores, color=colors)
        ax3.set_title('写入均衡性评分')
        ax3.set_ylabel('评分 (0-100)')
        ax3.set_ylim(0, 100)
        ax3.tick_params(axis='x', rotation=45)
        
        composite_analysis = self.recommendation.get('composite_key_analysis', {})
        if composite_analysis:
            sorted_composites = sorted(composite_analysis.items(), 
                                       key=lambda x: x[1]['total_score'], reverse=True)[:6]
            comp_names = [name for name, _ in sorted_composites]
            comp_scores = [data['total_score'] for _, data in sorted_composites]
            comp_write = [data['write_balance_score'] for _, data in sorted_composites]
            
            ax4 = fig.add_subplot(gs[1, 1])
            x_pos = np.arange(len(comp_names))
            ax4.bar(x_pos - 0.2, comp_scores, 0.4, label='综合评分', color='skyblue')
            ax4.bar(x_pos + 0.2, comp_write, 0.4, label='写入均衡', color='lightgreen')
            ax4.set_title('复合分片键评分 (Top 6)')
            ax4.set_ylabel('评分')
            ax4.set_xticks(x_pos)
            ax4.set_xticklabels(comp_names, rotation=45, fontsize=8)
            ax4.legend()
        
        all_routing = self.recommendation.get('all_routing_efficiency', {})
        if all_routing:
            sorted_routing = sorted(all_routing.items(), 
                                    key=lambda x: x[1]['routing_efficiency'], reverse=True)
            routing_names = [name for name, _ in sorted_routing]
            routing_eff = [r['routing_efficiency'] for _, r in sorted_routing]
            cross_shard = [r['avg_cross_shard_ratio'] for _, r in sorted_routing]
            
            ax5 = fig.add_subplot(gs[2, 0])
            x_pos = np.arange(len(routing_names))
            ax5.bar(x_pos - 0.2, routing_eff, 0.4, label='路由效率 (%)', color='mediumpurple')
            ax5.bar(x_pos + 0.2, cross_shard, 0.4, label='跨分片率 (%)', color='salmon')
            ax5.set_title('查询路由效率对比')
            ax5.set_ylabel('比例 (%)')
            ax5.set_xticks(x_pos)
            ax5.set_xticklabels(routing_names, rotation=45, fontsize=8)
            ax5.legend()
        
        shard_calc = self.recommendation.get('shard_count_calculation', {})
        if shard_calc:
            evals = shard_calc.get('all_evaluations', {})
            if evals:
                shard_nums = sorted([int(k) for k in evals.keys()])
                avg_balances = [evals[str(n)]['avg_balance_score'] for n in shard_nums]
                fill_ratios = [evals[str(n)]['fill_ratio'] * 100 for n in shard_nums]
                
                ax6 = fig.add_subplot(gs[2, 1])
                ax6_twin = ax6.twinx()
                ax6.bar(shard_nums, avg_balances, color='skyblue', alpha=0.7, label='均衡性评分')
                ax6_twin.plot(shard_nums, fill_ratios, 'ro-', label='填充率 (%)')
                ax6_twin.axhline(y=85, color='red', linestyle='--', alpha=0.5, label='填充率上限')
                ax6.set_title('分片数量 vs 均衡性/填充率')
                ax6.set_xlabel('分片数')
                ax6.set_ylabel('均衡性评分')
                ax6_twin.set_ylabel('填充率 (%)')
                ax6.legend(loc='upper left')
                ax6_twin.legend(loc='upper right')
        
        ranking = self.recommendation['ranking']
        rank_fields = [r['field'] for r in ranking]
        rank_scores = [r['score'] for r in ranking]
        
        ax7 = fig.add_subplot(gs[3, 0])
        colors = ['gold' if i == 0 else 'lightblue' for i in range(len(ranking))]
        ax7.barh(rank_fields[::-1], rank_scores[::-1], color=colors[::-1])
        ax7.set_title('单字段分片键综合评分')
        ax7.set_xlabel('得分')
        ax7.set_xlim(0, 100)
        
        dist = self.recommendation['expected_distribution']
        shards = list(dist['shard_distribution'].keys())
        percentages = [dist['shard_distribution'][s]['percentage'] for s in shards]
        
        ax8 = fig.add_subplot(gs[3, 1])
        ax8.bar(shards, percentages, color='mediumpurple')
        key_name = self.recommendation['recommended_shard_key']
        ax8.set_title(f'推荐分片键 [{key_name}] 预期分布')
        ax8.set_ylabel('数据比例 (%)')
        ax8.tick_params(axis='x', rotation=45)
        
        plt.savefig(f'{output_dir}/shard_key_analysis.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"图表已保存至: {output_dir}/shard_key_analysis.png")


def main():
    print("正在生成模拟访问日志...")
    generator = AccessLogGenerator(num_records=10000, write_ratio=0.4)
    access_logs = generator.generate_access_logs()
    
    print("正在分析访问模式...")
    analyzer = AccessPatternAnalyzer(access_logs)
    
    print("正在计算分片键评分、复合键评估、分片数量和路由效率...")
    num_shards = 4
    recommender = ShardKeyRecommender(analyzer, num_shards=num_shards)
    recommendation = recommender.recommend_shard_key(
        num_shards=num_shards,
        shard_capacity_limit=500000,
        daily_growth_rate=0.05,
        growth_period_months=12
    )
    
    print("正在生成报告...")
    reporter = ReportGenerator(recommendation, analyzer)
    
    report = reporter.generate_text_report()
    print(report)
    
    reporter.generate_charts()
    
    with open('shard_key_recommendation.json', 'w', encoding='utf-8') as f:
        json.dump(recommendation, f, ensure_ascii=False, indent=2, default=str)
    
    print("推荐结果已保存至: shard_key_recommendation.json")


if __name__ == '__main__':
    main()
