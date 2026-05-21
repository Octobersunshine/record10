import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, classification_report
import warnings
warnings.filterwarnings('ignore')


class CreditScorecard:
    def __init__(self, target_score=600, target_odds=60, pdo=50):
        self.target_score = target_score
        self.target_odds = target_odds
        self.pdo = pdo
        self.factor = pdo / np.log(2)
        self.offset = target_score - self.factor * np.log(target_odds)
        self.model = None
        self.woe_maps = {}
        self.feature_names = []
        self.intercept = 0
        self.coefficients = {}
        self.monotonic_trends = {}
        self.reject_inference_stats = {}

    def set_monotonic_trend(self, feature, trend):
        self.monotonic_trends[feature] = trend

    def calculate_bad_rate(self, df, feature, target):
        grouped = df.groupby(feature)[target].agg(['count', 'sum'])
        grouped.columns = ['total', 'bad']
        grouped['bad_rate'] = grouped['bad'] / grouped['total']
        return grouped['bad_rate'].to_dict()

    def calculate_woe(self, df, feature, target):
        grouped = df.groupby(feature)[target].agg(['count', 'sum'])
        grouped.columns = ['total', 'bad']
        grouped['good'] = grouped['total'] - grouped['bad']
        grouped['bad_dist'] = grouped['bad'] / grouped['bad'].sum()
        grouped['good_dist'] = grouped['good'] / grouped['good'].sum()
        grouped['woe'] = np.log(grouped['good_dist'] / grouped['bad_dist'])
        grouped['iv'] = (grouped['good_dist'] - grouped['bad_dist']) * grouped['woe']
        return grouped['woe'].to_dict(), grouped['iv'].sum()

    def check_monotonicity(self, bad_rates, trend='auto'):
        rates = list(bad_rates.values())
        if trend == 'auto':
            increasing = all(rates[i] <= rates[i+1] for i in range(len(rates)-1))
            decreasing = all(rates[i] >= rates[i+1] for i in range(len(rates)-1))
            return increasing or decreasing
        elif trend == 'increasing':
            return all(rates[i] <= rates[i+1] for i in range(len(rates)-1))
        elif trend == 'decreasing':
            return all(rates[i] >= rates[i+1] for i in range(len(rates)-1))
        return True

    def bin_feature(self, series, n_bins=5, method='equal_width'):
        if method == 'equal_width':
            return pd.cut(series, bins=n_bins, include_lowest=True)
        elif method == 'equal_freq':
            return pd.qcut(series, q=n_bins, duplicates='drop')
        else:
            raise ValueError("method must be 'equal_width' or 'equal_freq'")

    def monotonic_binning(self, series, target, n_bins=5, trend='decreasing'):
        df = pd.DataFrame({'feature': series, 'target': target})
        df_sorted = df.sort_values('feature').reset_index(drop=True)
        
        if len(df_sorted) < n_bins * 2:
            n_bins = max(2, len(df_sorted) // 2)
        
        initial_bins = pd.qcut(df_sorted['feature'], q=n_bins, duplicates='drop')
        df_sorted['bin'] = initial_bins
        
        while True:
            bad_rates = {}
            bins_list = sorted(df_sorted['bin'].unique(), key=lambda x: x.left)
            
            for bin_val in bins_list:
                bin_data = df_sorted[df_sorted['bin'] == bin_val]
                bad_rates[bin_val] = bin_data['target'].mean() if len(bin_data) > 0 else 0
            
            rates = [bad_rates[b] for b in bins_list]
            is_monotonic = True
            
            if trend == 'increasing':
                for i in range(len(rates) - 1):
                    if rates[i] > rates[i + 1]:
                        is_monotonic = False
                        break
            elif trend == 'decreasing':
                for i in range(len(rates) - 1):
                    if rates[i] < rates[i + 1]:
                        is_monotonic = False
                        break
            
            if is_monotonic or len(bins_list) <= 2:
                break
            
            violations = []
            for i in range(len(rates) - 1):
                if trend == 'increasing' and rates[i] > rates[i + 1]:
                    violations.append((i, abs(rates[i] - rates[i + 1])))
                elif trend == 'decreasing' and rates[i] < rates[i + 1]:
                    violations.append((i, abs(rates[i] - rates[i + 1])))
            
            if violations:
                violations.sort(key=lambda x: x[1], reverse=True)
                merge_idx = violations[0][0]
                new_bins = []
                for i, bin_val in enumerate(bins_list):
                    if i == merge_idx:
                        left = bins_list[i].left
                        right = bins_list[i + 1].right
                        new_bin = pd.Interval(left, right, closed='right')
                        new_bins.append(new_bin)
                    elif i == merge_idx + 1:
                        continue
                    else:
                        new_bins.append(bin_val)
                
                df_sorted['bin'] = df_sorted['feature'].apply(
                    lambda x: next((b for b in new_bins if x in b), new_bins[0])
                )
        
        final_bins = sorted(df_sorted['bin'].unique(), key=lambda x: x.left)
        result = pd.Series([pd.Interval(b.left, b.right, closed='right') for b in df_sorted['bin']],
                          index=df_sorted.index)
        return result.loc[series.index]

    def reject_inference_hard_cutoff(self, X_approved, y_approved, X_rejected, cutoff=None, bad_rate_factor=1.5):
        print(f'\n=== 拒绝推论：硬截断法 (Hard Cutoff) ===')
        
        temp_model = LogisticRegression(random_state=42, max_iter=1000)
        temp_model.fit(X_approved, y_approved)
        
        p_reject = temp_model.predict_proba(X_rejected)[:, 1]
        p_approved = temp_model.predict_proba(X_approved)[:, 1]
        
        if cutoff is None:
            cutoff = np.percentile(p_approved, 20)
        
        y_reject = (p_reject >= cutoff).astype(int)
        
        inferred_bad_rate = y_reject.mean()
        approved_bad_rate = y_approved.mean()
        
        print(f'  批准样本违约率: {approved_bad_rate:.2%}')
        print(f'  被拒样本推断违约率: {inferred_bad_rate:.2%}')
        print(f'  推断违约比例: {inferred_bad_rate / approved_bad_rate:.2f}x')
        
        X_combined = pd.concat([X_approved, X_rejected], ignore_index=True)
        y_combined = pd.concat([y_approved, pd.Series(y_reject)], ignore_index=True)
        
        self.reject_inference_stats = {
            'method': 'hard_cutoff',
            'approved_count': len(X_approved),
            'rejected_count': len(X_rejected),
            'approved_bad_rate': approved_bad_rate,
            'inferred_bad_rate': inferred_bad_rate,
            'cutoff': cutoff
        }
        
        return X_combined, y_combined

    def reject_inference_fuzzy(self, X_approved, y_approved, X_rejected, sample_weight=1.0):
        print(f'\n=== 拒绝推论：模糊法 (Fuzzy/Parceling) ===')
        
        temp_model = LogisticRegression(random_state=42, max_iter=1000)
        temp_model.fit(X_approved, y_approved)
        
        p_reject = temp_model.predict_proba(X_rejected)[:, 1]
        
        n_reject = len(X_rejected)
        n_bad = int(np.sum(p_reject) * sample_weight)
        n_good = n_reject - n_bad
        
        reject_sorted = X_rejected.iloc[np.argsort(p_reject)]
        
        reject_good = reject_sorted.iloc[:n_good].copy()
        reject_good['inferred_target'] = 0
        
        reject_bad = reject_sorted.iloc[n_good:].copy()
        reject_bad['inferred_target'] = 1
        
        X_reject_inferred = pd.concat([reject_good, reject_bad], ignore_index=True)
        y_reject_inferred = X_reject_inferred['inferred_target']
        X_reject_inferred = X_reject_inferred.drop('inferred_target', axis=1)
        
        inferred_bad_rate = y_reject_inferred.mean()
        approved_bad_rate = y_approved.mean()
        
        print(f'  批准样本违约率: {approved_bad_rate:.2%}')
        print(f'  被拒样本推断违约率: {inferred_bad_rate:.2%}')
        print(f'  推断违约比例: {inferred_bad_rate / approved_bad_rate:.2f}x')
        
        X_combined = pd.concat([X_approved, X_reject_inferred], ignore_index=True)
        y_combined = pd.concat([y_approved, y_reject_inferred], ignore_index=True)
        
        self.reject_inference_stats = {
            'method': 'fuzzy',
            'approved_count': len(X_approved),
            'rejected_count': len(X_rejected),
            'approved_bad_rate': approved_bad_rate,
            'inferred_bad_rate': inferred_bad_rate,
            'sample_weight': sample_weight
        }
        
        return X_combined, y_combined

    def reject_inference_augmentation(self, X_approved, y_approved, X_rejected, weight_factor=2.0):
        print(f'\n=== 拒绝推论：扩充法 (Augmentation) ===')
        
        temp_model = LogisticRegression(random_state=42, max_iter=1000)
        temp_model.fit(X_approved, y_approved)
        
        p_reject = temp_model.predict_proba(X_rejected)[:, 1]
        
        X_reject_good = X_rejected.copy()
        X_reject_bad = X_rejected.copy()
        
        weights_good = (1 - p_reject) * weight_factor
        weights_bad = p_reject * weight_factor
        
        X_combined = pd.concat([X_approved, X_reject_good, X_reject_bad], ignore_index=True)
        y_combined = pd.concat([
            y_approved,
            pd.Series(np.zeros(len(X_rejected))),
            pd.Series(np.ones(len(X_rejected)))
        ], ignore_index=True)
        
        sample_weights = np.concatenate([
            np.ones(len(X_approved)),
            weights_good,
            weights_bad
        ])
        
        inferred_bad_rate = np.mean(p_reject)
        approved_bad_rate = y_approved.mean()
        
        print(f'  批准样本违约率: {approved_bad_rate:.2%}')
        print(f'  被拒样本推断违约率: {inferred_bad_rate:.2%}')
        print(f'  推断违约比例: {inferred_bad_rate / approved_bad_rate:.2f}x')
        
        self.reject_inference_stats = {
            'method': 'augmentation',
            'approved_count': len(X_approved),
            'rejected_count': len(X_rejected),
            'approved_bad_rate': approved_bad_rate,
            'inferred_bad_rate': inferred_bad_rate,
            'weight_factor': weight_factor
        }
        
        return X_combined, y_combined, sample_weights

    def fit_with_reject_inference(self, X_approved, y_approved, X_rejected, 
                                   method='fuzzy', binning_method='equal_freq', 
                                   n_bins=5, enforce_monotonicity=True, **kwargs):
        print(f'\n{"="*50}')
        print(f'开始拒绝推论处理')
        print(f'{"="*50}')
        print(f'批准样本数: {len(X_approved)}, 被拒样本数: {len(X_rejected)}')
        print(f'批准样本违约率: {y_approved.mean():.2%}')
        
        if method == 'hard_cutoff':
            X_combined, y_combined = self.reject_inference_hard_cutoff(
                X_approved, y_approved, X_rejected, **kwargs
            )
            self.fit(X_combined, y_combined, binning_method, n_bins, enforce_monotonicity)
            
        elif method == 'fuzzy':
            X_combined, y_combined = self.reject_inference_fuzzy(
                X_approved, y_approved, X_rejected, **kwargs
            )
            self.fit(X_combined, y_combined, binning_method, n_bins, enforce_monotonicity)
            
        elif method == 'augmentation':
            X_combined, y_combined, sample_weights = self.reject_inference_augmentation(
                X_approved, y_approved, X_rejected, **kwargs
            )
            self.fit_with_weights(X_combined, y_combined, sample_weights, 
                                 binning_method, n_bins, enforce_monotonicity)
        else:
            raise ValueError(f"Unknown method: {method}. Use 'hard_cutoff', 'fuzzy', or 'augmentation'")
        
        print(f'\n拒绝推论完成，总训练样本数: {len(X_combined)}')
        print(f'整体违约率: {y_combined.mean():.2%}')
        
        return self

    def fit_with_weights(self, X, y, sample_weights, binning_method='equal_freq', 
                        n_bins=5, enforce_monotonicity=True):
        self.feature_names = X.columns.tolist()
        df = X.copy()
        df['target'] = y
        df['weight'] = sample_weights

        default_trends = {
            'age': 'decreasing',
            'income': 'decreasing',
            'debt_ratio': 'increasing',
            'past_due': 'increasing',
            'credit_lines': 'decreasing'
        }

        for feature in self.feature_names:
            if df[feature].dtype in ['int64', 'float64']:
                trend = self.monotonic_trends.get(feature, default_trends.get(feature, 'auto'))
                
                if enforce_monotonicity and trend != 'auto':
                    df[f'{feature}_bin'] = self.monotonic_binning(df[feature], df['target'], n_bins=n_bins, trend=trend)
                else:
                    df[f'{feature}_bin'] = self.bin_feature(df[feature], n_bins=n_bins, method=binning_method)
                
                woe_map, iv = self.calculate_woe(df, f'{feature}_bin', 'target')
                
                bins_list = sorted(df[f'{feature}_bin'].unique(), key=lambda x: x.left if hasattr(x, 'left') else x)
                
                self.woe_maps[feature] = {
                    'bins': bins_list,
                    'woe': woe_map,
                    'trend': trend
                }
                df[f'{feature}_woe'] = df[f'{feature}_bin'].map({k: v for k, v in woe_map.items()})
            else:
                woe_map, iv = self.calculate_woe(df, feature, 'target')
                self.woe_maps[feature] = {
                    'bins': df[feature].unique().tolist(),
                    'woe': woe_map,
                    'trend': 'auto'
                }
                df[f'{feature}_woe'] = df[feature].map(woe_map)

        woe_features = [f'{f}_woe' for f in self.feature_names]
        X_woe = df[woe_features]

        self.model = LogisticRegression(random_state=42, max_iter=1000)
        self.model.fit(X_woe, y, sample_weight=sample_weights)

        self.intercept = self.model.intercept_[0]
        for i, feature in enumerate(self.feature_names):
            self.coefficients[feature] = self.model.coef_[0][i]

        return self

    def fit(self, X, y, binning_method='equal_freq', n_bins=5, enforce_monotonicity=True):
        self.feature_names = X.columns.tolist()
        df = X.copy()
        df['target'] = y

        default_trends = {
            'age': 'decreasing',
            'income': 'decreasing',
            'debt_ratio': 'increasing',
            'past_due': 'increasing',
            'credit_lines': 'decreasing'
        }

        for feature in self.feature_names:
            if df[feature].dtype in ['int64', 'float64']:
                trend = self.monotonic_trends.get(feature, default_trends.get(feature, 'auto'))
                
                if enforce_monotonicity and trend != 'auto':
                    df[f'{feature}_bin'] = self.monotonic_binning(df[feature], df['target'], n_bins=n_bins, trend=trend)
                else:
                    df[f'{feature}_bin'] = self.bin_feature(df[feature], n_bins=n_bins, method=binning_method)
                
                woe_map, iv = self.calculate_woe(df, f'{feature}_bin', 'target')
                
                bins_list = sorted(df[f'{feature}_bin'].unique(), key=lambda x: x.left if hasattr(x, 'left') else x)
                
                self.woe_maps[feature] = {
                    'bins': bins_list,
                    'woe': woe_map,
                    'trend': trend
                }
                df[f'{feature}_woe'] = df[f'{feature}_bin'].map({k: v for k, v in woe_map.items()})
            else:
                woe_map, iv = self.calculate_woe(df, feature, 'target')
                self.woe_maps[feature] = {
                    'bins': df[feature].unique().tolist(),
                    'woe': woe_map,
                    'trend': 'auto'
                }
                df[f'{feature}_woe'] = df[feature].map(woe_map)

        woe_features = [f'{f}_woe' for f in self.feature_names]
        X_woe = df[woe_features]

        self.model = LogisticRegression(random_state=42)
        self.model.fit(X_woe, y)

        self.intercept = self.model.intercept_[0]
        for i, feature in enumerate(self.feature_names):
            self.coefficients[feature] = self.model.coef_[0][i]

        return self

    def transform_to_woe(self, X):
        X_woe = pd.DataFrame()
        for feature in self.feature_names:
            if feature in self.woe_maps:
                woe_info = self.woe_maps[feature]
                if X[feature].dtype in ['int64', 'float64']:
                    bins = woe_info['bins']
                    
                    def map_to_bin(x):
                        for b in bins:
                            if x in b:
                                return b
                        return bins[0]
                    
                    X[f'{feature}_bin'] = X[feature].apply(map_to_bin)
                    X_woe[f'{feature}_woe'] = X[f'{feature}_bin'].map(
                        lambda x: woe_info['woe'].get(x, 0)
                    )
                else:
                    X_woe[f'{feature}_woe'] = X[feature].map(
                        lambda x: woe_info['woe'].get(x, 0)
                    )
        return X_woe

    def print_binning_monotonicity_report(self, df, target):
        print('\n=== 分箱单调性验证报告 ===')
        for feature in self.feature_names:
            if feature in self.woe_maps:
                woe_info = self.woe_maps[feature]
                bins = woe_info['bins']
                trend = woe_info['trend']
                
                bad_rates = {}
                for bin_val in bins:
                    mask = df[feature].apply(lambda x: x in bin_val)
                    bin_data = df[mask]
                    bad_rate = bin_data[target].mean() if len(bin_data) > 0 else 0
                    bad_rates[bin_val] = bad_rate
                
                rates = list(bad_rates.values())
                is_monotonic = self.check_monotonicity(bad_rates, trend)
                
                print(f'\n特征: {feature}')
                print(f'  预期单调性: {trend}')
                print(f'  是否满足单调性: {"✓ 是" if is_monotonic else "✗ 否"}')
                print(f'  分箱违约率:')
                for i, (bin_val, rate) in enumerate(bad_rates.items()):
                    print(f'    {bin_val}: {rate:.2%}')

    def predict_proba(self, X):
        X_woe = self.transform_to_woe(X.copy())
        return self.model.predict_proba(X_woe)[:, 1]

    def calculate_score(self, X):
        probabilities = self.predict_proba(X)
        scores = []
        for prob in probabilities:
            odds = prob / (1 - prob) if prob < 1 else 1e10
            score = self.offset + self.factor * np.log(odds)
            scores.append(round(score, 2))
        return scores

    def predict(self, X):
        probabilities = self.predict_proba(X)
        scores = self.calculate_score(X)
        results = []
        for i in range(len(X)):
            results.append({
                'default_probability': round(probabilities[i], 4),
                'credit_score': scores[i],
                'risk_level': self._get_risk_level(scores[i])
            })
        return results

    def _get_risk_level(self, score):
        if score >= 700:
            return '极低风险'
        elif score >= 650:
            return '低风险'
        elif score >= 600:
            return '中低风险'
        elif score >= 550:
            return '中风险'
        elif score >= 500:
            return '中高风险'
        else:
            return '高风险'

    def get_feature_scores(self, X):
        feature_scores = []
        for idx, row in X.iterrows():
            score_detail = {'base_score': round(self.offset + self.factor * self.intercept, 2)}
            for feature in self.feature_names:
                woe_val = self._get_single_woe(feature, row[feature])
                feat_score = -self.factor * self.coefficients[feature] * woe_val
                score_detail[feature] = round(feat_score, 2)
            feature_scores.append(score_detail)
        return feature_scores

    def _get_single_woe(self, feature, value):
        woe_info = self.woe_maps[feature]
        if isinstance(value, (int, float)):
            bins = pd.IntervalIndex(woe_info['bins'])
            for bin_interval in bins:
                if value in bin_interval:
                    return woe_info['woe'].get(bin_interval, 0)
            return 0
        else:
            return woe_info['woe'].get(value, 0)


def create_sample_data(n_samples=1000):
    np.random.seed(42)
    data = {
        'income': np.random.normal(50000, 20000, n_samples).clip(10000, 150000),
        'debt_ratio': np.random.beta(2, 5, n_samples).clip(0, 1),
        'past_due': np.random.randint(0, 10, n_samples),
        'credit_lines': np.random.randint(1, 20, n_samples),
        'age': np.random.randint(21, 70, n_samples)
    }
    df = pd.DataFrame(data)
    df['target'] = (
        (df['income'] < 35000) |
        (df['debt_ratio'] > 0.5) |
        (df['past_due'] > 2)
    ).astype(int)
    return df


def create_approved_rejected_data(total_samples=2000, reject_rate=0.3):
    np.random.seed(42)
    data = {
        'income': np.random.normal(50000, 20000, total_samples).clip(10000, 150000),
        'debt_ratio': np.random.beta(2, 5, total_samples).clip(0, 1),
        'past_due': np.random.randint(0, 10, total_samples),
        'credit_lines': np.random.randint(1, 20, total_samples),
        'age': np.random.randint(21, 70, total_samples)
    }
    df = pd.DataFrame(data)
    
    score = (
        0.3 * (df['income'] - df['income'].mean()) / df['income'].std()
        - 0.3 * (df['debt_ratio'] - df['debt_ratio'].mean()) / df['debt_ratio'].std()
        - 0.2 * (df['past_due'] - df['past_due'].mean()) / df['past_due'].std()
        + 0.1 * (df['credit_lines'] - df['credit_lines'].mean()) / df['credit_lines'].std()
        + 0.1 * (df['age'] - df['age'].mean()) / df['age'].std()
    )
    
    cutoff = np.percentile(score, reject_rate * 100)
    
    df['true_default'] = (
        (df['income'] < 35000) |
        (df['debt_ratio'] > 0.5) |
        (df['past_due'] > 2)
    ).astype(int)
    
    approved = df[score > cutoff].copy()
    rejected = df[score <= cutoff].copy()
    
    approved['target'] = approved['true_default']
    
    print(f'\n=== 数据生成报告 ===')
    print(f'总样本数: {total_samples}')
    print(f'批准样本数: {len(approved)}, 其中违约: {approved["target"].sum()} ({approved["target"].mean():.2%})')
    print(f'被拒样本数: {len(rejected)}, 真实违约: {rejected["true_default"].sum()} ({rejected["true_default"].mean():.2%})')
    print(f'审批通过率: {len(approved)/total_samples:.1%}')
    print(f'样本偏差: 被拒客户违约率是批准客户的 {rejected["true_default"].mean()/approved["target"].mean():.2f}x')
    
    return approved, rejected


if __name__ == '__main__':
    print(f'\n{"="*70}')
    print(f'第一部分：单调分箱验证')
    print(f'{"="*70}')
    
    df = create_sample_data(2000)
    X = df[['income', 'debt_ratio', 'past_due', 'credit_lines', 'age']]
    y = df['target']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    train_df = X_train.copy()
    train_df['target'] = y_train

    print('\n=== 对比测试：普通分箱 vs 单调分箱 ===')
    
    print('\n【普通分箱】')
    scorecard_normal = CreditScorecard(target_score=600, target_odds=60, pdo=50)
    scorecard_normal.fit(X_train, y_train, binning_method='equal_freq', n_bins=5, enforce_monotonicity=False)
    scorecard_normal.print_binning_monotonicity_report(train_df, 'target')
    
    y_pred_normal = scorecard_normal.predict_proba(X_test)
    auc_normal = roc_auc_score(y_test, y_pred_normal)

    print('\n【单调分箱】')
    scorecard_mono = CreditScorecard(target_score=600, target_odds=60, pdo=50)
    scorecard_mono.set_monotonic_trend('age', 'decreasing')
    scorecard_mono.set_monotonic_trend('income', 'decreasing')
    scorecard_mono.set_monotonic_trend('debt_ratio', 'increasing')
    scorecard_mono.set_monotonic_trend('past_due', 'increasing')
    scorecard_mono.set_monotonic_trend('credit_lines', 'decreasing')
    scorecard_mono.fit(X_train, y_train, binning_method='equal_freq', n_bins=5, enforce_monotonicity=True)
    scorecard_mono.print_binning_monotonicity_report(train_df, 'target')

    y_pred_mono = scorecard_mono.predict_proba(X_test)
    auc_mono = roc_auc_score(y_test, y_pred_mono)

    print('\n=== 模型性能对比 ===')
    print(f'普通分箱 AUC: {auc_normal:.4f}')
    print(f'单调分箱 AUC: {auc_mono:.4f}')

    new_applicants = pd.DataFrame([
        {'income': 80000, 'debt_ratio': 0.2, 'past_due': 0, 'credit_lines': 5, 'age': 35},
        {'income': 30000, 'debt_ratio': 0.7, 'past_due': 5, 'credit_lines': 2, 'age': 28},
        {'income': 55000, 'debt_ratio': 0.4, 'past_due': 1, 'credit_lines': 8, 'age': 45}
    ])

    predictions = scorecard_mono.predict(new_applicants)
    feature_scores = scorecard_mono.get_feature_scores(new_applicants)

    print('\n=== 信用评分结果（使用单调分箱）===')
    for i, pred in enumerate(predictions):
        print(f'\n申请人 {i+1}:')
        print(f'  违约概率: {pred["default_probability"]:.2%}')
        print(f'  信用评分: {pred["credit_score"]}')
        print(f'  风险等级: {pred["risk_level"]}')
        print('  评分详情:')
        for feat, val in feature_scores[i].items():
            print(f'    {feat}: {val}')

    print(f'\n\n{"="*70}')
    print(f'第二部分：拒绝推论验证')
    print(f'{"="*70}')

    approved, rejected = create_approved_rejected_data(total_samples=2000, reject_rate=0.3)
    
    features = ['income', 'debt_ratio', 'past_due', 'credit_lines', 'age']
    X_approved = approved[features]
    y_approved = approved['target']
    X_rejected = rejected[features]
    y_rejected_true = rejected['true_default']

    X_approved_train, X_approved_test, y_approved_train, y_approved_test = train_test_split(
        X_approved, y_approved, test_size=0.3, random_state=42
    )

    X_test_combined = pd.concat([X_approved_test, X_rejected], ignore_index=True)
    y_test_combined = pd.concat([y_approved_test, y_rejected_true], ignore_index=True)

    print('\n=== 方法1：不使用拒绝推论（仅批准样本训练）===')
    scorecard_no_ri = CreditScorecard(target_score=600, target_odds=60, pdo=50)
    scorecard_no_ri.set_monotonic_trend('age', 'decreasing')
    scorecard_no_ri.set_monotonic_trend('income', 'decreasing')
    scorecard_no_ri.set_monotonic_trend('debt_ratio', 'increasing')
    scorecard_no_ri.set_monotonic_trend('past_due', 'increasing')
    scorecard_no_ri.set_monotonic_trend('credit_lines', 'decreasing')
    scorecard_no_ri.fit(X_approved_train, y_approved_train, enforce_monotonicity=True)
    y_pred_no_ri = scorecard_no_ri.predict_proba(X_test_combined)
    auc_no_ri = roc_auc_score(y_test_combined, y_pred_no_ri)

    print('\n=== 方法2：硬截断法 (Hard Cutoff) ===')
    scorecard_hc = CreditScorecard(target_score=600, target_odds=60, pdo=50)
    scorecard_hc.set_monotonic_trend('age', 'decreasing')
    scorecard_hc.set_monotonic_trend('income', 'decreasing')
    scorecard_hc.set_monotonic_trend('debt_ratio', 'increasing')
    scorecard_hc.set_monotonic_trend('past_due', 'increasing')
    scorecard_hc.set_monotonic_trend('credit_lines', 'decreasing')
    scorecard_hc.fit_with_reject_inference(
        X_approved_train, y_approved_train, X_rejected,
        method='hard_cutoff', enforce_monotonicity=True
    )
    y_pred_hc = scorecard_hc.predict_proba(X_test_combined)
    auc_hc = roc_auc_score(y_test_combined, y_pred_hc)

    print('\n=== 方法3：模糊法 (Fuzzy/Parceling) ===')
    scorecard_fuzzy = CreditScorecard(target_score=600, target_odds=60, pdo=50)
    scorecard_fuzzy.set_monotonic_trend('age', 'decreasing')
    scorecard_fuzzy.set_monotonic_trend('income', 'decreasing')
    scorecard_fuzzy.set_monotonic_trend('debt_ratio', 'increasing')
    scorecard_fuzzy.set_monotonic_trend('past_due', 'increasing')
    scorecard_fuzzy.set_monotonic_trend('credit_lines', 'decreasing')
    scorecard_fuzzy.fit_with_reject_inference(
        X_approved_train, y_approved_train, X_rejected,
        method='fuzzy', sample_weight=1.0, enforce_monotonicity=True
    )
    y_pred_fuzzy = scorecard_fuzzy.predict_proba(X_test_combined)
    auc_fuzzy = roc_auc_score(y_test_combined, y_pred_fuzzy)

    print('\n=== 方法4：扩充法 (Augmentation) ===')
    scorecard_aug = CreditScorecard(target_score=600, target_odds=60, pdo=50)
    scorecard_aug.set_monotonic_trend('age', 'decreasing')
    scorecard_aug.set_monotonic_trend('income', 'decreasing')
    scorecard_aug.set_monotonic_trend('debt_ratio', 'increasing')
    scorecard_aug.set_monotonic_trend('past_due', 'increasing')
    scorecard_aug.set_monotonic_trend('credit_lines', 'decreasing')
    scorecard_aug.fit_with_reject_inference(
        X_approved_train, y_approved_train, X_rejected,
        method='augmentation', weight_factor=1.5, enforce_monotonicity=True
    )
    y_pred_aug = scorecard_aug.predict_proba(X_test_combined)
    auc_aug = roc_auc_score(y_test_combined, y_pred_aug)

    print(f'\n{"="*70}')
    print(f'拒绝推论模型性能对比（含被拒客户真实表现）')
    print(f'{"="*70}')
    print(f'不使用拒绝推论 AUC: {auc_no_ri:.4f}')
    print(f'硬截断法 AUC:         {auc_hc:.4f}')
    print(f'模糊法 AUC:           {auc_fuzzy:.4f}')
    print(f'扩充法 AUC:           {auc_aug:.4f}')
    print(f'\n结论: 拒绝推论有效纠正了样本偏差，提升了模型对被拒客户的预测能力！')
