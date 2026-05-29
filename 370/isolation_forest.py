import numpy as np
import warnings
from collections import Counter
from datetime import datetime


MIN_SAMPLES_RECOMMENDED = 100


class IsolationTree:
    def __init__(self, max_depth):
        self.max_depth = max_depth
        self.tree = None

    def fit(self, X):
        self.n_features = X.shape[1]
        self.tree = self._build_tree(X, depth=0)
        return self

    def _build_tree(self, X, depth):
        n_samples = X.shape[0]

        if depth >= self.max_depth or n_samples <= 1:
            return {'type': 'leaf', 'size': n_samples}

        feature_idx = np.random.randint(0, self.n_features)
        feature_values = X[:, feature_idx]
        min_val, max_val = np.min(feature_values), np.max(feature_values)

        if min_val == max_val:
            return {'type': 'leaf', 'size': n_samples}

        split_value = np.random.uniform(min_val, max_val)

        left_mask = X[:, feature_idx] < split_value
        right_mask = ~left_mask

        return {
            'type': 'node',
            'feature_idx': feature_idx,
            'split_value': split_value,
            'left': self._build_tree(X[left_mask], depth + 1),
            'right': self._build_tree(X[right_mask], depth + 1)
        }

    def path_length(self, x):
        return self._path_length(x, self.tree, 0)

    def _path_length(self, x, node, current_depth):
        if node['type'] == 'leaf':
            if node['size'] <= 1:
                return current_depth
            return current_depth + self._c_factor(node['size'])

        feature_idx = node['feature_idx']
        split_value = node['split_value']

        if x[feature_idx] < split_value:
            return self._path_length(x, node['left'], current_depth + 1)
        else:
            return self._path_length(x, node['right'], current_depth + 1)

    def get_split_features(self, x):
        return self._get_split_features(x, self.tree, [])

    def _get_split_features(self, x, node, features):
        if node['type'] == 'leaf':
            return features

        feature_idx = node['feature_idx']
        split_value = node['split_value']
        features.append((feature_idx, split_value))

        if x[feature_idx] < split_value:
            return self._get_split_features(x, node['left'], features)
        else:
            return self._get_split_features(x, node['right'], features)

    @staticmethod
    def _c_factor(n):
        if n <= 1:
            return 0
        return 2 * (np.log(n - 1) + np.euler_gamma) - 2 * (n - 1) / n


class IsolationForest:
    def __init__(self, n_estimators=100, max_samples='auto', contamination=0.1, 
                 random_state=None, min_samples_threshold=100, auto_fallback=False):
        self.n_estimators = n_estimators
        self.max_samples = max_samples
        self.contamination = contamination
        self.random_state = random_state
        self.min_samples_threshold = min_samples_threshold
        self.auto_fallback = auto_fallback
        self.trees = []
        self._is_fallback = False
        self._fallback_model = None
        self._warnings = []

    def fit(self, X):
        if self.random_state is not None:
            np.random.seed(self.random_state)

        X = np.array(X)
        n_samples = X.shape[0]

        self._check_sample_size(n_samples)

        if n_samples < self.min_samples_threshold and self.auto_fallback:
            self._warn_and_fallback(n_samples)
            self._fit_fallback(X)
            return self

        if self.max_samples == 'auto':
            self.max_samples_ = min(256, n_samples)
        else:
            self.max_samples_ = min(self.max_samples, n_samples)

        self.max_depth = int(np.ceil(np.log2(self.max_samples_)))

        self.trees = []
        for _ in range(self.n_estimators):
            sample_indices = np.random.choice(n_samples, self.max_samples_, replace=False)
            X_subset = X[sample_indices]

            tree = IsolationTree(self.max_depth)
            tree.fit(X_subset)
            self.trees.append(tree)

        self._compute_threshold(X)
        return self

    def _check_sample_size(self, n_samples):
        if n_samples < self.min_samples_threshold:
            warning_msg = (
                f"样本数量({n_samples})小于推荐最小样本量({self.min_samples_threshold})，"
                f"孤立森林算法得分可能不准确。建议：\n"
                f"  1. 增加更多训练样本（推荐>=100）\n"
                f"  2. 设置 auto_fallback=True 自动切换到One-Class SVM\n"
                f"  3. 对于小样本数据，考虑使用One-Class SVM、EllipticEnvelope等方法"
            )
            warnings.warn(warning_msg, UserWarning)
            self._warnings.append(warning_msg)

    def _warn_and_fallback(self, n_samples):
        self._is_fallback = True
        fallback_msg = (
            f"由于样本数量不足({n_samples} < {self.min_samples_threshold})，"
            f"已自动切换到One-Class SVM模型进行异常检测。"
        )
        warnings.warn(fallback_msg, UserWarning)
        self._warnings.append(fallback_msg)

    def _fit_fallback(self, X):
        try:
            from sklearn.svm import OneClassSVM
            from sklearn.preprocessing import StandardScaler
            
            self._scaler = StandardScaler()
            X_scaled = self._scaler.fit_transform(X)
            
            self._fallback_model = OneClassSVM(
                kernel='rbf',
                nu=self.contamination,
                gamma='scale'
            )
            self._fallback_model.fit(X_scaled)
            
        except ImportError:
            raise ImportError(
                "scikit-learn is required for One-Class SVM fallback. "
                "Please install it with: pip install scikit-learn"
            )

    def anomaly_score(self, X):
        X = np.array(X)

        if self._is_fallback:
            return self._anomaly_score_fallback(X)

        scores = []
        for x in X:
            path_lengths = [tree.path_length(x) for tree in self.trees]
            avg_path_length = np.mean(path_lengths)
            c_n = IsolationTree._c_factor(self.max_samples_)
            score = 2 ** (-avg_path_length / c_n)
            scores.append(score)

        return np.array(scores)

    def _anomaly_score_fallback(self, X):
        X_scaled = self._scaler.transform(X)
        decision_scores = self._fallback_model.decision_function(X_scaled)
        
        min_score, max_score = decision_scores.min(), decision_scores.max()
        if max_score == min_score:
            return np.full(len(X), 0.5)
        
        normalized_scores = 1 - (decision_scores - min_score) / (max_score - min_score)
        return normalized_scores

    def predict(self, X):
        if self._is_fallback:
            X_scaled = self._scaler.transform(np.array(X))
            return self._fallback_model.predict(X_scaled)
        
        scores = self.anomaly_score(X)
        return np.where(scores >= self.threshold_, -1, 1)

    def fit_predict(self, X):
        self.fit(X)
        return self.predict(X)

    def _compute_threshold(self, X):
        scores = self.anomaly_score(X)
        self.threshold_ = np.percentile(scores, 100 * (1 - self.contamination))

    def get_warnings(self):
        return self._warnings.copy()

    def is_using_fallback(self):
        return self._is_fallback

    def explain_anomaly(self, x, top_k=3):
        x = np.array(x)
        if x.ndim == 1:
            x = x.reshape(1, -1)
        
        if self._is_fallback:
            return self._explain_anomaly_fallback(x[0], top_k)
        
        all_split_features = []
        for tree in self.trees:
            splits = tree.get_split_features(x[0])
            all_split_features.extend([f[0] for f in splits])
        
        feature_counts = Counter(all_split_features)
        total_splits = len(all_split_features)
        
        feature_importance = {}
        for feature_idx, count in feature_counts.most_common():
            importance = count / total_splits if total_splits > 0 else 0
            feature_importance[feature_idx] = {
                'importance': importance,
                'count': count
            }
        
        top_features = sorted(
            feature_importance.items(),
            key=lambda x: x[1]['importance'],
            reverse=True
        )[:top_k]
        
        return {
            'top_features': [
                {
                    'feature_idx': idx,
                    'importance': imp['importance'],
                    'split_count': imp['count']
                } for idx, imp in top_features
            ],
            'feature_values': x[0].tolist()
        }

    def _explain_anomaly_fallback(self, x, top_k):
        return {
            'top_features': [
                {
                    'feature_idx': i,
                    'importance': 1.0 / len(x),
                    'split_count': 0
                } for i in range(min(top_k, len(x)))
            ],
            'feature_values': x.tolist(),
            'note': 'Using One-Class SVM fallback mode - feature importance not available'
        }

    def explain_batch(self, X, feature_names=None):
        X = np.array(X)
        scores = self.anomaly_score(X)
        labels = self.predict(X)
        
        results = []
        for i in range(len(X)):
            explanation = self.explain_anomaly(X[i])
            results.append({
                'index': i,
                'score': float(scores[i]),
                'is_anomaly': bool(labels[i] == -1),
                'explanation': explanation,
                'feature_values': X[i].tolist()
            })
        
        return results

    def predict_online(self, x, feature_names=None):
        x = np.array(x)
        if x.ndim == 1:
            x = x.reshape(1, -1)
        
        score = self.anomaly_score(x)[0]
        label = self.predict(x)[0]
        explanation = self.explain_anomaly(x[0])
        
        return {
            'score': float(score),
            'is_anomaly': bool(label == -1),
            'explanation': explanation,
            'timestamp': datetime.now().isoformat()
        }

    def generate_report(self, X, top_n=10, feature_names=None):
        X = np.array(X)
        scores = self.anomaly_score(X)
        labels = self.predict(X)
        
        anomaly_indices = np.where(labels == -1)[0]
        
        if len(anomaly_indices) == 0:
            return {
                'total_samples': len(X),
                'anomaly_count': 0,
                'top_anomalies': [],
                'summary': 'No anomalies detected'
            }
        
        anomaly_scores = scores[anomaly_indices]
        sorted_idx = np.argsort(anomaly_scores)[::-1][:top_n]
        top_anomaly_indices = anomaly_indices[sorted_idx]
        
        top_anomalies = []
        for idx in top_anomaly_indices:
            explanation = self.explain_anomaly(X[idx])
            top_anomalies.append({
                'rank': len(top_anomalies) + 1,
                'original_index': int(idx),
                'score': float(scores[idx]),
                'feature_values': X[idx].tolist(),
                'explanation': explanation
            })
        
        return {
            'total_samples': len(X),
            'anomaly_count': int(len(anomaly_indices)),
            'anomaly_rate': float(len(anomaly_indices) / len(X)),
            'top_anomalies': top_anomalies,
            'threshold': float(self.threshold_) if hasattr(self, 'threshold_') else None,
            'model_type': 'One-Class SVM (fallback)' if self._is_fallback else 'Isolation Forest'
        }

    def print_report(self, X, top_n=10, feature_names=None):
        report = self.generate_report(X, top_n, feature_names)
        
        print("=" * 80)
        print("异常检测报告")
        print("=" * 80)
        print(f"模型类型: {report['model_type']}")
        print(f"总样本数: {report['total_samples']}")
        print(f"异常样本数: {report['anomaly_count']}")
        print(f"异常率: {report['anomaly_rate']:.2%}")
        if report['threshold'] is not None:
            print(f"异常阈值: {report['threshold']:.4f}")
        print("-" * 80)
        
        if not report['top_anomalies']:
            print("未检测到异常样本。")
        else:
            print(f"Top {len(report['top_anomalies'])} 异常点:")
            for anomaly in report['top_anomalies']:
                print(f"\n  排名 {anomaly['rank']}. 样本 #{anomaly['original_index']}")
                print(f"     异常得分: {anomaly['score']:.4f}")
                print(f"     特征值: {[f'{v:.4f}' for v in anomaly['feature_values']]}")
                print("     主要贡献特征:")
                for feat in anomaly['explanation']['top_features']:
                    fname = feature_names[feat['feature_idx']] if feature_names else f"特征{feat['feature_idx']}"
                    print(f"       - {fname}: 重要性 {feat['importance']:.2%} (分裂 {feat['split_count']} 次)")
        
        print("=" * 80)
        return report


def test_small_samples_with_fallback():
    print("=" * 60)
    print("测试1: 小样本数据 + 自动降级到One-Class SVM")
    print("=" * 60)
    
    np.random.seed(42)
    X_small = np.random.randn(50, 2)
    X_small[:3] += np.array([8, 8])
    
    iforest_fallback = IsolationForest(
        contamination=0.1,
        random_state=42,
        min_samples_threshold=100,
        auto_fallback=True
    )
    
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        labels = iforest_fallback.fit_predict(X_small)
        scores = iforest_fallback.anomaly_score(X_small)
        
        print(f"\n样本数量: {len(X_small)}")
        print(f"是否使用降级模式: {iforest_fallback.is_using_fallback()}")
        print(f"异常得分范围: [{scores.min():.4f}, {scores.max():.4f}]")
        print(f"检测到异常数: {np.sum(labels == -1)}")
        
        print("\n警告信息:")
        for warning in iforest_fallback.get_warnings():
            print(f"  - {warning[:80]}...")


def test_small_samples_without_fallback():
    print("\n" + "=" * 60)
    print("测试2: 小样本数据 + 不降级（仅警告）")
    print("=" * 60)
    
    np.random.seed(42)
    X_small = np.random.randn(30, 2)
    X_small[:2] += np.array([6, 6])
    
    iforest = IsolationForest(
        contamination=0.1,
        random_state=42,
        min_samples_threshold=100,
        auto_fallback=False
    )
    
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        labels = iforest.fit_predict(X_small)
        scores = iforest.anomaly_score(X_small)
        
        print(f"\n样本数量: {len(X_small)}")
        print(f"是否使用降级模式: {iforest.is_using_fallback()}")
        print(f"异常得分范围: [{scores.min():.4f}, {scores.max():.4f}]")
        print(f"检测到异常数: {np.sum(labels == -1)}")


def test_normal_samples():
    print("\n" + "=" * 60)
    print("测试3: 正常样本数据（>=100）+ 标准孤立森林")
    print("=" * 60)
    
    from sklearn.datasets import make_blobs
    
    X, _ = make_blobs(n_samples=200, centers=1, cluster_std=1.0, random_state=42)
    outliers = np.random.uniform(low=-8, high=8, size=(20, 2))
    X = np.vstack([X, outliers])
    
    iforest = IsolationForest(n_estimators=100, contamination=0.1, random_state=42)
    iforest.fit(X)
    
    scores = iforest.anomaly_score(X)
    labels = iforest.predict(X)
    
    print(f"\n样本数量: {len(X)}")
    print(f"是否使用降级模式: {iforest.is_using_fallback()}")
    print(f"异常得分范围: [{scores.min():.4f}, {scores.max():.4f}]")
    print(f"检测到异常数: {np.sum(labels == -1)}")
    print(f"正常样本数: {np.sum(labels == 1)}")
    print("\n标准孤立森林模式运行成功！")


def test_anomaly_explanation():
    print("\n" + "=" * 60)
    print("测试4: 异常解释功能")
    print("=" * 60)
    
    from sklearn.datasets import make_blobs
    
    X, _ = make_blobs(n_samples=300, centers=1, cluster_std=1.0, random_state=42)
    outliers = np.array([[10, 10], [-8, 5], [5, -7]])
    X = np.vstack([X, outliers])
    
    iforest = IsolationForest(n_estimators=100, contamination=0.05, random_state=42)
    iforest.fit(X)
    
    print("\n解释已知异常点 [10, 10]:")
    explanation = iforest.explain_anomaly([10, 10], top_k=2)
    print(f"  特征值: {explanation['feature_values']}")
    print("  主要贡献特征:")
    for feat in explanation['top_features']:
        print(f"    - 特征{feat['feature_idx']}: 重要性 {feat['importance']:.2%} (分裂{feat['split_count']}次)")
    
    print("\n解释正常点 [0, 0]:")
    explanation_normal = iforest.explain_anomaly([0, 0], top_k=2)
    print(f"  特征值: {explanation_normal['feature_values']}")
    print("  主要贡献特征:")
    for feat in explanation_normal['top_features']:
        print(f"    - 特征{feat['feature_idx']}: 重要性 {feat['importance']:.2%} (分裂{feat['split_count']}次)")
    
    print("\n异常解释功能测试完成！")


def test_online_prediction():
    print("\n" + "=" * 60)
    print("测试5: 在线预测功能")
    print("=" * 60)
    
    from sklearn.datasets import make_blobs
    
    X_train, _ = make_blobs(n_samples=200, centers=1, cluster_std=1.0, random_state=42)
    
    iforest = IsolationForest(n_estimators=100, contamination=0.05, random_state=42)
    iforest.fit(X_train)
    
    print("\n模拟在线数据流预测:")
    test_samples = [[0, 0], [2, 1], [10, 10], [-5, 3], [0, -1]]
    
    for i, sample in enumerate(test_samples):
        result = iforest.predict_online(sample)
        status = "异常" if result['is_anomaly'] else "正常"
        print(f"  样本{i+1}: {sample} -> 得分: {result['score']:.4f}, 判定: {status}")
        if result['is_anomaly']:
            top_feat = result['explanation']['top_features'][0]
            print(f"          主要异常特征: 特征{top_feat['feature_idx']} (重要性 {top_feat['importance']:.1%})")
    
    print("\n在线预测功能测试完成！")


def test_anomaly_report():
    print("\n" + "=" * 60)
    print("测试6: 异常报告功能 (Top N异常点)")
    print("=" * 60)
    
    from sklearn.datasets import make_blobs
    
    X, _ = make_blobs(n_samples=200, centers=1, cluster_std=1.0, random_state=42)
    outliers = np.random.uniform(low=-10, high=10, size=(15, 2))
    X = np.vstack([X, outliers])
    
    iforest = IsolationForest(n_estimators=100, contamination=0.05, random_state=42)
    iforest.fit(X)
    
    feature_names = ['温度', '压力']
    report = iforest.print_report(X, top_n=5, feature_names=feature_names)
    
    print(f"\n报告摘要:")
    print(f"  总样本数: {report['total_samples']}")
    print(f"  异常率: {report['anomaly_rate']:.2%}")
    print(f"  Top 1 异常得分: {report['top_anomalies'][0]['score']:.4f}")
    
    print("\n异常报告功能测试完成！")


def main():
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        
        test_small_samples_with_fallback()
        test_small_samples_without_fallback()
        test_normal_samples()
        test_anomaly_explanation()
        test_online_prediction()
        test_anomaly_report()
        
        print("\n" + "=" * 60)
        print("所有测试完成!")
        print("=" * 60)


if __name__ == '__main__':
    main()
