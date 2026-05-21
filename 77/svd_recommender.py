import numpy as np
from collections import defaultdict
import math


class GRU4Rec:
    def __init__(self, n_items, embedding_dim=50, hidden_dim=100, 
                 n_epochs=10, lr=0.01, batch_size=32, dropout=0.0,
                 loss='bpr', neg_samples=1):
        self.n_items = n_items
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.n_epochs = n_epochs
        self.lr = lr
        self.batch_size = batch_size
        self.dropout = dropout
        self.loss = loss
        self.neg_samples = neg_samples
        
        self.item_embeddings = None
        self.Wz = None
        self.Wr = None
        self.Wh = None
        self.Uz = None
        self.Ur = None
        self.Uh = None
        self.bz = None
        self.br = None
        self.bh = None
        self.output_bias = None
        
        self._init_weights()

    def _init_weights(self):
        std = 1.0 / math.sqrt(self.embedding_dim)
        self.item_embeddings = np.random.normal(0, std, (self.n_items, self.embedding_dim))
        
        std_gru = 1.0 / math.sqrt(self.hidden_dim)
        self.Wz = np.random.normal(0, std_gru, (self.embedding_dim, self.hidden_dim))
        self.Wr = np.random.normal(0, std_gru, (self.embedding_dim, self.hidden_dim))
        self.Wh = np.random.normal(0, std_gru, (self.embedding_dim, self.hidden_dim))
        
        self.Uz = np.random.normal(0, std_gru, (self.hidden_dim, self.hidden_dim))
        self.Ur = np.random.normal(0, std_gru, (self.hidden_dim, self.hidden_dim))
        self.Uh = np.random.normal(0, std_gru, (self.hidden_dim, self.hidden_dim))
        
        self.bz = np.zeros(self.hidden_dim)
        self.br = np.zeros(self.hidden_dim)
        self.bh = np.zeros(self.hidden_dim)
        self.output_bias = np.zeros(self.n_items)

    def _sigmoid(self, x):
        return 1.0 / (1.0 + np.exp(-np.clip(x, -10, 10)))

    def _gru_step(self, x, h):
        z = self._sigmoid(x @ self.Wz + h @ self.Uz + self.bz)
        r = self._sigmoid(x @ self.Wr + h @ self.Ur + self.br)
        h_tilde = np.tanh(x @ self.Wh + (r * h) @ self.Uh + self.bh)
        h_new = (1 - z) * h + z * h_tilde
        return h_new, (z, r, h_tilde, h)

    def _forward_sequence(self, sequence):
        h = np.zeros(self.hidden_dim)
        hidden_states = []
        
        for item_id in sequence:
            x = self.item_embeddings[item_id]
            h, _ = self._gru_step(x, h)
            hidden_states.append(h.copy())
        
        return hidden_states

    def _predict_scores(self, hidden_state):
        scores = hidden_state @ self.item_embeddings.T + self.output_bias
        return scores

    def _sample_negative(self, pos_item, n_samples=1):
        negatives = []
        while len(negatives) < n_samples:
            neg = np.random.randint(0, self.n_items)
            if neg != pos_item:
                negatives.append(neg)
        return negatives

    def fit(self, sequences):
        n_sequences = len(sequences)
        
        for epoch in range(self.n_epochs):
            total_loss = 0.0
            indices = np.random.permutation(n_sequences)
            
            for idx in indices:
                sequence = sequences[idx]
                if len(sequence) < 2:
                    continue
                
                h = np.zeros(self.hidden_dim)
                
                for t in range(len(sequence) - 1):
                    current_item = sequence[t]
                    next_item = sequence[t + 1]
                    
                    x = self.item_embeddings[current_item]
                    h, cache = self._gru_step(x, h)
                    z, r, h_tilde, h_prev = cache
                    
                    scores = self._predict_scores(h)
                    pos_score = scores[next_item]
                    
                    neg_items = self._sample_negative(next_item, self.neg_samples)
                    
                    for neg_item in neg_items:
                        neg_score = scores[neg_item]
                        
                        if self.loss == 'bpr':
                            score_diff = pos_score - neg_score
                            sigmoid_diff = self._sigmoid(score_diff)
                            loss = -np.log(sigmoid_diff + 1e-10)
                            total_loss += loss
                            
                            d_pos = sigmoid_diff - 1
                            d_neg = 1 - sigmoid_diff
                        else:
                            pos_prob = self._sigmoid(pos_score)
                            neg_prob = self._sigmoid(neg_score)
                            d_pos = pos_prob - 1
                            d_neg = neg_prob
                            loss = -np.log(pos_prob + 1e-10) - np.log(1 - neg_prob + 1e-10)
                            total_loss += loss
                        
                        d_h = np.zeros(self.hidden_dim)
                        d_h += d_pos * self.item_embeddings[next_item]
                        d_h += d_neg * self.item_embeddings[neg_item]
                        
                        dz = d_h * (h_tilde - h_prev) * z * (1 - z)
                        dr = d_h * z * (1 - h_tilde ** 2) * h_prev * r * (1 - r)
                        dh_tilde = d_h * z * (1 - h_tilde ** 2)
                        
                        self.output_bias[next_item] -= self.lr * d_pos
                        self.output_bias[neg_item] -= self.lr * d_neg
                        
                        self.item_embeddings[next_item] -= self.lr * d_pos * h
                        self.item_embeddings[neg_item] -= self.lr * d_neg * h
                        
                        d_x = (dh_tilde @ self.Wh.T) + (dz @ self.Wz.T) + (dr @ self.Wr.T)
                        self.item_embeddings[current_item] -= self.lr * d_x
                        
                        self.Wh -= self.lr * np.outer(x, dh_tilde)
                        self.Wz -= self.lr * np.outer(x, dz)
                        self.Wr -= self.lr * np.outer(x, dr)
                        
                        self.Uh -= self.lr * np.outer(r * h_prev, dh_tilde)
                        self.Uz -= self.lr * np.outer(h_prev, dz)
                        self.Ur -= self.lr * np.outer(h_prev, dr)
                        
                        self.bh -= self.lr * dh_tilde
                        self.bz -= self.lr * dz
                        self.br -= self.lr * dr
            
            avg_loss = total_loss / n_sequences
            print(f"Epoch {epoch + 1}/{self.n_epochs}, Average Loss: {avg_loss:.4f}")
        
        return self

    def predict_next(self, sequence, top_k=10, exclude_seen=True):
        if len(sequence) == 0:
            return []
        
        hidden_states = self._forward_sequence(sequence)
        final_hidden = hidden_states[-1]
        
        scores = self._predict_scores(final_hidden)
        
        if exclude_seen:
            seen_items = set(sequence)
            for item in seen_items:
                scores[item] = -np.inf
        
        top_indices = np.argsort(scores)[::-1][:top_k]
        return [(item_id, scores[item_id]) for item_id in top_indices]


class ImplicitFeedbackSVD:
    def __init__(self, n_factors=50, n_epochs=20, lr=0.01, reg=0.02, 
                 neg_ratio=5, use_popularity_weighted_sampling=True):
        self.n_factors = n_factors
        self.n_epochs = n_epochs
        self.lr = lr
        self.reg = reg
        self.neg_ratio = neg_ratio
        self.use_popularity_weighted_sampling = use_popularity_weighted_sampling
        self.user_factors = None
        self.item_factors = None
        self.user_bias = None
        self.item_bias = None
        self.n_users = None
        self.n_items = None
        self.item_popularity = None
        self.user_pos_items = defaultdict(set)

    def _compute_item_popularity(self, ratings):
        popularity = np.sum(~np.isnan(ratings), axis=0)
        popularity = popularity ** 0.75
        self.item_popularity = popularity / np.sum(popularity)
        return self.item_popularity

    def _build_user_pos_items(self, ratings):
        for u in range(ratings.shape[0]):
            pos_items = set(np.where(~np.isnan(ratings[u]))[0])
            self.user_pos_items[u] = pos_items

    def _sample_negative_items(self, user_idx, n_samples):
        all_items = set(range(self.n_items))
        neg_candidates = list(all_items - self.user_pos_items[user_idx])
        
        if not neg_candidates:
            return []
        
        if self.use_popularity_weighted_sampling and self.item_popularity is not None:
            weights = self.item_popularity[neg_candidates]
            weights = weights / np.sum(weights)
            sampled = np.random.choice(neg_candidates, size=min(n_samples, len(neg_candidates)), 
                                      replace=False, p=weights)
        else:
            sampled = np.random.choice(neg_candidates, size=min(n_samples, len(neg_candidates)), 
                                      replace=False)
        
        return list(sampled)

    def fit(self, ratings):
        self.n_users, self.n_items = ratings.shape
        
        self.user_factors = np.random.normal(0, 0.1, (self.n_users, self.n_factors))
        self.item_factors = np.random.normal(0, 0.1, (self.n_items, self.n_factors))
        self.user_bias = np.zeros(self.n_users)
        self.item_bias = np.zeros(self.n_items)
        
        self._compute_item_popularity(ratings)
        self._build_user_pos_items(ratings)
        
        pos_user_indices, pos_item_indices = np.where(~np.isnan(ratings))
        
        for epoch in range(self.n_epochs):
            indices = np.arange(len(pos_user_indices))
            np.random.shuffle(indices)
            
            for idx in indices:
                u = pos_user_indices[idx]
                i = pos_item_indices[idx]
                
                pos_prediction = self._predict_single(u, i)
                pos_error = 1.0 - pos_prediction
                
                self.user_bias[u] += self.lr * (pos_error - self.reg * self.user_bias[u])
                self.item_bias[i] += self.lr * (pos_error - self.reg * self.item_bias[i])
                
                self.user_factors[u] += self.lr * (
                    pos_error * self.item_factors[i] - self.reg * self.user_factors[u]
                )
                self.item_factors[i] += self.lr * (
                    pos_error * self.user_factors[u] - self.reg * self.item_factors[i]
                )
                
                neg_items = self._sample_negative_items(u, self.neg_ratio)
                
                for j in neg_items:
                    neg_prediction = self._predict_single(u, j)
                    neg_error = 0.0 - neg_prediction
                    
                    self.user_bias[u] += self.lr * (neg_error - self.reg * self.user_bias[u])
                    self.item_bias[j] += self.lr * (neg_error - self.reg * self.item_bias[j])
                    
                    self.user_factors[u] += self.lr * (
                        neg_error * self.item_factors[j] - self.reg * self.user_factors[u]
                    )
                    self.item_factors[j] += self.lr * (
                        neg_error * self.user_factors[u] - self.reg * self.item_factors[j]
                    )
        
        return self

    def _predict_single(self, user_idx, item_idx):
        prediction = (
            self.user_bias[user_idx]
            + self.item_bias[item_idx]
            + np.dot(self.user_factors[user_idx], self.item_factors[item_idx])
        )
        return 1.0 / (1.0 + np.exp(-np.clip(prediction, -10, 10)))

    def predict(self):
        predictions = np.zeros((self.n_users, self.n_items))
        for u in range(self.n_users):
            for i in range(self.n_items):
                predictions[u, i] = self._predict_single(u, i)
        return predictions

    def predict_single(self, user_idx, item_idx):
        return self._predict_single(user_idx, item_idx)

    def recommend_items(self, user_idx, top_k=10, exclude_seen=True):
        scores = self.predict()[user_idx]
        
        if exclude_seen and user_idx in self.user_pos_items:
            for i in self.user_pos_items[user_idx]:
                scores[i] = -1
        
        top_indices = np.argsort(scores)[::-1][:top_k]
        return [(i, scores[i]) for i in top_indices]


class PureSVDRecommender:
    def __init__(self, n_factors=100):
        self.n_factors = n_factors
        self.U = None
        self.S = None
        self.Vt = None
        self.global_mean = None
        self.user_means = None
        self.item_means = None

    def fit(self, ratings):
        self.global_mean = np.nanmean(ratings)
        self.user_means = np.nanmean(ratings, axis=1, keepdims=True)
        self.item_means = np.nanmean(ratings, axis=0, keepdims=True)
        
        filled_ratings = self._fill_ratings(ratings)
        
        U, S, Vt = np.linalg.svd(filled_ratings, full_matrices=False)
        
        k = min(self.n_factors, len(S))
        self.U = U[:, :k]
        self.S = np.diag(S[:k])
        self.Vt = Vt[:k, :]
        
        return self

    def _fill_ratings(self, ratings):
        filled = ratings.copy()
        for u in range(ratings.shape[0]):
            for i in range(ratings.shape[1]):
                if np.isnan(ratings[u, i]):
                    filled[u, i] = self.global_mean
        return filled

    def predict(self):
        predictions = self.U @ self.S @ self.Vt
        return np.clip(predictions, 1, 5)

    def predict_single(self, user_idx, item_idx):
        prediction = self.U[user_idx] @ self.S @ self.Vt[:, item_idx]
        return np.clip(prediction, 1, 5)


class SVDRecommender:
    def __init__(self, n_factors=100, n_epochs=20, lr=0.005, reg=0.02):
        self.n_factors = n_factors
        self.n_epochs = n_epochs
        self.lr = lr
        self.reg = reg
        self.user_factors = None
        self.item_factors = None
        self.user_bias = None
        self.item_bias = None
        self.global_bias = None
        self.n_users = None
        self.n_items = None

    def fit(self, ratings):
        self.n_users, self.n_items = ratings.shape
        self.global_bias = np.nanmean(ratings)
        
        self.user_factors = np.random.normal(0, 0.1, (self.n_users, self.n_factors))
        self.item_factors = np.random.normal(0, 0.1, (self.n_items, self.n_factors))
        self.user_bias = np.zeros(self.n_users)
        self.item_bias = np.zeros(self.n_items)
        
        user_indices, item_indices = np.where(~np.isnan(ratings))
        
        for epoch in range(self.n_epochs):
            for u, i in zip(user_indices, item_indices):
                prediction = self._predict_single(u, i)
                error = ratings[u, i] - prediction
                
                self.user_bias[u] += self.lr * (error - self.reg * self.user_bias[u])
                self.item_bias[i] += self.lr * (error - self.reg * self.item_bias[i])
                
                self.user_factors[u] += self.lr * (
                    error * self.item_factors[i] - self.reg * self.user_factors[u]
                )
                self.item_factors[i] += self.lr * (
                    error * self.user_factors[u] - self.reg * self.item_factors[i]
                )
        
        return self

    def _predict_single(self, user_idx, item_idx):
        prediction = (
            self.global_bias
            + self.user_bias[user_idx]
            + self.item_bias[item_idx]
            + np.dot(self.user_factors[user_idx], self.item_factors[item_idx])
        )
        return prediction

    def predict(self):
        predictions = np.zeros((self.n_users, self.n_items))
        for u in range(self.n_users):
            for i in range(self.n_items):
                predictions[u, i] = self._predict_single(u, i)
        return predictions

    def predict_single(self, user_idx, item_idx):
        return self._predict_single(user_idx, item_idx)


def create_sample_matrix():
    ratings = np.array([
        [5.0, 3.0, np.nan, 1.0],
        [4.0, np.nan, np.nan, 1.0],
        [1.0, 1.0, np.nan, 5.0],
        [np.nan, np.nan, 4.0, 4.0],
        [np.nan, 1.0, 5.0, 4.0],
    ])
    return ratings


def create_implicit_feedback_matrix(n_users=10, n_items=20, sparsity=0.1):
    ratings = np.full((n_users, n_items), np.nan)
    n_interactions = int(n_users * n_items * sparsity)
    
    for _ in range(n_interactions):
        u = np.random.randint(0, n_users)
        i = np.random.randint(0, n_items)
        ratings[u, i] = 1.0
    
    return ratings


def generate_sequence_data(n_sequences=100, n_items=50, min_seq_len=5, max_seq_len=20):
    sequences = []
    
    for _ in range(n_sequences):
        seq_len = np.random.randint(min_seq_len, max_seq_len + 1)
        sequence = []
        
        current_item = np.random.randint(0, n_items)
        sequence.append(current_item)
        
        for _ in range(seq_len - 1):
            if np.random.random() < 0.7 and len(sequence) > 0:
                bias = np.random.randint(-5, 6)
                next_item = (sequence[-1] + bias) % n_items
            else:
                next_item = np.random.randint(0, n_items)
            
            sequence.append(next_item)
        
        sequences.append(sequence)
    
    return sequences


if __name__ == "__main__":
    print("=" * 70)
    print("【测试1：显式评分预测】")
    print("=" * 70)
    ratings = create_sample_matrix()
    print("\n原始评分矩阵:")
    print(ratings)
    
    print("\n【方法一：基于梯度下降的 FunkSVD】")
    funk_svd = SVDRecommender(n_factors=2, n_epochs=100, lr=0.01, reg=0.01)
    funk_svd.fit(ratings)
    funk_predictions = funk_svd.predict()
    print("预测后的完整评分矩阵:")
    print(np.round(funk_predictions, 2))
    print("\n缺失值预测结果:")
    for u in range(ratings.shape[0]):
        for i in range(ratings.shape[1]):
            if np.isnan(ratings[u, i]):
                print(f"  用户{u+1}对物品{i+1}的预测评分: {funk_predictions[u, i]:.2f}")
    
    print("\n【方法二：基于NumPy的纯SVD】")
    pure_svd = PureSVDRecommender(n_factors=2)
    pure_svd.fit(ratings)
    pure_predictions = pure_svd.predict()
    print("预测后的完整评分矩阵:")
    print(np.round(pure_predictions, 2))
    print("\n缺失值预测结果:")
    for u in range(ratings.shape[0]):
        for i in range(ratings.shape[1]):
            if np.isnan(ratings[u, i]):
                print(f"  用户{u+1}对物品{i+1}的预测评分: {pure_predictions[u, i]:.2f}")
    
    print("\n" + "=" * 70)
    print("【测试2：隐式反馈推荐（优化负采样）】")
    print("=" * 70)
    
    implicit_ratings = create_implicit_feedback_matrix(n_users=10, n_items=15, sparsity=0.15)
    print(f"\n隐式反馈矩阵形状: {implicit_ratings.shape}")
    print(f"有交互的样本数: {np.sum(~np.isnan(implicit_ratings))}")
    print(f"稀疏度: {np.sum(~np.isnan(implicit_ratings)) / (implicit_ratings.shape[0] * implicit_ratings.shape[1]):.2%}")
    
    print("\n【隐式反馈SVD - 负采样比例1:5 + 流行度加权采样】")
    implicit_svd = ImplicitFeedbackSVD(
        n_factors=10, 
        n_epochs=30, 
        lr=0.01, 
        reg=0.02,
        neg_ratio=5,
        use_popularity_weighted_sampling=True
    )
    implicit_svd.fit(implicit_ratings)
    
    print("\n为用户0推荐Top 5物品:")
    recommendations = implicit_svd.recommend_items(0, top_k=5, exclude_seen=True)
    for item_id, score in recommendations:
        print(f"  物品{item_id+1}: 预测得分 = {score:.4f}")
    
    print("\n为用户1推荐Top 5物品:")
    recommendations = implicit_svd.recommend_items(1, top_k=5, exclude_seen=True)
    for item_id, score in recommendations:
        print(f"  物品{item_id+1}: 预测得分 = {score:.4f}")
    
    print("\n" + "=" * 70)
    print("【负采样优化说明】")
    print("=" * 70)
    print("  ✓ 负采样比例: 1:5（每个正样本采样5个负样本）")
    print("  ✓ 避免了 1:1000 的极端比例导致的正样本淹没")
    print("  ✓ 支持流行度加权采样（困难负样本学习）")
    print("  ✓ 使用 sigmoid 输出置信度分数")
    
    print("\n" + "=" * 70)
    print("【测试3：序列推荐 - GRU4Rec】")
    print("=" * 70)
    
    n_items = 50
    sequences = generate_sequence_data(n_sequences=200, n_items=n_items, 
                                       min_seq_len=5, max_seq_len=15)
    
    print(f"\n数据集统计:")
    print(f"  序列数量: {len(sequences)}")
    print(f"  物品数量: {n_items}")
    print(f"  平均序列长度: {np.mean([len(s) for s in sequences]):.1f}")
    print(f"  序列长度范围: [{min(len(s) for s in sequences)}, {max(len(s) for s in sequences)}]")
    
    print("\n示例序列 (前5个):")
    for i, seq in enumerate(sequences[:5]):
        print(f"  序列{i+1}: {seq[:8]}{'...' if len(seq) > 8 else ''} (长度: {len(seq)})")
    
    print("\n【训练GRU4Rec模型】")
    gru4rec = GRU4Rec(
        n_items=n_items,
        embedding_dim=32,
        hidden_dim=64,
        n_epochs=5,
        lr=0.01,
        loss='bpr',
        neg_samples=1
    )
    gru4rec.fit(sequences)
    
    print("\n【下一个物品预测示例】")
    test_sequence = sequences[0][:5]
    print(f"\n输入序列: {test_sequence}")
    print(f"实际后续物品: {sequences[0][5:8]}...")
    
    predictions = gru4rec.predict_next(test_sequence, top_k=5)
    print("\n预测的Top 5下一个物品:")
    for rank, (item_id, score) in enumerate(predictions, 1):
        print(f"  {rank}. 物品{item_id}: 得分 = {score:.4f}")
    
    test_sequence2 = sequences[1][:3]
    print(f"\n输入序列: {test_sequence2}")
    predictions2 = gru4rec.predict_next(test_sequence2, top_k=5)
    print("预测的Top 5下一个物品:")
    for rank, (item_id, score) in enumerate(predictions2, 1):
        print(f"  {rank}. 物品{item_id}: 得分 = {score:.4f}")
    
    print("\n" + "=" * 70)
    print("【GRU4Rec序列推荐说明】")
    print("=" * 70)
    print("  ✓ 考虑用户行为时间顺序，建模序列依赖")
    print("  ✓ GRU门控循环单元捕捉长期和短期兴趣")
    print("  ✓ BPR损失函数优化排序性能")
    print("  ✓ 支持会话级推荐（下一个物品预测）")
    print("  ✓ 输出: Top-K候选物品及匹配分数")
    print("=" * 70)
