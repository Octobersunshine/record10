import numpy as np


class TransET:
    def __init__(self, num_entities, num_relations, num_timestamps, 
                 embedding_dim=50, margin=1.0, learning_rate=0.01):
        self.num_entities = num_entities
        self.num_relations = num_relations
        self.num_timestamps = num_timestamps
        self.embedding_dim = embedding_dim
        self.margin = margin
        self.learning_rate = learning_rate
        
        self.entity_embeddings = None
        self.relation_embeddings = None
        self.timestamp_embeddings = None
        self._initialize_embeddings()
    
    def _initialize_embeddings(self):
        bound = 6 / np.sqrt(self.embedding_dim)
        self.entity_embeddings = np.random.uniform(-bound, bound, (self.num_entities, self.embedding_dim))
        self.relation_embeddings = np.random.uniform(-bound, bound, (self.num_relations, self.embedding_dim))
        self.timestamp_embeddings = np.random.uniform(-bound, bound, (self.num_timestamps, self.embedding_dim))
        
        self.entity_embeddings = self._normalize(self.entity_embeddings)
        self.relation_embeddings = self._normalize(self.relation_embeddings)
        self.timestamp_embeddings = self._normalize(self.timestamp_embeddings)
    
    def _normalize(self, embeddings):
        norm = np.linalg.norm(embeddings, axis=1, keepdims=True)
        return embeddings / (norm + 1e-8)
    
    def _distance(self, h, r, t, tau):
        return np.linalg.norm(h + r + tau - t, axis=1)
    
    def predict_score(self, h, r, t, ts):
        h_emb = self.entity_embeddings[h]
        r_emb = self.relation_embeddings[r]
        t_emb = self.entity_embeddings[t]
        tau_emb = self.timestamp_embeddings[ts]
        return -np.linalg.norm(h_emb + r_emb + tau_emb - t_emb)
    
    def train_step(self, positive_quadruples, negative_quadruples):
        pos_h = np.array([q[0] for q in positive_quadruples])
        pos_r = np.array([q[1] for q in positive_quadruples])
        pos_t = np.array([q[2] for q in positive_quadruples])
        pos_ts = np.array([q[3][0] if isinstance(q[3], (list, tuple)) else q[3] for q in positive_quadruples])
        
        neg_h = np.array([q[0] for q in negative_quadruples])
        neg_r = np.array([q[1] for q in negative_quadruples])
        neg_t = np.array([q[2] for q in negative_quadruples])
        neg_ts = np.array([q[3] for q in negative_quadruples])
        
        min_len = min(len(pos_h), len(neg_h))
        if min_len == 0:
            return 0
        
        pos_h = pos_h[:min_len]
        pos_r = pos_r[:min_len]
        pos_t = pos_t[:min_len]
        pos_ts = pos_ts[:min_len]
        neg_h = neg_h[:min_len]
        neg_r = neg_r[:min_len]
        neg_t = neg_t[:min_len]
        neg_ts = neg_ts[:min_len]
        
        pos_h_emb = self.entity_embeddings[pos_h]
        pos_r_emb = self.relation_embeddings[pos_r]
        pos_t_emb = self.entity_embeddings[pos_t]
        pos_ts_emb = self.timestamp_embeddings[pos_ts]
        
        neg_h_emb = self.entity_embeddings[neg_h]
        neg_r_emb = self.relation_embeddings[neg_r]
        neg_t_emb = self.entity_embeddings[neg_t]
        neg_ts_emb = self.timestamp_embeddings[neg_ts]
        
        pos_dist = self._distance(pos_h_emb, pos_r_emb, pos_t_emb, pos_ts_emb)
        neg_dist = self._distance(neg_h_emb, neg_r_emb, neg_t_emb, neg_ts_emb)
        
        loss = np.maximum(0, self.margin + pos_dist - neg_dist)
        
        mask = loss > 0
        if np.sum(mask) == 0:
            return 0
        
        pos_h_emb_grad = np.zeros_like(self.entity_embeddings)
        pos_t_emb_grad = np.zeros_like(self.entity_embeddings)
        pos_r_emb_grad = np.zeros_like(self.relation_embeddings)
        pos_ts_emb_grad = np.zeros_like(self.timestamp_embeddings)
        
        neg_h_emb_grad = np.zeros_like(self.entity_embeddings)
        neg_t_emb_grad = np.zeros_like(self.entity_embeddings)
        neg_r_emb_grad = np.zeros_like(self.relation_embeddings)
        neg_ts_emb_grad = np.zeros_like(self.timestamp_embeddings)
        
        for i in range(len(pos_h)):
            if mask[i]:
                h_idx, r_idx, t_idx, ts_idx = pos_h[i], pos_r[i], pos_t[i], pos_ts[i]
                grad = 2 * (pos_h_emb[i] + pos_r_emb[i] + pos_ts_emb[i] - pos_t_emb[i])
                pos_h_emb_grad[h_idx] += grad
                pos_r_emb_grad[r_idx] += grad
                pos_ts_emb_grad[ts_idx] += grad
                pos_t_emb_grad[t_idx] -= grad
                
                h_idx_neg, r_idx_neg, t_idx_neg, ts_idx_neg = neg_h[i], neg_r[i], neg_t[i], neg_ts[i]
                grad_neg = 2 * (neg_h_emb[i] + neg_r_emb[i] + neg_ts_emb[i] - neg_t_emb[i])
                neg_h_emb_grad[h_idx_neg] -= grad_neg
                neg_r_emb_grad[r_idx_neg] -= grad_neg
                neg_ts_emb_grad[ts_idx_neg] -= grad_neg
                neg_t_emb_grad[t_idx_neg] += grad_neg
        
        self.entity_embeddings -= self.learning_rate * (pos_h_emb_grad + pos_t_emb_grad + neg_h_emb_grad + neg_t_emb_grad)
        self.relation_embeddings -= self.learning_rate * (pos_r_emb_grad + neg_r_emb_grad)
        self.timestamp_embeddings -= self.learning_rate * (pos_ts_emb_grad + neg_ts_emb_grad)
        
        self.entity_embeddings = self._normalize(self.entity_embeddings)
        self.timestamp_embeddings = self._normalize(self.timestamp_embeddings)
        
        return np.sum(loss[mask])
    
    def get_entity_embedding(self, entity_id):
        return self.entity_embeddings[entity_id]
    
    def get_relation_embedding(self, relation_id):
        return self.relation_embeddings[relation_id]
    
    def get_timestamp_embedding(self, timestamp_id):
        return self.timestamp_embeddings[timestamp_id]
