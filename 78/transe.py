import numpy as np


class TransE:
    def __init__(self, num_entities, num_relations, embedding_dim=50, margin=1.0, learning_rate=0.01):
        self.num_entities = num_entities
        self.num_relations = num_relations
        self.embedding_dim = embedding_dim
        self.margin = margin
        self.learning_rate = learning_rate
        
        self.entity_embeddings = None
        self.relation_embeddings = None
        self._initialize_embeddings()
    
    def _initialize_embeddings(self):
        bound = 6 / np.sqrt(self.embedding_dim)
        self.entity_embeddings = np.random.uniform(-bound, bound, (self.num_entities, self.embedding_dim))
        self.relation_embeddings = np.random.uniform(-bound, bound, (self.num_relations, self.embedding_dim))
        
        self.entity_embeddings = self._normalize(self.entity_embeddings)
        self.relation_embeddings = self._normalize(self.relation_embeddings)
    
    def _normalize(self, embeddings):
        norm = np.linalg.norm(embeddings, axis=1, keepdims=True)
        return embeddings / (norm + 1e-8)
    
    def _distance(self, h, r, t):
        return np.linalg.norm(h + r - t, axis=1)
    
    def predict_score(self, h, r, t):
        h_emb = self.entity_embeddings[h]
        r_emb = self.relation_embeddings[r]
        t_emb = self.entity_embeddings[t]
        return -np.linalg.norm(h_emb + r_emb - t_emb)
    
    def train_step(self, positive_triples, negative_triples):
        min_len = min(len(positive_triples), len(negative_triples))
        if min_len == 0:
            return 0
        
        positive_triples = positive_triples[:min_len]
        negative_triples = negative_triples[:min_len]
        
        pos_h = np.array([t[0] for t in positive_triples])
        pos_r = np.array([t[1] for t in positive_triples])
        pos_t = np.array([t[2] for t in positive_triples])
        
        neg_h = np.array([t[0] for t in negative_triples])
        neg_r = np.array([t[1] for t in negative_triples])
        neg_t = np.array([t[2] for t in negative_triples])
        
        pos_h_emb = self.entity_embeddings[pos_h]
        pos_r_emb = self.relation_embeddings[pos_r]
        pos_t_emb = self.entity_embeddings[pos_t]
        
        neg_h_emb = self.entity_embeddings[neg_h]
        neg_r_emb = self.relation_embeddings[neg_r]
        neg_t_emb = self.entity_embeddings[neg_t]
        
        pos_dist = self._distance(pos_h_emb, pos_r_emb, pos_t_emb)
        neg_dist = self._distance(neg_h_emb, neg_r_emb, neg_t_emb)
        
        loss = np.maximum(0, self.margin + pos_dist - neg_dist)
        
        mask = loss > 0
        if np.sum(mask) == 0:
            return 0
        
        pos_h_emb_grad = np.zeros_like(self.entity_embeddings)
        pos_t_emb_grad = np.zeros_like(self.entity_embeddings)
        pos_r_emb_grad = np.zeros_like(self.relation_embeddings)
        neg_h_emb_grad = np.zeros_like(self.entity_embeddings)
        neg_t_emb_grad = np.zeros_like(self.entity_embeddings)
        neg_r_emb_grad = np.zeros_like(self.relation_embeddings)
        
        for i in range(len(positive_triples)):
            if mask[i]:
                h_idx, r_idx, t_idx = positive_triples[i]
                grad = 2 * (pos_h_emb[i] + pos_r_emb[i] - pos_t_emb[i])
                pos_h_emb_grad[h_idx] += grad
                pos_r_emb_grad[r_idx] += grad
                pos_t_emb_grad[t_idx] -= grad
                
                h_idx_neg, r_idx_neg, t_idx_neg = negative_triples[i]
                grad_neg = 2 * (neg_h_emb[i] + neg_r_emb[i] - neg_t_emb[i])
                neg_h_emb_grad[h_idx_neg] -= grad_neg
                neg_r_emb_grad[r_idx_neg] -= grad_neg
                neg_t_emb_grad[t_idx_neg] += grad_neg
        
        self.entity_embeddings -= self.learning_rate * (pos_h_emb_grad + pos_t_emb_grad + neg_h_emb_grad + neg_t_emb_grad)
        self.relation_embeddings -= self.learning_rate * (pos_r_emb_grad + neg_r_emb_grad)
        
        self.entity_embeddings = self._normalize(self.entity_embeddings)
        
        return np.sum(loss[mask])
    
    def get_entity_embedding(self, entity_id):
        return self.entity_embeddings[entity_id]
    
    def get_relation_embedding(self, relation_id):
        return self.relation_embeddings[relation_id]
