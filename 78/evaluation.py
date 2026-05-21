import numpy as np


class LinkPredictionEvaluator:
    def __init__(self, model, kg):
        self.model = model
        self.kg = kg
        self.all_triples = set(kg.get_id_triples())
        self.num_entities = kg.num_entities()
    
    def _get_rank(self, h, r, t, filter_triples=True, head_prediction=False):
        if head_prediction:
            scores = []
            for candidate_h in range(self.num_entities):
                score = self.model.predict_score(candidate_h, r, t)
                scores.append((score, candidate_h))
            
            scores.sort(reverse=True)
            rank = 1
            for score, candidate_h in scores:
                if candidate_h == h:
                    break
                if filter_triples and (candidate_h, r, t) in self.all_triples:
                    continue
                rank += 1
        else:
            scores = []
            for candidate_t in range(self.num_entities):
                score = self.model.predict_score(h, r, candidate_t)
                scores.append((score, candidate_t))
            
            scores.sort(reverse=True)
            rank = 1
            for score, candidate_t in scores:
                if candidate_t == t:
                    break
                if filter_triples and (h, r, candidate_t) in self.all_triples:
                    continue
                rank += 1
        
        return rank
    
    def evaluate(self, test_triples, filter_triples=True):
        ranks = []
        for h, r, t in test_triples:
            rank_head = self._get_rank(h, r, t, filter_triples, head_prediction=True)
            rank_tail = self._get_rank(h, r, t, filter_triples, head_prediction=False)
            ranks.append(rank_head)
            ranks.append(rank_tail)
        
        ranks = np.array(ranks)
        mean_rank = np.mean(ranks)
        hits1 = np.mean(ranks <= 1)
        hits3 = np.mean(ranks <= 3)
        hits10 = np.mean(ranks <= 10)
        
        return {
            'mean_rank': mean_rank,
            'hits@1': hits1,
            'hits@3': hits3,
            'hits@10': hits10
        }
    
    def predict_tail(self, head, relation, top_k=10):
        h_id = self.kg.entity2id[head]
        r_id = self.kg.relation2id[relation]
        
        scores = []
        for t_id in range(self.num_entities):
            score = self.model.predict_score(h_id, r_id, t_id)
            scores.append((score, t_id))
        
        scores.sort(reverse=True)
        results = []
        for score, t_id in scores[:top_k]:
            results.append({
                'tail': self.kg.id2entity[t_id],
                'score': float(score)
            })
        
        return results
    
    def predict_head(self, tail, relation, top_k=10):
        t_id = self.kg.entity2id[tail]
        r_id = self.kg.relation2id[relation]
        
        scores = []
        for h_id in range(self.num_entities):
            score = self.model.predict_score(h_id, r_id, t_id)
            scores.append((score, h_id))
        
        scores.sort(reverse=True)
        results = []
        for score, h_id in scores[:top_k]:
            results.append({
                'head': self.kg.id2entity[h_id],
                'score': float(score)
            })
        
        return results
