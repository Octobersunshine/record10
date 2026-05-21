import numpy as np


class TemporalLinkPredictionEvaluator:
    def __init__(self, model, tkg):
        self.model = model
        self.tkg = tkg
        self.all_quadruples = set()
        
        for h_id, r_id, t_id, time_ids in tkg.get_id_quadruples():
            for ts_id in time_ids:
                self.all_quadruples.add((h_id, r_id, t_id, ts_id))
        
        self.num_entities = tkg.num_entities()
        self.num_timestamps = tkg.num_timestamps()
    
    def _get_rank(self, h, r, t, ts, filter_quadruples=True, head_prediction=False):
        if head_prediction:
            scores = []
            for candidate_h in range(self.num_entities):
                score = self.model.predict_score(candidate_h, r, t, ts)
                scores.append((score, candidate_h))
            
            scores.sort(reverse=True)
            rank = 1
            for score, candidate_h in scores:
                if candidate_h == h:
                    break
                if filter_quadruples and (candidate_h, r, t, ts) in self.all_quadruples:
                    continue
                rank += 1
        else:
            scores = []
            for candidate_t in range(self.num_entities):
                score = self.model.predict_score(h, r, candidate_t, ts)
                scores.append((score, candidate_t))
            
            scores.sort(reverse=True)
            rank = 1
            for score, candidate_t in scores:
                if candidate_t == t:
                    break
                if filter_quadruples and (h, r, candidate_t, ts) in self.all_quadruples:
                    continue
                rank += 1
        
        return rank
    
    def evaluate(self, test_quadruples, filter_quadruples=True):
        ranks = []
        for h, r, t, time_ids in test_quadruples:
            for ts in time_ids:
                rank_head = self._get_rank(h, r, t, ts, filter_quadruples, head_prediction=True)
                rank_tail = self._get_rank(h, r, t, ts, filter_quadruples, head_prediction=False)
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
    
    def predict_tail(self, head, relation, timestamp, top_k=10):
        h_id = self.tkg.entity2id[head]
        r_id = self.tkg.relation2id[relation]
        ts_id = self.tkg.timestamp2id[timestamp]
        
        scores = []
        for t_id in range(self.num_entities):
            score = self.model.predict_score(h_id, r_id, t_id, ts_id)
            scores.append((score, t_id))
        
        scores.sort(reverse=True)
        results = []
        for score, t_id in scores[:top_k]:
            results.append({
                'tail': self.tkg.id2entity[t_id],
                'score': float(score)
            })
        
        return results
    
    def predict_head(self, tail, relation, timestamp, top_k=10):
        t_id = self.tkg.entity2id[tail]
        r_id = self.tkg.relation2id[relation]
        ts_id = self.tkg.timestamp2id[timestamp]
        
        scores = []
        for h_id in range(self.num_entities):
            score = self.model.predict_score(h_id, r_id, t_id, ts_id)
            scores.append((score, h_id))
        
        scores.sort(reverse=True)
        results = []
        for score, h_id in scores[:top_k]:
            results.append({
                'head': self.tkg.id2entity[h_id],
                'score': float(score)
            })
        
        return results
    
    def predict_time(self, head, relation, tail, top_k=10):
        h_id = self.tkg.entity2id[head]
        r_id = self.tkg.relation2id[relation]
        t_id = self.tkg.entity2id[tail]
        
        scores = []
        for ts_id in range(self.num_timestamps):
            score = self.model.predict_score(h_id, r_id, t_id, ts_id)
            scores.append((score, ts_id))
        
        scores.sort(reverse=True)
        results = []
        for score, ts_id in scores[:top_k]:
            results.append({
                'timestamp': self.tkg.id2timestamp[ts_id],
                'score': float(score)
            })
        
        return results
