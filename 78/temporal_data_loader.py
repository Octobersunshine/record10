import numpy as np


class TemporalKnowledgeGraph:
    def __init__(self, quadruples):
        self.quadruples = quadruples
        self.entity2id = {}
        self.relation2id = {}
        self.timestamp2id = {}
        self.id2entity = {}
        self.id2relation = {}
        self.id2timestamp = {}
        self._build_vocab()
        
    def _build_vocab(self):
        entities = set()
        relations = set()
        timestamps = set()
        
        for h, r, t, time_info in self.quadruples:
            entities.add(h)
            entities.add(t)
            relations.add(r)
            
            if isinstance(time_info, (list, tuple)):
                for ts in time_info:
                    timestamps.add(ts)
            else:
                timestamps.add(time_info)
        
        for i, entity in enumerate(entities):
            self.entity2id[entity] = i
            self.id2entity[i] = entity
        
        for i, relation in enumerate(relations):
            self.relation2id[relation] = i
            self.id2relation[i] = relation
        
        for i, ts in enumerate(sorted(timestamps)):
            self.timestamp2id[ts] = i
            self.id2timestamp[i] = ts
    
    def get_id_quadruples(self):
        id_quads = []
        for h, r, t, time_info in self.quadruples:
            h_id = self.entity2id[h]
            r_id = self.relation2id[r]
            t_id = self.entity2id[t]
            
            if isinstance(time_info, (list, tuple)):
                time_ids = [self.timestamp2id[ts] for ts in time_info]
            else:
                time_ids = [self.timestamp2id[time_info]]
            
            id_quads.append((h_id, r_id, t_id, time_ids))
        
        return id_quads
    
    def num_entities(self):
        return len(self.entity2id)
    
    def num_relations(self):
        return len(self.relation2id)
    
    def num_timestamps(self):
        return len(self.timestamp2id)


class TemporalDataLoader:
    def __init__(self, tkg, batch_size=128, negative_ratio=1):
        self.tkg = tkg
        self.batch_size = batch_size
        self.negative_ratio = negative_ratio
        self.id_quadruples = tkg.get_id_quadruples()
        self.quadruple_set = set()
        self.max_retries = 100
        
        for h_id, r_id, t_id, time_ids in self.id_quadruples:
            for ts_id in time_ids:
                self.quadruple_set.add((h_id, r_id, t_id, ts_id))
        
        self.entity_count = tkg.num_entities()
        self.relation_count = tkg.num_relations()
        self.timestamp_count = tkg.num_timestamps()
        
        self.hrt_ts = {}
        self.ts_hrt = {}
        for h_id, r_id, t_id, time_ids in self.id_quadruples:
            for ts_id in time_ids:
                key = (h_id, r_id, t_id)
                if key not in self.hrt_ts:
                    self.hrt_ts[key] = set()
                self.hrt_ts[key].add(ts_id)
                
                key_ts = (ts_id, h_id, r_id)
                if key_ts not in self.ts_hrt:
                    self.ts_hrt[key_ts] = set()
                self.ts_hrt[key_ts].add(t_id)
                
                key_ts2 = (ts_id, r_id, t_id)
                if key_ts2 not in self.ts_hrt:
                    self.ts_hrt[key_ts2] = set()
                self.ts_hrt[key_ts2].add(h_id)
        
    def generate_negative_quadruples(self, positive_quadruple):
        h, r, t, time_ids = positive_quadruple
        negatives = []
        
        ts = time_ids[0] if time_ids else 0
        
        for _ in range(self.negative_ratio):
            mode = np.random.randint(0, 3)
            
            if mode == 0:
                negative = self._corrupt_head(h, r, t, ts)
            elif mode == 1:
                negative = self._corrupt_tail(h, r, t, ts)
            else:
                negative = self._corrupt_time(h, r, t, ts)
            
            if negative is not None:
                negatives.append(negative)
        
        return negatives
    
    def _corrupt_head(self, h, r, t, ts):
        valid_heads = list(set(range(self.entity_count)) - 
                          self.ts_hrt.get((ts, r, t), set()) - {h})
        
        if not valid_heads:
            for _ in range(self.max_retries):
                new_h = np.random.randint(0, self.entity_count)
                if new_h != h and (new_h, r, t, ts) not in self.quadruple_set:
                    return (new_h, r, t, ts)
            return None
        
        new_h = np.random.choice(valid_heads)
        return (new_h, r, t, ts)
    
    def _corrupt_tail(self, h, r, t, ts):
        valid_tails = list(set(range(self.entity_count)) - 
                          self.ts_hrt.get((ts, h, r), set()) - {t})
        
        if not valid_tails:
            for _ in range(self.max_retries):
                new_t = np.random.randint(0, self.entity_count)
                if new_t != t and (h, r, new_t, ts) not in self.quadruple_set:
                    return (h, r, new_t, ts)
            return None
        
        new_t = np.random.choice(valid_tails)
        return (h, r, new_t, ts)
    
    def _corrupt_time(self, h, r, t, ts):
        valid_times = list(set(range(self.timestamp_count)) - 
                           self.hrt_ts.get((h, r, t), set()) - {ts})
        
        if not valid_times:
            for _ in range(self.max_retries):
                new_ts = np.random.randint(0, self.timestamp_count)
                if new_ts != ts and (h, r, t, new_ts) not in self.quadruple_set:
                    return (h, r, t, new_ts)
            return None
        
        new_ts = np.random.choice(valid_times)
        return (h, r, t, new_ts)
    
    def generate_batches(self):
        np.random.shuffle(self.id_quadruples)
        
        for i in range(0, len(self.id_quadruples), self.batch_size):
            batch_pos = self.id_quadruples[i:i + self.batch_size]
            batch_neg = []
            
            for quad in batch_pos:
                batch_neg.extend(self.generate_negative_quadruples(quad))
            
            yield batch_pos, batch_neg


def load_temporal_sample_data():
    quadruples = [
        ('比尔·盖茨', '创立', '微软', 1975),
        ('比尔·盖茨', '担任CEO', '微软', [1975, 1980, 1985, 1990, 1995, 2000]),
        ('史蒂夫·鲍尔默', '担任CEO', '微软', [2000, 2005, 2010, 2014]),
        ('萨提亚·纳德拉', '担任CEO', '微软', [2014, 2018, 2022, 2024]),
        ('史蒂夫·乔布斯', '创立', '苹果', 1976),
        ('史蒂夫·乔布斯', '担任CEO', '苹果', [1976, 1980, 1985, 1997, 2000, 2005, 2010]),
        ('蒂姆·库克', '担任CEO', '苹果', [2011, 2015, 2020, 2024]),
        ('拉里·佩奇', '创立', '谷歌', 1998),
        ('拉里·佩奇', '担任CEO', '谷歌', [1998, 2000, 2005, 2010]),
        ('桑达尔·皮查伊', '担任CEO', '谷歌', [2015, 2018, 2022, 2024]),
        ('微软', '总部在', '西雅图', [1975, 1985, 1995, 2005, 2015, 2024]),
        ('苹果', '总部在', '库比蒂诺', [1976, 1985, 1995, 2005, 2015, 2024]),
        ('谷歌', '总部在', '山景城', [1998, 2005, 2015, 2024]),
        ('微软', '属于', '科技行业', [1975, 1985, 1995, 2005, 2015, 2024]),
        ('苹果', '属于', '科技行业', [1976, 1985, 1995, 2005, 2015, 2024]),
        ('谷歌', '属于', '科技行业', [1998, 2005, 2015, 2024]),
    ]
    return TemporalKnowledgeGraph(quadruples)
