import numpy as np
from collections import defaultdict


class KnowledgeGraph:
    def __init__(self, triples):
        self.triples = triples
        self.entity2id = {}
        self.relation2id = {}
        self.id2entity = {}
        self.id2relation = {}
        self._build_vocab()
        
    def _build_vocab(self):
        entities = set()
        relations = set()
        for h, r, t in self.triples:
            entities.add(h)
            entities.add(t)
            relations.add(r)
        
        for i, entity in enumerate(entities):
            self.entity2id[entity] = i
            self.id2entity[i] = entity
        
        for i, relation in enumerate(relations):
            self.relation2id[relation] = i
            self.id2relation[i] = relation
    
    def get_id_triples(self):
        return [(self.entity2id[h], self.relation2id[r], self.entity2id[t]) 
                for h, r, t in self.triples]
    
    def num_entities(self):
        return len(self.entity2id)
    
    def num_relations(self):
        return len(self.relation2id)


class DataLoader:
    def __init__(self, kg, batch_size=128, negative_ratio=1):
        self.kg = kg
        self.batch_size = batch_size
        self.negative_ratio = negative_ratio
        self.id_triples = kg.get_id_triples()
        self.triple_set = set(self.id_triples)
        self.entity_count = kg.num_entities()
        self.relation_count = kg.num_relations()
        self.max_retries = 100
        
        self.hr_t = {}
        self.rt_h = {}
        for h, r, t in self.id_triples:
            if (h, r) not in self.hr_t:
                self.hr_t[(h, r)] = set()
            self.hr_t[(h, r)].add(t)
            
            if (r, t) not in self.rt_h:
                self.rt_h[(r, t)] = set()
            self.rt_h[(r, t)].add(h)
        
    def generate_negative_triples(self, positive_triple):
        h, r, t = positive_triple
        negatives = []
        
        for _ in range(self.negative_ratio):
            if np.random.random() < 0.5:
                negative = self._corrupt_head(h, r, t)
            else:
                negative = self._corrupt_tail(h, r, t)
            
            if negative is not None:
                negatives.append(negative)
        
        return negatives
    
    def _corrupt_head(self, h, r, t):
        valid_heads = list(set(range(self.entity_count)) - self.rt_h.get((r, t), set()))
        
        if not valid_heads:
            for _ in range(self.max_retries):
                new_h = np.random.randint(0, self.entity_count)
                if new_h != h and (new_h, r, t) not in self.triple_set:
                    return (new_h, r, t)
            return None
        
        new_h = np.random.choice(valid_heads)
        return (new_h, r, t)
    
    def _corrupt_tail(self, h, r, t):
        valid_tails = list(set(range(self.entity_count)) - self.hr_t.get((h, r), set()))
        
        if not valid_tails:
            for _ in range(self.max_retries):
                new_t = np.random.randint(0, self.entity_count)
                if new_t != t and (h, r, new_t) not in self.triple_set:
                    return (h, r, new_t)
            return None
        
        new_t = np.random.choice(valid_tails)
        return (h, r, new_t)
    
    def generate_batches(self):
        np.random.shuffle(self.id_triples)
        
        for i in range(0, len(self.id_triples), self.batch_size):
            batch_pos = self.id_triples[i:i + self.batch_size]
            batch_neg = []
            
            for triple in batch_pos:
                batch_neg.extend(self.generate_negative_triples(triple))
            
            yield batch_pos, batch_neg


def load_sample_data():
    triples = [
        ('比尔·盖茨', '创立', '微软'),
        ('比尔·盖茨', '是', '企业家'),
        ('微软', '总部在', '西雅图'),
        ('微软', '属于', '科技行业'),
        ('苹果', '属于', '科技行业'),
        ('苹果', '创立', '乔布斯'),
        ('乔布斯', '是', '企业家'),
        ('谷歌', '属于', '科技行业'),
        ('谷歌', '创立', '拉里·佩奇'),
        ('拉里·佩奇', '是', '企业家'),
        ('西雅图', '位于', '华盛顿州'),
        ('华盛顿州', '属于', '美国'),
        ('北京', '位于', '中国'),
        ('中国', '属于', '亚洲'),
        ('美国', '属于', '北美洲'),
    ]
    return KnowledgeGraph(triples)
