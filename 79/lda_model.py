#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import re
import jieba
from gensim import corpora, models
from gensim.models import CoherenceModel, LdaSeqModel
import numpy as np
from typing import List, Dict, Tuple, Optional, Union


class LDAModel:
    def __init__(self, num_topics: int = 5, random_state: Optional[int] = None, 
                 num_init: int = 5, selection_metric: str = 'coherence'):
        self.num_topics = num_topics
        self.random_state = random_state
        self.num_init = num_init
        self.selection_metric = selection_metric
        self.dictionary = None
        self.lda_model = None
        self.best_random_state = None
        self.all_models = []
        self.stop_words = self._load_stop_words()

    def _load_stop_words(self) -> set:
        default_stop_words = {
            '的', '了', '在', '是', '我', '有', '和', '就', '不', '人', '都', '一', '一个',
            '上', '也', '很', '到', '说', '要', '去', '你', '会', '着', '没有', '看', '好',
            '自己', '这', '那', '他', '她', '它', '们', '这个', '那个', '什么', '怎么',
            '为什么', '哪', '哪里', '谁', '多少', '几', '啊', '吧', '呢', '吗', '啦',
            'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of',
            'with', 'by', 'as', 'is', 'was', 'are', 'were', 'be', 'been', 'being',
            'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could', 'should',
            'may', 'might', 'must', 'shall', 'can', 'need', 'dare', 'ought', 'used'
        }
        return default_stop_words

    def _clean_text(self, text: str) -> str:
        text = re.sub(r'[^\w\s]', '', text)
        text = re.sub(r'\d+', '', text)
        text = text.lower()
        return text

    def _tokenize(self, text: str) -> List[str]:
        text = self._clean_text(text)
        words = jieba.lcut(text)
        words = [word for word in words if word.strip() and word not in self.stop_words and len(word) > 1]
        return words

    def preprocess_documents(self, documents: List[str]) -> List[List[str]]:
        return [self._tokenize(doc) for doc in documents]

    def _compute_perplexity(self, model, corpus) -> float:
        return float(-model.log_perplexity(corpus))

    def _compute_topic_mixture_score(self, model, corpus) -> float:
        doc_topic_dists = []
        for doc_bow in corpus:
            doc_topics = model.get_document_topics(doc_bow, minimum_probability=0)
            doc_topic_dists.append([prob for _, prob in doc_topics])
        
        doc_topic_dists = np.array(doc_topic_dists)
        entropy = -np.sum(doc_topic_dists * np.log(doc_topic_dists + 1e-10), axis=1)
        avg_entropy = np.mean(entropy)
        max_entropy = np.log(self.num_topics)
        normalized_entropy = avg_entropy / max_entropy if max_entropy > 0 else 0
        return float(normalized_entropy)

    def _train_single_model(self, corpus, processed_docs, random_seed: int):
        lda_model = models.LdaModel(
            corpus=corpus,
            id2word=self.dictionary,
            num_topics=self.num_topics,
            random_state=random_seed,
            update_every=1,
            chunksize=100,
            passes=10,
            alpha='auto',
            per_word_topics=True
        )
        
        coherence_model = CoherenceModel(
            model=lda_model,
            texts=processed_docs,
            dictionary=self.dictionary,
            coherence='c_v'
        )
        coherence = coherence_model.get_coherence()
        perplexity = self._compute_perplexity(lda_model, corpus)
        mixture_score = self._compute_topic_mixture_score(lda_model, corpus)
        
        return {
            'model': lda_model,
            'random_seed': random_seed,
            'coherence': coherence,
            'perplexity': perplexity,
            'mixture_score': mixture_score
        }

    def fit(self, documents: List[str]) -> None:
        processed_docs = self.preprocess_documents(documents)
        self.dictionary = corpora.Dictionary(processed_docs)
        self.dictionary.filter_extremes(no_below=2, no_above=0.9)
        corpus = [self.dictionary.doc2bow(doc) for doc in processed_docs]
        
        self.corpus = corpus
        self.processed_docs = processed_docs
        
        self.all_models = []
        
        if self.random_state is not None:
            seeds = [self.random_state + i * 1000 for i in range(self.num_init)]
        else:
            rng = np.random.RandomState(None)
            seeds = [rng.randint(0, 1000000) for _ in range(self.num_init)]
        
        for seed in seeds:
            model_info = self._train_single_model(corpus, processed_docs, seed)
            self.all_models.append(model_info)
        
        if self.selection_metric == 'coherence':
            self.all_models.sort(key=lambda x: x['coherence'], reverse=True)
        elif self.selection_metric == 'perplexity':
            self.all_models.sort(key=lambda x: x['perplexity'], reverse=False)
        elif self.selection_metric == 'mixture':
            self.all_models.sort(key=lambda x: x['mixture_score'], reverse=True)
        elif self.selection_metric == 'combined':
            for m in self.all_models:
                norm_coh = (m['coherence'] - min(x['coherence'] for x in self.all_models)) / \
                            (max(x['coherence'] for x in self.all_models) - min(x['coherence'] for x in self.all_models) + 1e-10)
                norm_mix = m['mixture_score']
                m['combined_score'] = 0.6 * norm_coh + 0.4 * norm_mix
            self.all_models.sort(key=lambda x: x['combined_score'], reverse=True)
        
        best = self.all_models[0]
        self.lda_model = best['model']
        self.best_random_state = best['random_seed']
        self.best_coherence = best['coherence']
        self.best_perplexity = best['perplexity']
        self.best_mixture_score = best['mixture_score']

    def get_topic_word_distribution(self, num_words: int = 10) -> Dict[int, List[Tuple[str, float]]]:
        if self.lda_model is None:
            raise ValueError("Model has not been trained yet. Call fit() first.")
        topic_word_dist = {}
        for topic_id in range(self.num_topics):
            topic_words = self.lda_model.show_topic(topic_id, topn=num_words)
            topic_word_dist[topic_id] = [(word, float(prob)) for word, prob in topic_words]
        return topic_word_dist

    def get_document_topic_distribution(self, documents: List[str] = None) -> List[List[Tuple[int, float]]]:
        if self.lda_model is None:
            raise ValueError("Model has not been trained yet. Call fit() first.")
        if documents is None:
            corpus = self.corpus
        else:
            processed_docs = self.preprocess_documents(documents)
            corpus = [self.dictionary.doc2bow(doc) for doc in processed_docs]
        doc_topic_dist = []
        for doc_bow in corpus:
            doc_topics = self.lda_model.get_document_topics(doc_bow, minimum_probability=0)
            doc_topic_dist.append([(topic_id, float(prob)) for topic_id, prob in doc_topics])
        return doc_topic_dist

    def get_coherence_score(self) -> float:
        if self.lda_model is None:
            raise ValueError("Model has not been trained yet. Call fit() first.")
        coherence_model = CoherenceModel(
            model=self.lda_model,
            texts=self.processed_docs,
            dictionary=self.dictionary,
            coherence='c_v'
        )
        return coherence_model.get_coherence()

    def get_mixture_score(self) -> float:
        if self.lda_model is None:
            raise ValueError("Model has not been trained yet. Call fit() first.")
        return self._compute_topic_mixture_score(self.lda_model, self.corpus)

    def print_all_initializations(self) -> None:
        if not self.all_models:
            print("No models have been trained.")
            return
        
        print(f"\n所有 {len(self.all_models)} 次初始化结果:")
        print("-" * 80)
        print(f"{'序号':<6} {'随机种子':<12} {'一致性':<12} {'困惑度':<12} {'混合度':<12}")
        print("-" * 80)
        for i, model_info in enumerate(self.all_models):
            marker = " ★" if i == 0 else ""
            print(f"{i+1:<6} {model_info['random_seed']:<12} "
                  f"{model_info['coherence']:<12.4f} "
                  f"{model_info['perplexity']:<12.4f} "
                  f"{model_info['mixture_score']:<12.4f}{marker}")
        print("-" * 80)
        print("★ 表示选中的最优模型")

    def print_topics(self, num_words: int = 10) -> None:
        topic_word_dist = self.get_topic_word_distribution(num_words)
        for topic_id, words in topic_word_dist.items():
            print(f"Topic {topic_id}:")
            for word, prob in words:
                print(f"  {word}: {prob:.4f}")
            print()


def load_documents_from_file(file_path: str) -> List[str]:
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    documents = [doc.strip() for doc in content.split('\n\n') if doc.strip()]
    return documents


def load_documents_from_folder(folder_path: str) -> List[str]:
    if not os.path.exists(folder_path):
        raise FileNotFoundError(f"Folder not found: {folder_path}")
    documents = []
    for filename in os.listdir(folder_path):
        if filename.endswith('.txt'):
            file_path = os.path.join(folder_path, filename)
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read().strip()
                if content:
                    documents.append(content)
    return documents


if __name__ == "__main__":
    sample_documents = [
        "机器学习是人工智能的一个分支，它使计算机能够从数据中学习并做出预测。",
        "深度学习是机器学习的子集，使用多层神经网络来学习复杂的模式。",
        "自然语言处理是计算机科学和语言学的交叉领域，研究如何让计算机理解人类语言。",
        "Python是一种流行的编程语言，广泛用于数据科学和机器学习。",
        "数据分析涉及检查、清理、转换和建模数据，以发现有用的信息。",
        "云计算通过互联网提供计算服务，包括服务器、存储和数据库。",
        "大数据技术处理大量数据，用于揭示隐藏的模式和未知的相关性。",
        "人工智能正在改变各个行业，从医疗保健到金融服务。",
        "神经网络受到人脑神经元连接的启发，用于模式识别任务。",
        "数据可视化使用图表和图形来表示数据，使信息更容易理解。"
    ]

    print("=" * 80)
    print("LDA 主题模型 - 多初始化策略演示")
    print("=" * 80)
    print(f"\n配置: 主题数=3, 初始化次数=5, 选择策略='combined'")
    
    lda = LDAModel(num_topics=3, random_state=None, num_init=5, selection_metric='combined')
    lda.fit(sample_documents)

    lda.print_all_initializations()
    
    print(f"\n最优模型使用的随机种子: {lda.best_random_state}")
    print(f"最优模型 - 一致性分数: {lda.best_coherence:.4f}")
    print(f"最优模型 - 困惑度: {lda.best_perplexity:.4f}")
    print(f"最优模型 - 主题混合度: {lda.best_mixture_score:.4f}")

    print("\n" + "=" * 80)
    print("主题-词分布 (Topic-Word Distribution):")
    print("=" * 80)
    lda.print_topics(num_words=5)

    print("=" * 80)
    print("文档-主题分布 (Document-Topic Distribution):")
    print("=" * 80)
    doc_topic_dist = lda.get_document_topic_distribution()
    for i, doc_topics in enumerate(doc_topic_dist):
        print(f"文档 {i + 1}:")
        for topic_id, prob in doc_topics:
            print(f"  主题 {topic_id}: {prob:.4f}")
        print()

    print("\n" + "=" * 80)
    print("说明:")
    print("=" * 80)
    print("- 随机种子=None: 使用完全随机的初始化，避免局部最优")
    print("- num_init=5: 训练5个不同初始化的模型")
    print("- selection_metric='combined': 综合考虑一致性(60%)和主题混合度(40%)")
    print("- 主题混合度越高，表示文档在主题上的分布越均匀，避免单一主题主导")


class DTMModel:
    def __init__(self, num_topics: int = 5, random_state: Optional[int] = 42,
                 chain_variance: float = 0.005, passes: int = 10):
        self.num_topics = num_topics
        self.random_state = random_state
        self.chain_variance = chain_variance
        self.passes = passes
        self.dictionary = None
        self.dtm_model = None
        self.time_slices = None
        self.time_labels = None
        self.stop_words = self._load_stop_words()

    def _load_stop_words(self) -> set:
        default_stop_words = {
            '的', '了', '在', '是', '我', '有', '和', '就', '不', '人', '都', '一', '一个',
            '上', '也', '很', '到', '说', '要', '去', '你', '会', '着', '没有', '看', '好',
            '自己', '这', '那', '他', '她', '它', '们', '这个', '那个', '什么', '怎么',
            '为什么', '哪', '哪里', '谁', '多少', '几', '啊', '吧', '呢', '吗', '啦',
            'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of',
            'with', 'by', 'as', 'is', 'was', 'are', 'were', 'be', 'been', 'being',
            'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could', 'should',
            'may', 'might', 'must', 'shall', 'can', 'need', 'dare', 'ought', 'used'
        }
        return default_stop_words

    def _clean_text(self, text: str) -> str:
        text = re.sub(r'[^\w\s]', '', text)
        text = re.sub(r'\d+', '', text)
        text = text.lower()
        return text

    def _tokenize(self, text: str) -> List[str]:
        text = self._clean_text(text)
        words = jieba.lcut(text)
        words = [word for word in words if word.strip() and word not in self.stop_words and len(word) > 1]
        return words

    def preprocess_documents(self, documents: List[str]) -> List[List[str]]:
        return [self._tokenize(doc) for doc in documents]

    def fit(self, documents: List[str], time_slice_counts: List[int], 
            time_labels: Optional[List[str]] = None) -> None:
        if sum(time_slice_counts) != len(documents):
            raise ValueError(f"时间片文档数总和({sum(time_slice_counts)})必须等于文档总数({len(documents)})")
        
        processed_docs = self.preprocess_documents(documents)
        self.dictionary = corpora.Dictionary(processed_docs)
        self.dictionary.filter_extremes(no_below=2, no_above=0.9)
        corpus = [self.dictionary.doc2bow(doc) for doc in processed_docs]
        
        self.time_slices = time_slice_counts
        self.time_labels = time_labels if time_labels else [f'T{i}' for i in range(len(time_slice_counts))]
        self.corpus = corpus
        self.processed_docs = processed_docs
        
        self.dtm_model = LdaSeqModel(
            corpus=corpus,
            id2word=self.dictionary,
            time_slice=time_slice_counts,
            num_topics=self.num_topics,
            chain_variance=self.chain_variance,
            passes=self.passes,
            random_state=self.random_state
        )

    def get_topic_words_at_time(self, time_idx: int, topic_idx: int, 
                                num_words: int = 10) -> List[Tuple[str, float]]:
        if self.dtm_model is None:
            raise ValueError("模型未训练，请先调用 fit()")
        if time_idx < 0 or time_idx >= len(self.time_slices):
            raise ValueError(f"时间索引超出范围，有效范围: 0-{len(self.time_slices)-1}")
        if topic_idx < 0 or topic_idx >= self.num_topics:
            raise ValueError(f"主题索引超出范围，有效范围: 0-{self.num_topics-1}")
        
        topic_words = self.dtm_model.print_topic_times(topic=topic_idx, top_terms=num_words)
        return [(word, float(prob)) for word, prob in topic_words[time_idx]]

    def get_topic_evolution(self, topic_idx: int, num_words: int = 5) -> Dict[str, List[Tuple[str, float]]]:
        if self.dtm_model is None:
            raise ValueError("模型未训练，请先调用 fit()")
        
        evolution = {}
        for time_idx, label in enumerate(self.time_labels):
            evolution[label] = self.get_topic_words_at_time(time_idx, topic_idx, num_words)
        return evolution

    def print_topic_evolution(self, topic_idx: int, num_words: int = 5) -> None:
        evolution = self.get_topic_evolution(topic_idx, num_words)
        print(f"\n主题 {topic_idx} 的时间演化:")
        print("-" * 60)
        for time_label, words in evolution.items():
            print(f"{time_label}:")
            for word, prob in words:
                print(f"  {word}: {prob:.4f}")
            print()

    def get_word_evolution_across_time(self, word: str) -> Dict[str, Dict[int, float]]:
        if self.dtm_model is None:
            raise ValueError("模型未训练，请先调用 fit()")
        
        word_id = self.dictionary.token2id.get(word)
        if word_id is None:
            return {}
        
        result = {}
        for time_idx, label in enumerate(self.time_labels):
            time_topics = {}
            for topic_idx in range(self.num_topics):
                topic_dist = self.dtm_model.print_topic_times(topic=topic_idx, top_terms=len(self.dictionary))
                word_prob = dict(topic_dist[time_idx]).get(word, 0.0)
                time_topics[topic_idx] = float(word_prob)
            result[label] = time_topics
        return result

    def get_document_topic_distribution(self, doc_idx: int) -> List[Tuple[int, float]]:
        if self.dtm_model is None:
            raise ValueError("模型未训练，请先调用 fit()")
        if doc_idx < 0 or doc_idx >= len(self.corpus):
            raise ValueError(f"文档索引超出范围")
        
        doc_bow = self.corpus[doc_idx]
        topic_dist = self.dtm_model[doc_bow]
        return [(i, float(prob)) for i, prob in enumerate(topic_dist)]

    def get_topic_popularity_at_time(self, time_idx: int) -> List[Tuple[int, float]]:
        if self.dtm_model is None:
            raise ValueError("模型未训练，请先调用 fit()")
        
        start_idx = sum(self.time_slices[:time_idx])
        end_idx = start_idx + self.time_slices[time_idx]
        
        topic_counts = np.zeros(self.num_topics)
        for doc_idx in range(start_idx, end_idx):
            doc_topics = self.get_document_topic_distribution(doc_idx)
            for topic_idx, prob in doc_topics:
                topic_counts[topic_idx] += prob
        
        topic_counts = topic_counts / self.time_slices[time_idx]
        return [(i, float(topic_counts[i])) for i in range(self.num_topics)]

    def get_topic_popularity_trend(self) -> Dict[int, List[Tuple[str, float]]]:
        if self.dtm_model is None:
            raise ValueError("模型未训练，请先调用 fit()")
        
        trend = {}
        for topic_idx in range(self.num_topics):
            topic_trend = []
            for time_idx, label in enumerate(self.time_labels):
                popularity = dict(self.get_topic_popularity_at_time(time_idx))[topic_idx]
                topic_trend.append((label, popularity))
            trend[topic_idx] = topic_trend
        return trend

    def print_popularity_trend(self) -> None:
        trend = self.get_topic_popularity_trend()
        print("\n主题流行度时间趋势:")
        print("-" * 80)
        
        time_labels_str = "".join([f"{label:<12}" for label in self.time_labels])
        print(f"{'主题':<8}{time_labels_str}")
        print("-" * 80)
        
        for topic_idx, topic_trend in trend.items():
            values = "".join([f"{pop:<12.4f}" for _, pop in topic_trend])
            print(f"{topic_idx:<8}{values}")

    def get_coherence_score(self) -> float:
        if self.dtm_model is None:
            raise ValueError("模型未训练，请先调用 fit()")
        
        topics = []
        for topic_idx in range(self.num_topics):
            topic_words = []
            for time_idx in range(len(self.time_slices)):
                words = self.get_topic_words_at_time(time_idx, topic_idx, num_words=20)
                topic_words.extend([w for w, _ in words])
            topics.append(list(set(topic_words))[:10])
        
        coherence_model = CoherenceModel(
            topics=topics,
            texts=self.processed_docs,
            dictionary=self.dictionary,
            coherence='c_v'
        )
        return coherence_model.get_coherence()


def load_documents_with_time(file_path: str) -> Tuple[List[str], List[int], List[str]]:
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"文件不存在: {file_path}")
    
    documents = []
    time_labels = []
    time_counts = []
    
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    time_blocks = content.split('\n\n')
    for block in time_blocks:
        lines = block.strip().split('\n')
        if not lines:
            continue
        
        time_label = lines[0]
        docs = [line.strip() for line in lines[1:] if line.strip()]
        
        if docs:
            time_labels.append(time_label)
            time_counts.append(len(docs))
            documents.extend(docs)
    
    return documents, time_counts, time_labels
