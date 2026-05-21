#!/usr/bin/env python
# -*- coding: utf-8 -*-

from lda_model import LDAModel, DTMModel, load_documents_from_file, load_documents_from_folder, load_documents_with_time


def example_dtm_basic():
    print("\n\n示例5: 动态主题模型(DTM) - 建模主题随时间的演化")
    print("=" * 80)

    documents_by_time = {
        "2018-2020": [
            "机器学习在工业界开始大规模应用。",
            "深度学习在图像识别领域取得突破。",
            "支持向量机是传统机器学习的重要方法。",
            "决策树和随机森林广泛用于分类任务。",
            "回归分析用于预测连续数值。"
        ],
        "2020-2022": [
            "自然语言处理技术快速发展。",
            "BERT等预训练模型改变了NLP格局。",
            "Transformer架构成为主流。",
            "注意力机制在各领域得到应用。",
            "语言模型生成能力大幅提升。"
        ],
        "2022-2024": [
            "大语言模型如GPT引起广泛关注。",
            "多模态学习融合文本图像视频。",
            "生成式AI在各行业得到应用。",
            "提示工程成为新的研究热点。",
            "模型压缩和推理优化技术发展迅速。"
        ],
        "2024-至今": [
            "Agent和AI代理系统成为研究热点。",
            "多智能体协作解决复杂问题。",
            "检索增强生成(RAG)提升大模型能力。",
            "AI安全和对齐研究受到重视。",
            "人工智能与各行各业深度融合。"
        ]
    }

    all_documents = []
    time_slice_counts = []
    time_labels = []
    
    for time_label, docs in documents_by_time.items():
        all_documents.extend(docs)
        time_slice_counts.append(len(docs))
        time_labels.append(time_label)

    print(f"\n数据集: {len(time_labels)}个时间片，共{len(all_documents)}篇文档")
    for label, count in zip(time_labels, time_slice_counts):
        print(f"  {label}: {count}篇")

    dtm = DTMModel(num_topics=3, random_state=42, chain_variance=0.005, passes=15)
    dtm.fit(all_documents, time_slice_counts, time_labels)
    
    print(f"\n模型一致性分数: {dtm.get_coherence_score():.4f}")

    for topic_idx in range(3):
        dtm.print_topic_evolution(topic_idx, num_words=4)

    print("\n" + "=" * 80)
    dtm.print_popularity_trend()
    print("=" * 80)


def example_dtm_word_evolution():
    print("\n\n示例6: 追踪特定词汇在不同时间片的主题分布变化")
    print("=" * 80)

    documents_by_time = {
        "T1": [
            "人工智能和机器学习快速发展。",
            "神经网络在各领域应用。",
            "数据驱动的方法受到关注。"
        ],
        "T2": [
            "深度学习引领人工智能进步。",
            "自然语言处理取得突破。",
            "大模型改变研究范式。"
        ],
        "T3": [
            "大语言模型和生成式AI兴起。",
            "多模态技术融合发展。",
            "AI应用落地各行各业。"
        ]
    }

    all_documents = []
    time_slice_counts = []
    time_labels = []
    
    for time_label, docs in documents_by_time.items():
        all_documents.extend(docs)
        time_slice_counts.append(len(docs))
        time_labels.append(time_label)

    dtm = DTMModel(num_topics=2, random_state=42)
    dtm.fit(all_documents, time_slice_counts, time_labels)

    target_word = "人工智能"
    print(f"\n追踪词汇: '{target_word}'")
    word_evolution = dtm.get_word_evolution_across_time(target_word)
    
    print("\n该词在各时间片的主题概率分布:")
    print("-" * 50)
    for time_label, topic_probs in word_evolution.items():
        probs_str = ", ".join([f"主题{t}: {p:.4f}" for t, p in topic_probs.items()])
        print(f"{time_label}: {probs_str}")


def example_dtm_document_topic():
    print("\n\n示例7: 文档的时间主题分布分析")
    print("=" * 80)

    documents_by_time = {
        "早期": [
            "机器学习基础理论研究。",
            "统计学习方法探讨。"
        ],
        "中期": [
            "深度学习技术革新。",
            "卷积神经网络应用。"
        ],
        "近期": [
            "大语言模型研究进展。",
            "生成式AI的未来展望。"
        ]
    }

    all_documents = []
    time_slice_counts = []
    time_labels = []
    
    for time_label, docs in documents_by_time.items():
        all_documents.extend(docs)
        time_slice_counts.append(len(docs))
        time_labels.append(time_label)

    dtm = DTMModel(num_topics=2, random_state=42)
    dtm.fit(all_documents, time_slice_counts, time_labels)

    print("\n各文档的主题分布:")
    print("-" * 60)
    
    doc_idx = 0
    for time_idx, time_label in enumerate(time_labels):
        for i in range(time_slice_counts[time_idx]):
            topic_dist = dtm.get_document_topic_distribution(doc_idx)
            dist_str = ", ".join([f"主题{t}: {p:.4f}" for t, p in topic_dist])
            print(f"[{time_label}] 文档{i+1}: {dist_str}")
            doc_idx += 1


def example_multi_initialization():
    print("示例1: 多初始化策略 - 避免局部最优，提高主题混合度")
    print("-" * 70)

    documents = [
        "机器学习是人工智能的一个重要分支，它使计算机能够从数据中学习。",
        "深度学习使用神经网络来处理复杂的数据模式。",
        "自然语言处理让计算机能够理解和生成人类语言。",
        "Python编程语言在数据科学领域非常流行。",
        "数据挖掘技术用于从大量数据中发现有价值的信息。",
        "统计学是数据分析的基础学科之一。"
    ]

    print("\n方案A: 传统单一固定种子初始化")
    lda_fixed = LDAModel(num_topics=2, random_state=42, num_init=1, selection_metric='coherence')
    lda_fixed.fit(documents)
    print(f"  一致性分数: {lda_fixed.best_coherence:.4f}")
    print(f"  主题混合度: {lda_fixed.best_mixture_score:.4f}")

    print("\n方案B: 多随机初始化 + 综合选择策略（推荐）")
    lda_multi = LDAModel(num_topics=2, random_state=None, num_init=10, selection_metric='combined')
    lda_multi.fit(documents)
    lda_multi.print_all_initializations()
    
    print(f"\n最优模型 - 一致性分数: {lda_multi.best_coherence:.4f}")
    print(f"最优模型 - 主题混合度: {lda_multi.best_mixture_score:.4f}")
    print(f"最优模型 - 使用种子: {lda_multi.best_random_state}")


def example_selection_strategies():
    print("\n\n示例2: 不同模型选择策略对比")
    print("-" * 70)

    documents = [
        "机器学习和深度学习是人工智能的核心技术。",
        "神经网络在图像识别方面表现出色。",
        "数据科学需要掌握统计学和编程技能。",
        "Python是数据科学家最常用的编程语言。",
        "云计算提供了强大的计算资源。",
        "大数据技术处理海量数据。"
    ]

    strategies = ['coherence', 'perplexity', 'mixture', 'combined']
    
    for strategy in strategies:
        lda = LDAModel(num_topics=3, random_state=None, num_init=5, selection_metric=strategy)
        lda.fit(documents)
        print(f"\n策略: {strategy}")
        print(f"  一致性: {lda.best_coherence:.4f}, 混合度: {lda.best_mixture_score:.4f}, "
              f"困惑度: {lda.best_perplexity:.4f}")


def example_reproducible_best_model():
    print("\n\n示例3: 使用最优种子复现最佳结果")
    print("-" * 70)

    documents = [
        "机器学习是人工智能的分支。",
        "深度学习使用神经网络。",
        "自然语言处理处理人类语言。",
        "Python是流行的编程语言。",
        "数据分析从数据中提取信息。"
    ]

    print("第一步: 多随机初始化，寻找最优模型")
    lda_search = LDAModel(num_topics=2, random_state=None, num_init=8, selection_metric='combined')
    lda_search.fit(documents)
    best_seed = lda_search.best_random_state
    print(f"找到的最优种子: {best_seed}")

    print("\n第二步: 使用最优种子重新训练，获得完全相同的结果")
    lda_reproduce = LDAModel(num_topics=2, random_state=best_seed, num_init=1, selection_metric='coherence')
    lda_reproduce.fit(documents)
    
    print(f"复现模型 - 一致性: {lda_reproduce.best_coherence:.4f}")
    print(f"复现模型 - 混合度: {lda_reproduce.best_mixture_score:.4f}")
    print("\n这样既避免了局部最优，又保证了结果的可复现性！")


def example_topic_mixture_analysis():
    print("\n\n示例4: 主题混合度详细分析")
    print("-" * 70)

    documents = [
        "机器学习和深度学习都属于人工智能领域。",
        "神经网络可以用于图像识别和自然语言处理。",
        "Python和R都是数据科学常用的编程语言。",
        "数据分析和数据挖掘有很多相似的技术。",
        "云计算和大数据是相辅相成的技术。"
    ]

    lda = LDAModel(num_topics=3, random_state=None, num_init=5, selection_metric='combined')
    lda.fit(documents)

    doc_topic_dist = lda.get_document_topic_distribution()
    
    print("\n各文档的主题分布（高混合度意味着文档在多个主题上均匀分布）:")
    for i, doc_topics in enumerate(doc_topic_dist):
        probs = [prob for _, prob in doc_topics]
        entropy = -sum(p * np.log(p + 1e-10) for p in probs)
        print(f"文档 {i+1}: 分布={[f'{p:.2f}' for p in probs]}, 熵={entropy:.4f}")


if __name__ == "__main__":
    import numpy as np
    
    example_multi_initialization()
    example_selection_strategies()
    example_reproducible_best_model()
    example_topic_mixture_analysis()
    
    example_dtm_basic()
    example_dtm_word_evolution()
    example_dtm_document_topic()

    print("\n\n" + "=" * 80)
    print("DTM动态主题模型功能总结")
    print("=" * 80)
    print("1. 主题演化分析: 查看每个主题在不同时间片的词汇分布变化")
    print("2. 词汇追踪: 追踪特定词汇在不同时间片各主题中的概率变化")
    print("3. 主题流行度趋势: 分析主题在不同时间段的流行程度变化")
    print("4. 文档-主题分布: 查看每篇文档在各时间点的主题归属")
    print("5. 模型评估: 计算主题一致性分数评估模型质量")
    print("\n适用场景: 科学文献的主题变迁、新闻热点的演化追踪、")
    print("         研究领域的发展趋势分析等时间序列文本数据")
    print("=" * 80)
