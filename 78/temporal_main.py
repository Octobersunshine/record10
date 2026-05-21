import numpy as np
from temporal_data_loader import TemporalKnowledgeGraph, TemporalDataLoader, load_temporal_sample_data
from transe_t import TransET
from temporal_evaluation import TemporalLinkPredictionEvaluator


def train_transe_t(tkg, embedding_dim=50, margin=1.0, learning_rate=0.01, 
                   epochs=100, batch_size=8, negative_ratio=1, verbose=True):
    model = TransET(
        num_entities=tkg.num_entities(),
        num_relations=tkg.num_relations(),
        num_timestamps=tkg.num_timestamps(),
        embedding_dim=embedding_dim,
        margin=margin,
        learning_rate=learning_rate
    )
    
    data_loader = TemporalDataLoader(tkg, batch_size=batch_size, negative_ratio=negative_ratio)
    
    for epoch in range(epochs):
        total_loss = 0
        num_batches = 0
        
        for batch_pos, batch_neg in data_loader.generate_batches():
            loss = model.train_step(batch_pos, batch_neg)
            total_loss += loss
            num_batches += 1
        
        avg_loss = total_loss / (num_batches * batch_size) if num_batches > 0 else 0
        
        if verbose and (epoch + 1) % 20 == 0:
            print(f"Epoch {epoch + 1}/{epochs}, Average Loss: {avg_loss:.4f}")
    
    return model


def main():
    print("=" * 60)
    print("TransE-T 时序知识图谱嵌入 - 时间感知链接预测")
    print("=" * 60)
    
    print("\n1. 加载时序知识图谱数据...")
    tkg = load_temporal_sample_data()
    print(f"   实体数量: {tkg.num_entities()}")
    print(f"   关系数量: {tkg.num_relations()}")
    print(f"   时间戳数量: {tkg.num_timestamps()}")
    print(f"   四元组数量: {len(tkg.quadruples)}")
    
    print("\n2. 训练 TransE-T 模型...")
    model = train_transe_t(
        tkg,
        embedding_dim=20,
        margin=2.0,
        learning_rate=0.05,
        epochs=200,
        batch_size=8,
        negative_ratio=2,
        verbose=True
    )
    
    print("\n3. 模型评估...")
    evaluator = TemporalLinkPredictionEvaluator(model, tkg)
    test_quadruples = tkg.get_id_quadruples()
    metrics = evaluator.evaluate(test_quadruples, filter_quadruples=True)
    
    print(f"   Mean Rank: {metrics['mean_rank']:.2f}")
    print(f"   Hits@1: {metrics['hits@1']:.4f}")
    print(f"   Hits@3: {metrics['hits@3']:.4f}")
    print(f"   Hits@10: {metrics['hits@10']:.4f}")
    
    print("\n4. 时序链接预测示例...")
    
    print("\n   A. 预测指定时间的尾实体:")
    print("      1990年, 微软的CEO是谁?")
    results = evaluator.predict_head('微软', '担任CEO', 1990, top_k=3)
    for i, res in enumerate(results, 1):
        print(f"         {i}. {res['head']} (分数: {res['score']:.4f})")
    
    print("\n      2020年, 微软的CEO是谁?")
    results = evaluator.predict_head('微软', '担任CEO', 2020, top_k=3)
    for i, res in enumerate(results, 1):
        print(f"         {i}. {res['head']} (分数: {res['score']:.4f})")
    
    print("\n      2020年, 苹果的CEO是谁?")
    results = evaluator.predict_head('苹果', '担任CEO', 2020, top_k=3)
    for i, res in enumerate(results, 1):
        print(f"         {i}. {res['head']} (分数: {res['score']:.4f})")
    
    print("\n   B. 预测事实发生的时间:")
    print("      比尔·盖茨 什么时候 担任CEO 微软?")
    results = evaluator.predict_time('比尔·盖茨', '担任CEO', '微软', top_k=5)
    for i, res in enumerate(results, 1):
        print(f"         {i}. {res['timestamp']}年 (分数: {res['score']:.4f})")
    
    print("\n      萨提亚·纳德拉 什么时候 担任CEO 微软?")
    results = evaluator.predict_time('萨提亚·纳德拉', '担任CEO', '微软', top_k=5)
    for i, res in enumerate(results, 1):
        print(f"         {i}. {res['timestamp']}年 (分数: {res['score']:.4f})")
    
    print("\n      史蒂夫·乔布斯 什么时候 担任CEO 苹果?")
    results = evaluator.predict_time('史蒂夫·乔布斯', '担任CEO', '苹果', top_k=5)
    for i, res in enumerate(results, 1):
        print(f"         {i}. {res['timestamp']}年 (分数: {res['score']:.4f})")
    
    print("\n5. 嵌入向量示例...")
    print(f"   实体嵌入形状: {model.get_entity_embedding(0).shape}")
    print(f"   关系嵌入形状: {model.get_relation_embedding(0).shape}")
    print(f"   时间戳嵌入形状: {model.get_timestamp_embedding(0).shape}")
    
    print("\n6. 时间戳连续性分析...")
    print("   相邻年份的时间嵌入相似度:")
    timestamps = sorted(tkg.timestamp2id.keys())
    for i in range(len(timestamps) - 1):
        ts1, ts2 = timestamps[i], timestamps[i + 1]
        emb1 = model.get_timestamp_embedding(tkg.timestamp2id[ts1])
        emb2 = model.get_timestamp_embedding(tkg.timestamp2id[ts2])
        similarity = np.dot(emb1, emb2) / (np.linalg.norm(emb1) * np.linalg.norm(emb2))
        print(f"      {ts1} - {ts2}: 相似度 = {similarity:.4f}")
    
    print("\n" + "=" * 60)
    print("TransE-T 时序知识图谱嵌入完成!")
    print("=" * 60)


if __name__ == "__main__":
    main()
