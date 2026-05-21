import numpy as np
from data_loader import KnowledgeGraph, DataLoader, load_sample_data
from transe import TransE
from evaluation import LinkPredictionEvaluator


def train_transe(kg, embedding_dim=50, margin=1.0, learning_rate=0.01, 
                 epochs=100, batch_size=8, negative_ratio=1, verbose=True):
    model = TransE(
        num_entities=kg.num_entities(),
        num_relations=kg.num_relations(),
        embedding_dim=embedding_dim,
        margin=margin,
        learning_rate=learning_rate
    )
    
    data_loader = DataLoader(kg, batch_size=batch_size, negative_ratio=negative_ratio)
    
    for epoch in range(epochs):
        total_loss = 0
        num_batches = 0
        
        for batch_pos, batch_neg in data_loader.generate_batches():
            loss = model.train_step(batch_pos, batch_neg)
            total_loss += loss
            num_batches += 1
        
        avg_loss = total_loss / (num_batches * batch_size) if num_batches > 0 else 0
        
        if verbose and (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch + 1}/{epochs}, Average Loss: {avg_loss:.4f}")
    
    return model


def main():
    print("=" * 60)
    print("TransE 知识图谱嵌入 - 链接预测")
    print("=" * 60)
    
    print("\n1. 加载知识图谱数据...")
    kg = load_sample_data()
    print(f"   实体数量: {kg.num_entities()}")
    print(f"   关系数量: {kg.num_relations()}")
    print(f"   三元组数量: {len(kg.triples)}")
    
    print("\n2. 训练 TransE 模型...")
    model = train_transe(
        kg,
        embedding_dim=20,
        margin=2.0,
        learning_rate=0.05,
        epochs=200,
        batch_size=8,
        negative_ratio=1,
        verbose=True
    )
    
    print("\n3. 模型评估...")
    evaluator = LinkPredictionEvaluator(model, kg)
    test_triples = kg.get_id_triples()
    metrics = evaluator.evaluate(test_triples, filter_triples=True)
    
    print(f"   Mean Rank: {metrics['mean_rank']:.2f}")
    print(f"   Hits@1: {metrics['hits@1']:.4f}")
    print(f"   Hits@3: {metrics['hits@3']:.4f}")
    print(f"   Hits@10: {metrics['hits@10']:.4f}")
    
    print("\n4. 链接预测示例...")
    
    print("\n   预测尾实体: '比尔·盖茨' + '创立' = ?")
    results = evaluator.predict_tail('比尔·盖茨', '创立', top_k=5)
    for i, res in enumerate(results, 1):
        print(f"      {i}. {res['tail']} (分数: {res['score']:.4f})")
    
    print("\n   预测尾实体: '微软' + '总部在' = ?")
    results = evaluator.predict_tail('微软', '总部在', top_k=5)
    for i, res in enumerate(results, 1):
        print(f"      {i}. {res['tail']} (分数: {res['score']:.4f})")
    
    print("\n   预测头实体: ? + '创立' = '微软'")
    results = evaluator.predict_head('微软', '创立', top_k=5)
    for i, res in enumerate(results, 1):
        print(f"      {i}. {res['head']} (分数: {res['score']:.4f})")
    
    print("\n5. 实体和关系嵌入示例...")
    print(f"   '比尔·盖茨' 嵌入向量形状: {model.get_entity_embedding(kg.entity2id['比尔·盖茨']).shape}")
    print(f"   '创立' 关系嵌入向量形状: {model.get_relation_embedding(kg.relation2id['创立']).shape}")
    
    print("\n" + "=" * 60)
    print("完成!")
    print("=" * 60)


if __name__ == "__main__":
    main()
