import numpy as np
from data_loader import KnowledgeGraph, DataLoader, load_sample_data


def test_negative_sampling():
    print("=" * 60)
    print("测试负采样正确性")
    print("=" * 60)
    
    kg = load_sample_data()
    print(f"\n实体数量: {kg.num_entities()}")
    print(f"关系数量: {kg.num_relations()}")
    print(f"三元组数量: {len(kg.triples)}")
    
    data_loader = DataLoader(kg, batch_size=8, negative_ratio=5)
    triple_set = data_loader.triple_set
    
    print("\n验证负采样是否产生假阴性...")
    
    total_negatives = 0
    false_negatives = 0
    
    for triple in data_loader.id_triples:
        negatives = data_loader.generate_negative_triples(triple)
        for neg in negatives:
            total_negatives += 1
            if neg in triple_set:
                false_negatives += 1
                h, r, t = neg
                print(f"  发现假阴性! ({kg.id2entity[h]}, {kg.id2relation[r]}, {kg.id2entity[t]})")
    
    print(f"\n总计生成负样本: {total_negatives}")
    print(f"假阴性数量: {false_negatives}")
    
    if false_negatives == 0:
        print("✓ 负采样正确，没有假阴性!")
    else:
        print("✗ 负采样有问题，存在假阴性!")
    
    print("\n" + "=" * 60)
    print("测试完成!")
    print("=" * 60)


if __name__ == "__main__":
    test_negative_sampling()
