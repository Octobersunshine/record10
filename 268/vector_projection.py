import numpy as np
import warnings


def vector_projection(a, b):
    a = np.array(a, dtype=float)
    b = np.array(b, dtype=float)
    
    dot_product = np.dot(a, b)
    b_norm_squared = np.dot(b, b)
    
    if b_norm_squared == 0:
        warnings.warn("Vector b is the zero vector, returning proj as zero vector and orth as a itself.")
        proj = np.zeros_like(a)
        orth = a.copy()
        return proj, orth
    
    proj = (dot_product / b_norm_squared) * b
    orth = a - proj
    
    return proj, orth


def gram_schmidt(vectors, modified=True, normalize=True):
    vectors = [np.array(v, dtype=float) for v in vectors]
    n = len(vectors)
    if n == 0:
        return []
    
    dim = vectors[0].shape[0]
    Q = []
    
    if modified:
        V = [v.copy() for v in vectors]
        for i in range(n):
            vi_norm = np.linalg.norm(V[i])
            if vi_norm < 1e-10:
                warnings.warn(f"Vector {i} is linearly dependent, skipping.")
                Q.append(np.zeros(dim))
                continue
            
            qi = V[i] / vi_norm
            Q.append(qi)
            
            for j in range(i + 1, n):
                V[j] = V[j] - np.dot(V[j], qi) * qi
    else:
        for i in range(n):
            vi = vectors[i].copy()
            
            for j in range(i):
                vi = vi - np.dot(vectors[i], Q[j]) * Q[j]
            
            vi_norm = np.linalg.norm(vi)
            if vi_norm < 1e-10:
                warnings.warn(f"Vector {i} is linearly dependent, skipping.")
                Q.append(np.zeros(dim))
                continue
            
            if normalize:
                Q.append(vi / vi_norm)
            else:
                Q.append(vi)
    
    return Q


if __name__ == "__main__":
    print("=== 测试用例 1 ===")
    a = [3, 4]
    b = [1, 0]
    
    proj, orth = vector_projection(a, b)
    
    print(f"向量 a: {a}")
    print(f"向量 b: {b}")
    print(f"投影向量 proj: {proj}")
    print(f"正交向量 orth: {orth}")
    print(f"验证 a = proj + orth: {np.allclose(a, proj + orth)}")
    
    print("\n=== 测试用例 2（3维向量） ===")
    a = [1, 2, 3]
    b = [4, 5, 6]
    
    proj, orth = vector_projection(a, b)
    
    print(f"向量 a: {a}")
    print(f"向量 b: {b}")
    print(f"投影向量 proj: {proj}")
    print(f"正交向量 orth: {orth}")
    print(f"验证 a = proj + orth: {np.allclose(a, proj + orth)}")
    
    print("\n=== 测试用例 3（任意方向） ===")
    a = [2, 3]
    b = [1, 1]
    
    proj, orth = vector_projection(a, b)
    
    print(f"向量 a: {a}")
    print(f"向量 b: {b}")
    print(f"投影向量 proj: {proj}")
    print(f"正交向量 orth: {orth}")
    print(f"验证 a = proj + orth: {np.allclose(a, proj + orth)}")
    print(f"验证 proj 与 orth 正交: {np.isclose(np.dot(proj, orth), 0)}")
    
    print("\n=== 测试用例 4（b为零向量） ===")
    a = [1, 2, 3]
    b = [0, 0, 0]
    
    proj, orth = vector_projection(a, b)
    
    print(f"向量 a: {a}")
    print(f"向量 b: {b}")
    print(f"投影向量 proj: {proj}")
    print(f"正交向量 orth: {orth}")
    print(f"验证 a = proj + orth: {np.allclose(a, proj + orth)}")
    print(f"验证 proj 为零向量: {np.allclose(proj, 0)}")
    print(f"验证 orth == a: {np.allclose(orth, a)}")
    
    print("\n=== 测试用例 5（Gram-Schmidt正交化 - MGS） ===")
    vectors = [[1, 1, 0], [1, 0, 1], [0, 1, 1]]
    
    Q = gram_schmidt(vectors, modified=True, normalize=True)
    
    print(f"原始向量:")
    for i, v in enumerate(vectors):
        print(f"  v{i}: {v}")
    print(f"正交基 (MGS):")
    for i, q in enumerate(Q):
        print(f"  q{i}: {q}")
    
    print(f"验证正交性:")
    for i in range(len(Q)):
        for j in range(i + 1, len(Q)):
            dot = np.dot(Q[i], Q[j])
            print(f"  q{i}·q{j} = {dot:.10f} (≈0: {np.isclose(dot, 0)})")
    print(f"验证单位长度:")
    for i, q in enumerate(Q):
        norm = np.linalg.norm(q)
        print(f"  ||q{i}|| = {norm:.10f} (≈1: {np.isclose(norm, 1) or np.isclose(norm, 0)})")
    
    print("\n=== 测试用例 6（Gram-Schmidt正交化 - 标准版本） ===")
    Q_standard = gram_schmidt(vectors, modified=False, normalize=True)
    
    print(f"正交基 (标准):")
    for i, q in enumerate(Q_standard):
        print(f"  q{i}: {q}")
    
    print("\n=== 测试用例 7（线性相关向量） ===")
    dependent_vectors = [[1, 2, 3], [2, 4, 6], [1, 0, 0]]
    
    Q_dep = gram_schmidt(dependent_vectors, modified=True, normalize=True)
    
    print(f"原始向量:")
    for i, v in enumerate(dependent_vectors):
        print(f"  v{i}: {v}")
    print(f"正交基:")
    for i, q in enumerate(Q_dep):
        print(f"  q{i}: {q}")
