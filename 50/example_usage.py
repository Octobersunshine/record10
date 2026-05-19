import numpy as np
import libpysal
from sar_model import (
    fit_sar_model, fit_sem_model, print_sar_results, print_sem_results,
    row_standardize, check_row_standardized, spatial_lm_test, print_lm_results,
    auto_select_model, print_model_comparison, generate_sar_data, generate_sem_data
)


def example_with_matrix():
    """
    示例：使用numpy数组作为空间权重矩阵
    """
    print("示例1: 使用numpy数组作为空间权重矩阵 - SAR模型")
    print("-" * 60)
    
    n = 25
    grid_size = int(np.sqrt(n))
    W_array = np.zeros((n, n))
    for i in range(grid_size):
        for j in range(grid_size):
            idx = i * grid_size + j
            if i > 0:
                W_array[idx, (i-1)*grid_size + j] = 1
            if i < grid_size-1:
                W_array[idx, (i+1)*grid_size + j] = 1
            if j > 0:
                W_array[idx, i*grid_size + j-1] = 1
            if j < grid_size-1:
                W_array[idx, i*grid_size + j+1] = 1
    
    W_std = row_standardize(W_array)
    print(f"原始W行和范围: [{W_array.sum(axis=1).min():.2f}, {W_array.sum(axis=1).max():.2f}]")
    print(f"已行标准化: {check_row_standardized(W_std)}")
    
    np.random.seed(123)
    rho_true = 0.4
    beta_true = np.array([1.0, 1.5])
    
    X = np.random.randn(n, 1)
    X = np.hstack([np.ones((n, 1)), X])
    
    y = generate_sar_data(W_std, rho=rho_true, beta=beta_true, X=X, sigma=0.3)
    
    results = fit_sar_model(W_array, y, X)
    print_sar_results(results)
    
    print("\n真实参数:")
    print(f"ρ = {rho_true}")
    print(f"β = {beta_true}")
    print()


def example_lm_test_and_auto_select():
    """
    示例：LM检验和自动模型选择
    """
    print("示例2: LM检验和自动模型选择")
    print("-" * 60)
    
    n = 36
    W = libpysal.weights.lat2W(6, 6)
    W.transform = 'r'
    W_matrix = W.full()[0]
    
    beta = np.array([0.5, 1.0])
    X = np.random.randn(n, 1)
    X = np.hstack([np.ones((n, 1)), X])
    
    print("\n=== 测试SAR数据 ===")
    y_sar = generate_sar_data(W_matrix, rho=0.5, beta=beta, X=X)
    
    lm_results = spatial_lm_test(W, y_sar, X)
    print_lm_results(lm_results)
    
    selected_model, model_results, _ = auto_select_model(W, y_sar, X, verbose=False)
    print(f"\nSAR数据 - 自动选择: {selected_model}")
    
    print("\n=== 测试SEM数据 ===")
    y_sem = generate_sem_data(W_matrix, lam=0.6, beta=beta, X=X)
    
    lm_results2 = spatial_lm_test(W, y_sem, X)
    print_lm_results(lm_results2)
    
    selected_model2, model_results2, _ = auto_select_model(W, y_sem, X, verbose=False)
    print(f"\nSEM数据 - 自动选择: {selected_model2}")


def example_model_comparison():
    """
    示例：模型比较
    """
    print("\n示例3: SAR vs SEM 模型比较")
    print("-" * 60)
    
    n = 49
    W = libpysal.weights.lat2W(7, 7)
    W.transform = 'r'
    W_matrix = W.full()[0]
    
    beta = np.array([1.0, 2.0])
    X = np.random.randn(n, 1)
    X = np.hstack([np.ones((n, 1)), X])
    
    y = generate_sem_data(W_matrix, lam=0.5, beta=beta, X=X)
    
    sar_results = fit_sar_model(W, y, X, verbose=False)
    sem_results = fit_sem_model(W, y, X, verbose=False)
    
    print_model_comparison(sar_results, sem_results)
    
    print("\n真实模型: SEM (λ=0.5)")
    print(f"真实β = {beta}")


def example_with_shapefile():
    """
    示例：使用实际shapefile数据（如果有的话）
    """
    print("示例2: 使用libpysal内置的示例数据")
    print("-" * 60)
    
    try:
        from libpysal.examples import load_example
        import geopandas as gpd
        
        columbus = load_example('Columbus')
        gdf = gpd.read_file(columbus.get_path('columbus.shp'))
        
        W = libpysal.weights.Queen.from_dataframe(gdf)
        
        y = gdf['CRIME'].values.reshape(-1, 1)
        
        X = np.column_stack([
            np.ones(len(gdf)),
            gdf['INC'].values,
            gdf['HOVAL'].values
        ])
        
        results = fit_sar_model(W, y, X)
        print_sar_results(results)
        
    except Exception as e:
        print(f"加载示例数据时出错: {e}")
        print("这是正常的，因为可能没有安装geopandas或没有示例数据")
        print("你可以用自己的数据来替换这个示例")


if __name__ == "__main__":
    example_with_matrix()
    example_with_shapefile()
