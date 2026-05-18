import numpy as np
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression
from sklearn.cross_decomposition import PLSRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error


class PCARegression:
    """
    PCA降维 + 线性回归模型类
    
    确保训练和预测时数据标准化处理完全一致
    """
    
    def __init__(self, n_components=None):
        self.n_components = n_components
        self.scaler = StandardScaler()
        self.pca = PCA(n_components=n_components)
        self.regressor = LinearRegression()
        self._is_fitted = False
    
    def fit(self, X, y):
        """
        训练模型：标准化 -> PCA降维 -> 线性回归
        
        参数:
            X: 特征矩阵，形状为 (n_samples, n_features)
            y: 目标变量，形状为 (n_samples,)
        
        返回:
            self: 训练好的模型
        """
        X_scaled = self.scaler.fit_transform(X)
        X_pca = self.pca.fit_transform(X_scaled)
        self.regressor.fit(X_pca, y)
        self._is_fitted = True
        return self
    
    def predict(self, X):
        """
        预测：使用训练时相同的标准化和PCA参数
        
        参数:
            X: 特征矩阵，形状为 (n_samples, n_features)
        
        返回:
            y_pred: 预测值，形状为 (n_samples,)
        """
        if not self._is_fitted:
            raise ValueError("模型未训练，请先调用fit()方法")
        
        X_scaled = self.scaler.transform(X)
        X_pca = self.pca.transform(X_scaled)
        return self.regressor.predict(X_pca)
    
    def score(self, X, y):
        """
        计算模型R²得分
        
        参数:
            X: 特征矩阵
            y: 真实目标值
        
        返回:
            score: R²得分
        """
        y_pred = self.predict(X)
        return self.regressor.score(self.pca.transform(self.scaler.transform(X)), y)
    
    @property
    def components_(self):
        """获取PCA主成分"""
        return self.pca.components_
    
    @property
    def explained_variance_ratio_(self):
        """获取解释方差比例"""
        return self.pca.explained_variance_ratio_
    
    @property
    def mean_(self):
        """获取标准化的均值"""
        return self.scaler.mean_
    
    @property
    def scale_(self):
        """获取标准化的标准差"""
        return self.scaler.scale_
    
    @property
    def coef_(self):
        """获取线性回归系数（在PCA空间中）"""
        return self.regressor.coef_
    
    @property
    def intercept_(self):
        """获取线性回归截距"""
        return self.regressor.intercept_


def pca_linear_regression(X, y, n_components=None):
    """
    先对X做PCA降维，再用降维后的主成分做线性回归
    
    确保训练和预测时数据标准化处理完全一致
    
    参数:
        X: 特征矩阵，形状为 (n_samples, n_features)
        y: 目标变量，形状为 (n_samples,)
        n_components: PCA降维后的主成分数量，默认为None（保留所有主成分）
    
    返回:
        model: PCARegression模型对象
    """
    model = PCARegression(n_components=n_components)
    model.fit(X, y)
    return model


def get_pca_components(model):
    """
    从模型中获取PCA的主成分
    
    参数:
        model: pca_linear_regression返回的模型
    
    返回:
        components: 主成分矩阵
    """
    return model.components_


def get_explained_variance_ratio(model):
    """
    从模型中获取每个主成分的解释方差比例
    
    参数:
        model: pca_linear_regression返回的模型
    
    返回:
        explained_variance_ratio: 解释方差比例数组
    """
    return model.explained_variance_ratio_


def verify_standardization_consistency(model, X_train, X_test):
    """
    验证训练和预测时标准化处理的一致性
    
    参数:
        model: 训练好的模型
        X_train: 训练数据
        X_test: 测试数据
    
    返回:
        dict: 包含验证结果的字典
    """
    scaler = model.scaler
    
    X_train_scaled = scaler.transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    manual_train_scaled = (X_train - scaler.mean_) / scaler.scale_
    manual_test_scaled = (X_test - scaler.mean_) / scaler.scale_
    
    train_consistent = np.allclose(X_train_scaled, manual_train_scaled)
    test_consistent = np.allclose(X_test_scaled, manual_test_scaled)
    
    return {
        'train_consistent': train_consistent,
        'test_consistent': test_consistent,
        'mean_used': scaler.mean_,
        'scale_used': scaler.scale_
    }


class PLSRegressionModel:
    """
    偏最小二乘回归 (PLS) 模型类
    
    确保训练和预测时数据标准化处理完全一致
    """
    
    def __init__(self, n_components=None):
        self.n_components = n_components
        self.scaler = StandardScaler()
        self.pls = PLSRegression(n_components=n_components)
        self._is_fitted = False
    
    def fit(self, X, y):
        """
        训练PLS模型：标准化 -> PLS回归
        
        参数:
            X: 特征矩阵，形状为 (n_samples, n_features)
            y: 目标变量，形状为 (n_samples,)
        
        返回:
            self: 训练好的模型
        """
        X_scaled = self.scaler.fit_transform(X)
        self.pls.fit(X_scaled, y)
        self._is_fitted = True
        return self
    
    def predict(self, X):
        """
        预测：使用训练时相同的标准化参数
        
        参数:
            X: 特征矩阵，形状为 (n_samples, n_features)
        
        返回:
            y_pred: 预测值，形状为 (n_samples,)
        """
        if not self._is_fitted:
            raise ValueError("模型未训练，请先调用fit()方法")
        
        X_scaled = self.scaler.transform(X)
        return self.pls.predict(X_scaled).flatten()
    
    def score(self, X, y):
        """
        计算模型R²得分
        
        参数:
            X: 特征矩阵
            y: 真实目标值
        
        返回:
            score: R²得分
        """
        y_pred = self.predict(X)
        return self.pls.score(self.scaler.transform(X), y)
    
    @property
    def x_weights_(self):
        """获取X的权重"""
        return self.pls.x_weights_
    
    @property
    def x_loadings_(self):
        """获取X的载荷"""
        return self.pls.x_loadings_
    
    @property
    def y_loadings_(self):
        """获取y的载荷"""
        return self.pls.y_loadings_
    
    @property
    def coef_(self):
        """获取回归系数（在原始特征空间中）"""
        return self.pls.coef_
    
    @property
    def intercept_(self):
        """获取截距"""
        return self.pls.intercept_
    
    @property
    def mean_(self):
        """获取标准化的均值"""
        return self.scaler.mean_
    
    @property
    def scale_(self):
        """获取标准化的标准差"""
        return self.scaler.scale_


def pls_regression(X, y, n_components=None):
    """
    偏最小二乘回归 (PLS)
    
    确保训练和预测时数据标准化处理完全一致
    
    参数:
        X: 特征矩阵，形状为 (n_samples, n_features)
        y: 目标变量，形状为 (n_samples,)
        n_components: PLS成分数量，默认为None（使用min(n_samples, n_features)）
    
    返回:
        model: PLSRegressionModel模型对象
    """
    model = PLSRegressionModel(n_components=n_components)
    model.fit(X, y)
    return model


def auto_select_best_model(X, y, n_components=None, val_size=0.2, metric='mse', random_state=42):
    """
    自动选择PCR（PCA+线性回归）和PLS中预测误差更小的模型
    
    参数:
        X: 特征矩阵，形状为 (n_samples, n_features)
        y: 目标变量，形状为 (n_samples,)
        n_components: 主成分数量，默认为None
        val_size: 验证集比例，用于比较模型性能
        metric: 评估指标，'mse' 或 'rmse'
        random_state: 随机种子
    
    返回:
        dict: 包含最佳模型和比较结果的字典
            - 'best_model': 最佳模型对象
            - 'best_method': 最佳方法名称 ('PCR' 或 'PLS')
            - 'pcr_score': PCR模型在验证集上的得分
            - 'pls_score': PLS模型在验证集上的得分
            - 'pcr_train_score': PCR模型在训练集上的得分
            - 'pls_train_score': PLS模型在训练集上的得分
    """
    np.random.seed(random_state)
    n_samples = X.shape[0]
    indices = np.random.permutation(n_samples)
    split_idx = int(n_samples * (1 - val_size))
    train_idx, val_idx = indices[:split_idx], indices[split_idx:]
    
    X_train, X_val = X[train_idx], X[val_idx]
    y_train, y_val = y[train_idx], y[val_idx]
    
    if n_components is None:
        n_components = min(X_train.shape[0], X_train.shape[1])
    
    pcr_model = PCARegression(n_components=n_components)
    pcr_model.fit(X_train, y_train)
    
    pls_model = PLSRegressionModel(n_components=n_components)
    pls_model.fit(X_train, y_train)
    
    y_val_pred_pcr = pcr_model.predict(X_val)
    y_val_pred_pls = pls_model.predict(X_val)
    
    if metric == 'mse':
        pcr_score = mean_squared_error(y_val, y_val_pred_pcr)
        pls_score = mean_squared_error(y_val, y_val_pred_pls)
    elif metric == 'rmse':
        pcr_score = np.sqrt(mean_squared_error(y_val, y_val_pred_pcr))
        pls_score = np.sqrt(mean_squared_error(y_val, y_val_pred_pls))
    else:
        raise ValueError("metric必须是'mse'或'rmse'")
    
    y_train_pred_pcr = pcr_model.predict(X_train)
    y_train_pred_pls = pls_model.predict(X_train)
    
    if metric == 'mse':
        pcr_train_score = mean_squared_error(y_train, y_train_pred_pcr)
        pls_train_score = mean_squared_error(y_train, y_train_pred_pls)
    else:
        pcr_train_score = np.sqrt(mean_squared_error(y_train, y_train_pred_pcr))
        pls_train_score = np.sqrt(mean_squared_error(y_train, y_train_pred_pls))
    
    if pcr_score < pls_score:
        best_model = pcr_model
        best_method = 'PCR'
    elif pls_score < pcr_score:
        best_model = pls_model
        best_method = 'PLS'
    else:
        best_model = pcr_model
        best_method = 'PCR'
    
    pcr_model_final = PCARegression(n_components=n_components)
    pcr_model_final.fit(X, y)
    
    pls_model_final = PLSRegressionModel(n_components=n_components)
    pls_model_final.fit(X, y)
    
    if best_method == 'PCR':
        best_model = pcr_model_final
    else:
        best_model = pls_model_final
    
    return {
        'best_model': best_model,
        'best_method': best_method,
        'pcr_score': pcr_score,
        'pls_score': pls_score,
        'pcr_train_score': pcr_train_score,
        'pls_train_score': pls_train_score,
        'metric': metric,
        'n_components': n_components
    }
