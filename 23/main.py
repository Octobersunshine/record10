from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import accuracy_score, r2_score
import numpy as np
from typing import List, Union, Optional

app = FastAPI(title="随机森林特征重要性API", description="接收特征矩阵X和目标y，训练随机森林模型，返回基于基尼不纯度的特征重要性排序")


class TrainingData(BaseModel):
    X: List[List[float]]
    y: List[Union[float, int]]
    n_estimators: int = 100
    random_state: int = 42
    task_type: str = "classification"
    perform_significance_test: bool = True
    n_permutations: int = 100
    alpha: float = 0.05


class FeatureImportance(BaseModel):
    feature_index: int
    importance: float
    rank: int
    p_value: Optional[float] = None
    is_significant: Optional[bool] = None


class TrainingResponse(BaseModel):
    success: bool
    message: str
    feature_importances: List[FeatureImportance]
    significant_features: List[FeatureImportance]
    model_type: str
    total_features: int
    significant_count: int
    significance_test_performed: bool


def permutation_importance_test(model, X, y, n_permutations=100, task_type="classification", random_state=42):
    np.random.seed(random_state)
    n_features = X.shape[1]
    p_values = np.ones(n_features)
    
    y_pred = model.predict(X)
    if task_type == "classification":
        baseline_score = accuracy_score(y, y_pred)
    else:
        baseline_score = r2_score(y, y_pred)
    
    for feature_idx in range(n_features):
        permuted_scores = []
        for _ in range(n_permutations):
            X_permuted = X.copy()
            np.random.shuffle(X_permuted[:, feature_idx])
            y_pred_permuted = model.predict(X_permuted)
            
            if task_type == "classification":
                score = accuracy_score(y, y_pred_permuted)
            else:
                score = r2_score(y, y_pred_permuted)
            permuted_scores.append(score)
        
        permuted_scores = np.array(permuted_scores)
        p_value = np.mean(permuted_scores >= baseline_score)
        p_values[feature_idx] = max(p_value, 1.0 / n_permutations)
    
    return p_values


@app.post("/train", response_model=TrainingResponse)
async def train_random_forest(data: TrainingData):
    try:
        X = np.array(data.X)
        y = np.array(data.y)

        if X.ndim != 2:
            raise HTTPException(status_code=400, detail="特征矩阵X必须是二维数组")
        
        if y.ndim != 1:
            raise HTTPException(status_code=400, detail="目标y必须是一维数组")
        
        if X.shape[0] != y.shape[0]:
            raise HTTPException(status_code=400, detail=f"样本数不匹配: X有{X.shape[0]}个样本，y有{y.shape[0]}个样本")
        
        if data.n_permutations < 10:
            raise HTTPException(status_code=400, detail="置换次数n_permutations必须大于等于10")
        
        if data.alpha <= 0 or data.alpha >= 1:
            raise HTTPException(status_code=400, detail="显著性水平alpha必须在(0, 1)之间")

        if data.task_type == "classification":
            model = RandomForestClassifier(
                n_estimators=data.n_estimators,
                random_state=data.random_state,
                criterion="gini"
            )
            model_type = "RandomForestClassifier (Gini)"
        elif data.task_type == "regression":
            model = RandomForestRegressor(
                n_estimators=data.n_estimators,
                random_state=data.random_state,
                criterion="squared_error"
            )
            model_type = "RandomForestRegressor"
        else:
            raise HTTPException(status_code=400, detail="task_type必须是'classification'或'regression'")

        model.fit(X, y)

        importances = model.feature_importances_
        importances = importances / importances.sum()
        indices = np.argsort(importances)[::-1]

        p_values = None
        if data.perform_significance_test:
            p_values = permutation_importance_test(
                model, X, y, 
                n_permutations=data.n_permutations,
                task_type=data.task_type,
                random_state=data.random_state
            )

        feature_importances = []
        significant_features = []
        
        for rank, idx in enumerate(indices, 1):
            p_val = float(p_values[idx]) if p_values is not None else None
            is_sig = (p_val < data.alpha) if p_val is not None else None
            
            feat_imp = FeatureImportance(
                feature_index=int(idx),
                importance=float(importances[idx]),
                rank=rank,
                p_value=p_val,
                is_significant=is_sig
            )
            feature_importances.append(feat_imp)
            
            if is_sig:
                significant_features.append(feat_imp)

        message = "模型训练完成，特征重要性已计算"
        if data.perform_significance_test:
            message += f"，显著性检验完成（{len(significant_features)}个显著特征，α={data.alpha}）"

        return TrainingResponse(
            success=True,
            message=message,
            feature_importances=feature_importances,
            significant_features=significant_features,
            model_type=model_type,
            total_features=X.shape[1],
            significant_count=len(significant_features),
            significance_test_performed=data.perform_significance_test
        )

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"训练过程中发生错误: {str(e)}")


@app.get("/")
async def root():
    return {
        "message": "随机森林特征重要性API",
        "endpoints": {
            "/": "API信息",
            "/train": "POST - 训练模型并获取特征重要性",
            "/docs": "Swagger文档",
            "/redoc": "ReDoc文档"
        }
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
