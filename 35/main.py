from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import numpy as np
from statsmodels.tsa.stattools import acf, pacf, acorr_ljungbox
from typing import List, Optional

app = FastAPI(title="时间序列 ACF/PACF 计算 API")


class TimeSeriesRequest(BaseModel):
    data: List[float]
    nlags: Optional[int] = None
    alpha: Optional[float] = 0.05


class ACFResult(BaseModel):
    lags: List[int]
    acf_values: List[float]
    confidence_interval_lower: List[float]
    confidence_interval_upper: List[float]


class PACFResult(BaseModel):
    lags: List[int]
    pacf_values: List[float]
    confidence_interval_lower: List[float]
    confidence_interval_upper: List[float]


class LjungBoxResult(BaseModel):
    lags: List[int]
    lb_statistics: List[float]
    p_values: List[float]
    is_white_noise: bool
    significance_level: float


class CorrelationResponse(BaseModel):
    acf: ACFResult
    pacf: PACFResult
    ljung_box: LjungBoxResult


def calculate_default_nlags(nobs: int) -> int:
    """
    计算默认的延迟阶数
    策略：根据序列长度动态调整，确保长周期序列不会丢失信息
    - 短序列 (<30): nobs // 2
    - 中等序列 (30-100): max(nobs // 3, 20)
    - 长序列 (>100): max(nobs // 4, 50)
    最大不超过 nobs - 1（避免过拟合）
    """
    if nobs < 30:
        nlags = nobs // 2
    elif nobs < 100:
        nlags = max(nobs // 3, 20)
    else:
        nlags = max(nobs // 4, 50)
    
    return min(nlags, nobs - 1)


def calculate_ljung_box(data: np.ndarray, nlags: int, alpha: float) -> LjungBoxResult:
    """
    执行 Ljung-Box 白噪声检验
    - 如果 p_value > alpha，则接受原假设：序列是白噪声
    """
    lb_test = acorr_ljungbox(data, lags=nlags, return_df=True)
    
    lb_statistics = lb_test['lb_stat'].tolist()
    p_values = lb_test['lb_pvalue'].tolist()
    
    min_p_value = min(p_values)
    is_white_noise = min_p_value > alpha
    
    return LjungBoxResult(
        lags=list(range(1, nlags + 1)),
        lb_statistics=lb_statistics,
        p_values=p_values,
        is_white_noise=is_white_noise,
        significance_level=alpha
    )


@app.post("/api/correlation", response_model=CorrelationResponse)
async def calculate_correlation(request: TimeSeriesRequest):
    """
    计算时间序列的ACF（自相关函数）、PACF（偏自相关函数）和 Ljung-Box 白噪声检验
    
    - **data**: 时间序列数据列表
    - **nlags**: 延迟阶数（可选，默认为动态计算值，短序列用nobs/2，长序列用更大值）
    - **alpha**: 显著性水平，用于计算置信区间和白噪声检验（可选，默认为0.05，即95%置信区间）
    
    **白噪声检验说明**:
    - Ljung-Box 检验用于判断序列是否为白噪声
    - is_white_noise=True 表示序列是白噪声（p值 > 显著性水平）
    """
    data_array = np.array(request.data)
    
    if len(data_array) < 2:
        raise HTTPException(status_code=400, detail="时间序列数据至少需要2个点")
    
    nobs = len(data_array)
    nlags = request.nlags if request.nlags is not None else calculate_default_nlags(nobs)
    
    if nlags <= 0:
        raise HTTPException(status_code=400, detail="延迟阶数必须大于0")
    
    if nlags >= nobs:
        raise HTTPException(status_code=400, detail="延迟阶数必须小于数据长度")
    
    try:
        acf_values, acf_confint = acf(
            data_array,
            nlags=nlags,
            alpha=request.alpha,
            fft=True
        )
        
        pacf_values, pacf_confint = pacf(
            data_array,
            nlags=nlags,
            alpha=request.alpha,
            method='ywm'
        )
        
        ljung_box_result = calculate_ljung_box(
            data_array,
            nlags=nlags,
            alpha=request.alpha
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"计算失败: {str(e)}")
    
    lags = list(range(nlags + 1))
    
    return CorrelationResponse(
        acf=ACFResult(
            lags=lags,
            acf_values=acf_values.tolist(),
            confidence_interval_lower=acf_confint[:, 0].tolist(),
            confidence_interval_upper=acf_confint[:, 1].tolist()
        ),
        pacf=PACFResult(
            lags=lags,
            pacf_values=pacf_values.tolist(),
            confidence_interval_lower=pacf_confint[:, 0].tolist(),
            confidence_interval_upper=pacf_confint[:, 1].tolist()
        ),
        ljung_box=ljung_box_result
    )


@app.get("/")
async def root():
    return {
        "message": "时间序列 ACF/PACF 计算 API",
        "docs": "/docs",
        "usage": "POST /api/correlation"
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)