import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Optional, Literal
import sympy as sp
import mpmath as mp

app = FastAPI(title="拉普拉斯变换数值反演服务", description="使用自适应Talbot方法进行拉普拉斯变换数值反演")

class LaplaceInversionRequest(BaseModel):
    F_s: str
    t_values: List[float]
    N: Optional[int] = None
    precision: Optional[Literal["double", "high", "extreme"]] = "double"

class LaplaceInversionResponse(BaseModel):
    t_values: List[float]
    f_t_values: List[float]
    N_used: List[int]
    precision_used: List[str]

def get_adaptive_params(t):
    """
    根据t值自适应选择Talbot方法参数
    
    参数:
        t: 时间点
    
    返回:
        (N, c1, c2, c3, c4, shift): 自适应参数
    """
    if t < 0.1:
        N = 64
        c1, c2, c3, c4 = 0.12, 0.5, 0.1, 0.2
        shift = 5.0
    elif t < 1.0:
        N = 48
        c1, c2, c3, c4 = 0.25, 0.55, 0.25, 0.22
        shift = 2.0
    elif t < 10.0:
        N = 32
        c1, c2, c3, c4 = 0.5017, 0.6407, 0.6122, 0.2645
        shift = 0.5
    elif t < 100.0:
        N = 48
        c1, c2, c3, c4 = 0.8, 0.7, 0.8, 0.3
        shift = 0.1
    else:
        N = 64
        c1, c2, c3, c4 = 1.2, 0.75, 1.0, 0.35
        shift = 0.05
    
    return N, c1, c2, c3, c4, shift

def talbot_inversion(F_s_str, t, N=None, use_adaptive=True):
    """
    自适应Talbot方法数值反演拉普拉斯变换
    
    参数:
        F_s_str: 拉普拉斯变换象函数F(s)的字符串表达式
        t: 时间点
        N: 积分点数，None时自动选择
        use_adaptive: 是否使用自适应参数
    
    返回:
        (f(t), N_used): 原函数在t点的近似值和使用的积分点数
    """
    if t <= 0:
        raise ValueError("t必须大于0")
    
    s = sp.symbols('s')
    try:
        F_s = sp.sympify(F_s_str)
        F_lambda = sp.lambdify(s, F_s, modules=['numpy', 'scipy'])
    except Exception as e:
        raise ValueError(f"无法解析表达式: {str(e)}")
    
    if use_adaptive and N is None:
        N, c1, c2, c3, c4, shift = get_adaptive_params(t)
    elif N is None:
        N = 32
        c1, c2, c3, c4 = 0.5017, 0.6407, 0.6122, 0.2645
        shift = 0.5
    else:
        c1, c2, c3, c4 = 0.5017, 0.6407, 0.6122, 0.2645
        shift = 0.5
    
    h = 2 * np.pi / N
    result = 0.0 + 0.0j
    
    for k in range(N):
        theta = -np.pi + (k + 0.5) * h
        
        tan_term = np.tan(c2 * theta)
        z = N / t * (c1 * theta / tan_term - c3 + c4 * theta * 1j)
        s_val = z + shift
        
        sin_term = np.sin(c2 * theta)
        dz_dtheta = N / t * (-c1 * c2 * theta / (sin_term ** 2) + c1 / tan_term + c4 * 1j)
        
        try:
            F_val = F_lambda(s_val)
            if np.isnan(F_val) or np.isinf(F_val):
                continue
            exp_term = np.exp(z * t)
            if np.isnan(exp_term).any() or np.isinf(exp_term).any():
                continue
            result += exp_term * F_val * dz_dtheta
        except Exception as e:
            continue
    
    result = result * h / (2j * np.pi)
    
    return np.real(result), N

def talbot_inversion_enhanced(F_s_str, t, N=None):
    """
    增强版Talbot方法 - 使用双精度验证和Richardson外推
    
    参数:
        F_s_str: 拉普拉斯变换象函数F(s)的字符串表达式
        t: 时间点
        N: 积分点数，None时自动选择
    
    返回:
        (f(t), N_used): 原函数在t点的近似值和使用的积分点数
    """
    if t <= 0:
        raise ValueError("t必须大于0")
    
    if N is None:
        N_base, _, _, _, _, _ = get_adaptive_params(t)
    else:
        N_base = N
    
    f1, _ = talbot_inversion(F_s_str, t, N_base, use_adaptive=False)
    
    if t > 10:
        f2, _ = talbot_inversion(F_s_str, t, N_base * 2, use_adaptive=False)
        f_enhanced = (4 * f2 - f1) / 3
        return f_enhanced, N_base * 2
    
    return f1, N_base

def talbot_inversion_mp(F_s_str, t, prec=50, N=None):
    """
    高精度Talbot方法 - 使用mpmath实现多精度计算
    
    参数:
        F_s_str: 拉普拉斯变换象函数F(s)的字符串表达式
        t: 时间点
        prec: 精度位数，默认50位
        N: 积分点数，None时自动选择
    
    返回:
        (f(t), N_used): 原函数在t点的近似值和使用的积分点数
    """
    if t <= 0:
        raise ValueError("t必须大于0")
    
    mp.mp.dps = prec
    
    t_mp = mp.mpf(t)
    
    if N is None:
        N_base, c1, c2, c3, c4, shift = get_adaptive_params(t)
        if prec >= 100:
            N = N_base * 2
        else:
            N = N_base
    else:
        _, c1, c2, c3, c4, shift = get_adaptive_params(t)
    
    c1_mp = mp.mpf(c1)
    c2_mp = mp.mpf(c2)
    c3_mp = mp.mpf(c3)
    c4_mp = mp.mpf(c4)
    shift_mp = mp.mpf(shift)
    
    s = sp.symbols('s')
    try:
        F_s = sp.sympify(F_s_str)
        F_lambda = sp.lambdify(s, F_s, modules=['mpmath'])
    except Exception as e:
        raise ValueError(f"无法解析表达式: {str(e)}")
    
    h = mp.mpf(2) * mp.pi / N
    result = mp.mpc(0)
    
    for k in range(N):
        theta = -mp.pi + (k + mp.mpf(0.5)) * h
        
        tan_term = mp.tan(c2_mp * theta)
        z = N / t_mp * (c1_mp * theta / tan_term - c3_mp + c4_mp * theta * 1j)
        s_val = z + shift_mp
        
        sin_term = mp.sin(c2_mp * theta)
        dz_dtheta = N / t_mp * (-c1_mp * c2_mp * theta / (sin_term ** 2) + c1_mp / tan_term + c4_mp * 1j)
        
        try:
            F_val = F_lambda(s_val)
            exp_term = mp.e ** (z * t_mp)
            result += exp_term * F_val * dz_dtheta
        except Exception as e:
            continue
    
    result = result * h / (2j * mp.pi)
    
    return float(mp.re(result)), N

def talbot_inversion_extreme(F_s_str, t, N=None):
    """
    极端精度Talbot方法 - 使用100位精度和双外推
    
    参数:
        F_s_str: 拉普拉斯变换象函数F(s)的字符串表达式
        t: 时间点
        N: 积分点数，None时自动选择
    
    返回:
        (f(t), N_used): 原函数在t点的近似值和使用的积分点数
    """
    if N is None:
        N_base, _, _, _, _, _ = get_adaptive_params(t)
    else:
        N_base = N
    
    f1, _ = talbot_inversion_mp(F_s_str, t, prec=100, N=N_base)
    f2, _ = talbot_inversion_mp(F_s_str, t, prec=100, N=N_base * 2)
    f3, _ = talbot_inversion_mp(F_s_str, t, prec=100, N=N_base * 4)
    
    f_extrap = (64 * f3 - 56 * f2 + 9 * f1) / 17
    
    return f_extrap, N_base * 4

@app.post("/invert", response_model=LaplaceInversionResponse, summary="拉普拉斯变换数值反演")
async def invert_laplace(request: LaplaceInversionRequest):
    """
    使用自适应Talbot方法进行拉普拉斯变换数值反演
    
    - **F_s**: 拉普拉斯变换象函数F(s)的字符串表达式
    - **t_values**: 时间点列表，每个值必须大于0
    - **N**: 积分点数（可选，不提供时自动选择）
    - **precision**: 精度级别: double(双精度), high(高精度), extreme(极端精度)
    """
    try:
        f_t_values = []
        N_used_list = []
        precision_used_list = []
        
        for t in request.t_values:
            if t <= 0:
                raise HTTPException(status_code=400, detail=f"时间值t={t}必须大于0")
            
            precision = request.precision or "double"
            
            if precision == "extreme":
                f_t, N_used = talbot_inversion_extreme(request.F_s, t, request.N)
                precision_used = "extreme (100-bit)"
            elif precision == "high":
                f_t, N_used = talbot_inversion_mp(request.F_s, t, prec=50, N=request.N)
                precision_used = "high (50-bit)"
            else:
                f_t, N_used = talbot_inversion_enhanced(request.F_s, t, request.N)
                precision_used = "double (64-bit)"
            
            f_t_values.append(float(f_t))
            N_used_list.append(N_used)
            precision_used_list.append(precision_used)
        
        return LaplaceInversionResponse(
            t_values=request.t_values,
            f_t_values=f_t_values,
            N_used=N_used_list,
            precision_used=precision_used_list
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"计算错误: {str(e)}")

@app.get("/", summary="根路径")
async def root():
    return {
        "message": "拉普拉斯变换数值反演服务",
        "version": "3.0.0",
        "features": [
            "多精度计算支持 - double(64位), high(50位), extreme(100位)",
            "自适应参数选择 - 根据t值自动调整轮廓参数",
            "大t值精度优化 - 专门针对t>10的情况优化",
            "Richardson外推 - 对大t值使用双精度外推提高精度",
            "鲁棒性增强 - 数值稳定性检查"
        ],
        "usage": "使用POST /invert端点进行反演计算，可通过precision参数选择精度级别"
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
