import numpy as np
import pandas as pd
from typing import Tuple, Optional
import os


def load_from_text(file_path: str, delimiter: str = None,
                   skiprows: int = 0,
                   wavelength_col: int = 0,
                   intensity_col: int = 1) -> Tuple[np.ndarray, np.ndarray]:
    """
    从文本文件加载光谱数据
    
    Args:
        file_path: 文件路径
        delimiter: 分隔符，None表示自动检测
        skiprows: 跳过的行数
        wavelength_col: 波长列索引
        intensity_col: 强度列索引
    
    Returns:
        (wavelengths, intensities)
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")
    
    if delimiter is None:
        with open(file_path, 'r') as f:
            for _ in range(skiprows):
                f.readline()
            line = f.readline()
            if '\t' in line:
                delimiter = '\t'
            elif ',' in line:
                delimiter = ','
            else:
                delimiter = None
    
    data = np.loadtxt(file_path, delimiter=delimiter, skiprows=skiprows)
    
    if data.ndim == 1:
        wavelengths = np.arange(len(data))
        intensities = data
    else:
        wavelengths = data[:, wavelength_col]
        intensities = data[:, intensity_col]
    
    return wavelengths, intensities


def load_from_csv(file_path: str,
                  wavelength_col: str = None,
                  intensity_col: str = None,
                  **kwargs) -> Tuple[np.ndarray, np.ndarray]:
    """
    从CSV文件加载光谱数据
    
    Args:
        file_path: 文件路径
        wavelength_col: 波长列名
        intensity_col: 强度列名
        **kwargs: 传递给pd.read_csv的参数
    
    Returns:
        (wavelengths, intensities)
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")
    
    df = pd.read_csv(file_path, **kwargs)
    
    if wavelength_col is None and intensity_col is None:
        if df.shape[1] == 1:
            wavelengths = np.arange(len(df))
            intensities = df.iloc[:, 0].values
        else:
            wavelengths = df.iloc[:, 0].values
            intensities = df.iloc[:, 1].values
    else:
        wavelengths = df[wavelength_col].values
        intensities = df[intensity_col].values
    
    return wavelengths, intensities


def load_spectrum(file_path: str, **kwargs) -> Tuple[np.ndarray, np.ndarray]:
    """
    通用光谱加载函数，根据文件扩展名自动选择加载方法
    
    Args:
        file_path: 文件路径
        **kwargs: 传递给具体加载函数的参数
    
    Returns:
        (wavelengths, intensities)
    """
    ext = os.path.splitext(file_path)[1].lower()
    
    if ext in ['.txt', '.dat', '.spa', '.spc']:
        return load_from_text(file_path, **kwargs)
    elif ext in ['.csv']:
        return load_from_csv(file_path, **kwargs)
    elif ext in ['.npy']:
        data = np.load(file_path)
        if data.ndim == 2 and data.shape[1] >= 2:
            return data[:, 0], data[:, 1]
        else:
            return np.arange(len(data)), data
    else:
        try:
            return load_from_text(file_path, **kwargs)
        except Exception as e:
            raise ValueError(f"Unsupported file format: {ext}. Error: {e}")


def save_spectrum(file_path: str, wavelengths: np.ndarray,
                  intensities: np.ndarray, **kwargs):
    """
    保存光谱数据到文件
    
    Args:
        file_path: 文件路径
        wavelengths: 波长数组
        intensities: 强度数组
        **kwargs: 传递给保存函数的参数
    """
    ext = os.path.splitext(file_path)[1].lower()
    
    if ext in ['.txt', '.dat']:
        np.savetxt(file_path, np.column_stack((wavelengths, intensities)), **kwargs)
    elif ext in ['.csv']:
        df = pd.DataFrame({'wavelength': wavelengths, 'intensity': intensities})
        df.to_csv(file_path, index=False, **kwargs)
    elif ext in ['.npy']:
        np.save(file_path, np.column_stack((wavelengths, intensities)))
    else:
        np.savetxt(file_path, np.column_stack((wavelengths, intensities)), **kwargs)
