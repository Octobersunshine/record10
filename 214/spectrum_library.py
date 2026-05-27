import numpy as np
import os
import json
from typing import List, Dict, Optional, Tuple
from spectrum_loader import load_spectrum, save_spectrum
from preprocessing import preprocess_pipeline
from peak_detection import analyze_peaks, create_peak_vector
from utils import align_spectra, resample_spectrum


class SpectrumLibrary:
    """光谱库管理类"""
    
    def __init__(self):
        self.names: List[str] = []
        self.wavelengths: List[np.ndarray] = []
        self.intensities: List[np.ndarray] = []
        self.processed_intensities: List[np.ndarray] = []
        self.peak_analyses: List[Dict] = []
        self.peak_vectors: List[np.ndarray] = []
        self.metadata: List[Dict] = []
        self.common_wavelengths: Optional[np.ndarray] = None
        self.num_points: int = 1000
        self.peak_vector_bins: int = 200
    
    def add_spectrum(self, name: str, wavelengths: np.ndarray,
                     intensities: np.ndarray,
                     preprocess: bool = True,
                     preprocess_kwargs: dict = None,
                     peak_kwargs: dict = None,
                     metadata: dict = None):
        """
        添加光谱到库中
        
        Args:
            name: 物质名称
            wavelengths: 波长数组
            intensities: 强度数组
            preprocess: 是否预处理
            preprocess_kwargs: 预处理参数
            peak_kwargs: 峰检测参数
            metadata: 元数据
        """
        if preprocess_kwargs is None:
            preprocess_kwargs = {}
        if peak_kwargs is None:
            peak_kwargs = {}
        if metadata is None:
            metadata = {}
        
        wl, inten = resample_spectrum(wavelengths, intensities, self.num_points)
        
        processed_inten = inten.copy()
        if preprocess:
            _, processed_inten, _ = preprocess_pipeline(
                wl, inten, **preprocess_kwargs
            )
        
        peak_analysis = analyze_peaks(wl, processed_inten, **peak_kwargs)
        
        self.names.append(name)
        self.wavelengths.append(wl)
        self.intensities.append(inten)
        self.processed_intensities.append(processed_inten)
        self.peak_analyses.append(peak_analysis)
        self.metadata.append(metadata)
        
        self._update_common_wavelengths()
        self._update_peak_vectors()
    
    def add_from_file(self, name: str, file_path: str,
                      preprocess: bool = True,
                      preprocess_kwargs: dict = None,
                      peak_kwargs: dict = None,
                      metadata: dict = None,
                      **load_kwargs):
        """
        从文件添加光谱到库中
        
        Args:
            name: 物质名称
            file_path: 文件路径
            preprocess: 是否预处理
            preprocess_kwargs: 预处理参数
            peak_kwargs: 峰检测参数
            metadata: 元数据
            **load_kwargs: 传递给load_spectrum的参数
        """
        wl, inten = load_spectrum(file_path, **load_kwargs)
        self.add_spectrum(name, wl, inten, preprocess,
                          preprocess_kwargs, peak_kwargs, metadata)
    
    def _update_common_wavelengths(self):
        """更新公共波长轴"""
        if len(self.wavelengths) == 0:
            self.common_wavelengths = None
            return
        
        min_wl = max(wl.min() for wl in self.wavelengths)
        max_wl = min(wl.max() for wl in self.wavelengths)
        self.common_wavelengths = np.linspace(min_wl, max_wl, self.num_points)
        
        _, aligned = align_spectra(
            self.wavelengths, self.processed_intensities, self.num_points
        )
        self.processed_intensities = aligned
    
    def _update_peak_vectors(self):
        """更新峰向量"""
        if self.common_wavelengths is None:
            return
        
        wl_range = (self.common_wavelengths.min(), self.common_wavelengths.max())
        
        self.peak_vectors = []
        for i in range(len(self.names)):
            peak_vector = create_peak_vector(
                self.common_wavelengths,
                self.processed_intensities[i],
                self.peak_analyses[i],
                num_bins=self.peak_vector_bins,
                wl_range=wl_range
            )
            self.peak_vectors.append(peak_vector)
    
    def remove_spectrum(self, name: str):
        """从库中移除光谱"""
        if name in self.names:
            idx = self.names.index(name)
            self.names.pop(idx)
            self.wavelengths.pop(idx)
            self.intensities.pop(idx)
            self.processed_intensities.pop(idx)
            self.peak_analyses.pop(idx)
            self.peak_vectors.pop(idx)
            self.metadata.pop(idx)
            
            self._update_common_wavelengths()
            self._update_peak_vectors()
    
    def get_spectrum(self, name: str) -> Optional[Tuple[np.ndarray, np.ndarray]]:
        """获取光谱数据"""
        if name in self.names:
            idx = self.names.index(name)
            return self.wavelengths[idx], self.intensities[idx]
        return None
    
    def get_processed_spectrum(self, name: str) -> Optional[Tuple[np.ndarray, np.ndarray]]:
        """获取预处理后的光谱数据"""
        if name in self.names:
            idx = self.names.index(name)
            return self.common_wavelengths, self.processed_intensities[idx]
        return None
    
    def get_peak_analysis(self, name: str) -> Optional[Dict]:
        """获取谱峰分析结果"""
        if name in self.names:
            idx = self.names.index(name)
            return self.peak_analyses[idx]
        return None
    
    def list_materials(self) -> List[str]:
        """列出库中所有物质"""
        return self.names.copy()
    
    def save_library(self, directory: str):
        """
        保存光谱库到目录
        
        Args:
            directory: 目录路径
        """
        os.makedirs(directory, exist_ok=True)
        
        manifest = {
            'names': self.names,
            'num_points': self.num_points,
            'peak_vector_bins': self.peak_vector_bins,
            'metadata': self.metadata
        }
        
        with open(os.path.join(directory, 'manifest.json'), 'w') as f:
            json.dump(manifest, f, indent=2)
        
        for i, name in enumerate(self.names):
            data = np.column_stack((
                self.wavelengths[i],
                self.intensities[i],
                self.processed_intensities[i]
            ))
            np.save(os.path.join(directory, f'{name}_data.npy'), data)
            
            np.save(os.path.join(directory, f'{name}_peak_vector.npy'),
                    self.peak_vectors[i])
            
            peak_info = {
                'positions': self.peak_analyses[i]['positions'].tolist(),
                'heights': self.peak_analyses[i]['heights'].tolist(),
                'widths': self.peak_analyses[i]['widths'].tolist(),
                'prominences': self.peak_analyses[i]['prominences'].tolist(),
                'areas': self.peak_analyses[i]['areas'].tolist()
            }
            with open(os.path.join(directory, f'{name}_peaks.json'), 'w') as f:
                json.dump(peak_info, f, indent=2)
    
    def load_library(self, directory: str):
        """
        从目录加载光谱库
        
        Args:
            directory: 目录路径
        """
        manifest_path = os.path.join(directory, 'manifest.json')
        if not os.path.exists(manifest_path):
            raise FileNotFoundError(f"Manifest not found: {manifest_path}")
        
        with open(manifest_path, 'r') as f:
            manifest = json.load(f)
        
        self.__init__()
        self.num_points = manifest.get('num_points', 1000)
        self.peak_vector_bins = manifest.get('peak_vector_bins', 200)
        self.names = manifest.get('names', [])
        self.metadata = manifest.get('metadata', [{}] * len(self.names))
        
        for name in self.names:
            data_path = os.path.join(directory, f'{name}_data.npy')
            data = np.load(data_path)
            
            self.wavelengths.append(data[:, 0])
            self.intensities.append(data[:, 1])
            self.processed_intensities.append(data[:, 2])
            
            peak_vector_path = os.path.join(directory, f'{name}_peak_vector.npy')
            self.peak_vectors.append(np.load(peak_vector_path))
            
            peaks_path = os.path.join(directory, f'{name}_peaks.json')
            with open(peaks_path, 'r') as f:
                peak_info = json.load(f)
            
            peak_analysis = {
                'positions': np.array(peak_info['positions']),
                'heights': np.array(peak_info['heights']),
                'widths': np.array(peak_info['widths']),
                'prominences': np.array(peak_info['prominences']),
                'areas': np.array(peak_info['areas']),
                'peaks': np.array([]),
                'properties': {}
            }
            self.peak_analyses.append(peak_analysis)
        
        self._update_common_wavelengths()
    
    def __len__(self) -> int:
        return len(self.names)
    
    def __contains__(self, name: str) -> bool:
        return name in self.names
