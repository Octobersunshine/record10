import numpy as np
from dataclasses import dataclass
from typing import List, Tuple, Optional, Dict
import warnings


@dataclass
class PolarityChron:
    """
    极性时数据类
    
    Attributes
    ----------
    name : str
        极性时名称 (如 'C1n', 'C1r', 'C2n' 等)
    start_age : float
        开始年龄 (Ma)
    end_age : float
        结束年龄 (Ma)
    polarity : int
        极性: 1=正向, -1=反向
    """
    name: str
    start_age: float
    end_age: float
    polarity: int
    
    @property
    def duration(self) -> float:
        """极性时持续时间 (Myr)"""
        return abs(self.end_age - self.start_age)
    
    @property
    def mid_age(self) -> float:
        """极性时中点年龄"""
        return (self.start_age + self.end_age) / 2


class GeomagneticPolarityTimeScale:
    """
    地磁极性时间标尺
    
    基于 Cande & Kent (1995, CK95) 时间标尺
    支持 0-150 Ma 的极性时数据
    
    参考: Cande, S.C. & Kent, D.V. (1995) JGR, 100(B3), 6093-6095
    """
    
    CK95_CHRONS = [
        ("C1n", 0.0, 0.781, 1),
        ("C1r", 0.781, 0.99, -1),
        ("C1r.1n", 0.99, 1.07, 1),
        ("C1r.1r", 1.07, 1.201, -1),
        ("C2n", 1.201, 1.778, 1),
        ("C2r.1n", 1.778, 1.95, 1),
        ("C2r.1r", 1.95, 2.14, -1),
        ("C2r.2n", 2.14, 2.28, 1),
        ("C2r.2r", 2.28, 2.581, -1),
        ("C2An.1n", 2.581, 3.04, 1),
        ("C2An.1r", 3.04, 3.11, -1),
        ("C2An.2n", 3.11, 3.22, 1),
        ("C2An.2r", 3.22, 3.33, -1),
        ("C2An.3n", 3.33, 3.58, 1),
        ("C2An.3r", 3.58, 3.70, -1),
        ("C3n.1n", 3.70, 4.00, 1),
        ("C3n.1r", 4.00, 4.18, -1),
        ("C3n.2n", 4.18, 4.29, 1),
        ("C3n.2r", 4.29, 4.49, -1),
        ("C3n.3n", 4.49, 4.63, 1),
        ("C3n.3r", 4.63, 4.79, -1),
        ("C3n.4n", 4.79, 4.90, 1),
        ("C3n.4r", 4.90, 5.07, -1),
        ("C3n.5n", 5.07, 5.23, 1),
        ("C3n.5r", 5.23, 5.38, -1),
        ("C3r.1n", 5.38, 5.54, 1),
        ("C3r.1r", 5.54, 5.73, -1),
        ("C3r.2n", 5.73, 5.89, 1),
        ("C3r.2r", 5.89, 6.14, -1),
        ("C4n.1n", 6.14, 6.27, 1),
        ("C4n.1r", 6.27, 6.39, -1),
        ("C4n.2n", 6.39, 6.57, 1),
        ("C4n.2r", 6.57, 6.64, -1),
        ("C4n.3n", 6.64, 6.93, 1),
        ("C4n.3r", 6.93, 7.09, -1),
        ("C4r.1n", 7.09, 7.17, 1),
        ("C4r.1r", 7.17, 7.34, -1),
        ("C4r.2n", 7.34, 7.43, 1),
        ("C4r.2r", 7.43, 7.56, -1),
        ("C5n.1n", 7.56, 7.65, 1),
        ("C5n.1r", 7.65, 7.75, -1),
        ("C5n.2n", 7.75, 7.92, 1),
        ("C5n.2r", 7.92, 8.07, -1),
        ("C5n.3n", 8.07, 8.23, 1),
        ("C5n.3r", 8.23, 8.42, -1),
        ("C5r.1n", 8.42, 8.57, 1),
        ("C5r.1r", 8.57, 8.70, -1),
        ("C5r.2n", 8.70, 8.90, 1),
        ("C5r.2r", 8.90, 9.09, -1),
        ("C5r.3n", 9.09, 9.30, 1),
        ("C5r.3r", 9.30, 9.64, -1),
        ("C5An.1n", 9.64, 9.74, 1),
        ("C5An.1r", 9.74, 9.83, -1),
        ("C5An.2n", 9.83, 10.04, 1),
        ("C5An.2r", 10.04, 10.17, -1),
        ("C5An.3n", 10.17, 10.53, 1),
        ("C5An.3r", 10.53, 10.95, -1),
        ("C5AAn.1n", 10.95, 11.05, 1),
        ("C5AAn.1r", 11.05, 11.15, -1),
        ("C5AAn.2n", 11.15, 11.35, 1),
        ("C5AAn.2r", 11.35, 11.47, -1),
        ("C5AAn.3n", 11.47, 11.53, 1),
        ("C5AAn.3r", 11.53, 11.63, -1),
        ("C5AAr.1n", 11.63, 11.74, 1),
        ("C5AAr.1r", 11.74, 11.86, -1),
        ("C5AAr.2n", 11.86, 11.94, 1),
        ("C5AAr.2r", 11.94, 12.05, -1),
        ("C5ABn.1n", 12.05, 12.13, 1),
        ("C5ABn.1r", 12.13, 12.19, -1),
        ("C5ABn.2n", 12.19, 12.40, 1),
        ("C5ABn.2r", 12.40, 12.68, -1),
        ("C5ABr.1n", 12.68, 12.77, 1),
        ("C5ABr.1r", 12.77, 12.88, -1),
        ("C5ABr.2n", 12.88, 13.05, 1),
        ("C5ABr.2r", 13.05, 13.14, -1),
        ("C5ACn.1n", 13.14, 13.30, 1),
        ("C5ACn.1r", 13.30, 13.41, -1),
        ("C5ACn.2n", 13.41, 13.52, 1),
        ("C5ACn.2r", 13.52, 13.70, -1),
        ("C5ACr.1n", 13.70, 13.80, 1),
        ("C5ACr.1r", 13.80, 13.97, -1),
        ("C5ACr.2n", 13.97, 14.08, 1),
        ("C5ACr.2r", 14.08, 14.18, -1),
        ("C5ADn", 14.18, 14.30, 1),
        ("C5ADr", 14.30, 14.61, -1),
        ("C5AEn.1n", 14.61, 14.68, 1),
        ("C5AEn.1r", 14.68, 14.79, -1),
        ("C5AEn.2n", 14.79, 14.90, 1),
        ("C5AEn.2r", 14.90, 15.10, -1),
        ("C5AEr.1n", 15.10, 15.20, 1),
        ("C5AEr.1r", 15.20, 15.36, -1),
        ("C5AEr.2n", 15.36, 15.51, 1),
        ("C5AEr.2r", 15.51, 15.69, -1),
        ("C5Bn.1n", 15.69, 15.76, 1),
        ("C5Bn.1r", 15.76, 15.90, -1),
        ("C5Bn.2n", 15.90, 16.01, 1),
        ("C5Bn.2r", 16.01, 16.29, -1),
        ("C5Br.1n", 16.29, 16.39, 1),
        ("C5Br.1r", 16.39, 16.58, -1),
        ("C5Br.2n", 16.58, 16.72, 1),
        ("C5Br.2r", 16.72, 16.92, -1),
        ("C5Cn.1n", 16.92, 17.04, 1),
        ("C5Cn.1r", 17.04, 17.28, -1),
        ("C5Cn.2n", 17.28, 17.43, 1),
        ("C5Cn.2r", 17.43, 17.53, -1),
        ("C5Cn.3n", 17.53, 17.73, 1),
        ("C5Cr.1n", 17.73, 17.80, 1),
        ("C5Cr.1r", 17.80, 18.06, -1),
        ("C5Cr.2n", 18.06, 18.20, 1),
        ("C5Cr.2r", 18.20, 18.38, -1),
        ("C5Cr.3n", 18.38, 18.53, 1),
        ("C5Cr.3r", 18.53, 18.75, -1),
        ("C6n.1n", 18.75, 19.05, 1),
        ("C6n.1r", 19.05, 19.25, -1),
        ("C6n.2n", 19.25, 19.35, 1),
        ("C6n.2r", 19.35, 19.50, -1),
        ("C6n.3n", 19.50, 19.72, 1),
        ("C6n.3r", 19.72, 19.85, -1),
        ("C6r.1n", 19.85, 20.04, 1),
        ("C6r.1r", 20.04, 20.25, -1),
        ("C6r.2n", 20.25, 20.44, 1),
        ("C6r.2r", 20.44, 20.56, -1),
        ("C6r.3n", 20.56, 20.73, 1),
        ("C6r.3r", 20.73, 20.95, -1),
        ("C7n.1n", 20.95, 21.32, 1),
        ("C7n.1r", 21.32, 21.45, -1),
        ("C7n.2n", 21.45, 21.76, 1),
        ("C7n.2r", 21.76, 21.86, -1),
        ("C7n.3n", 21.86, 22.15, 1),
        ("C7r.1n", 22.15, 22.25, 1),
        ("C7r.1r", 22.25, 22.46, -1),
        ("C7r.2n", 22.46, 22.56, 1),
        ("C7r.2r", 22.56, 22.75, -1),
        ("C7r.3n", 22.75, 22.90, 1),
        ("C7r.3r", 22.90, 23.03, -1),
        ("C8n.1n", 23.03, 23.36, 1),
        ("C8n.1r", 23.36, 23.53, -1),
        ("C8n.2n", 23.53, 23.68, 1),
        ("C8n.2r", 23.68, 23.87, -1),
        ("C8r.1n", 23.87, 24.01, 1),
        ("C8r.1r", 24.01, 24.11, -1),
        ("C8r.2n", 24.11, 24.25, 1),
        ("C8r.2r", 24.25, 24.63, -1),
        ("C9n.1n", 24.63, 24.82, 1),
        ("C9n.1r", 24.82, 25.04, -1),
        ("C9n.2n", 25.04, 25.18, 1),
        ("C9n.2r", 25.18, 25.41, -1),
        ("C9n.3n", 25.41, 25.65, 1),
        ("C9n.3r", 25.65, 25.82, -1),
        ("C9r.1n", 25.82, 26.03, 1),
        ("C9r.1r", 26.03, 26.28, -1),
        ("C9r.2n", 26.28, 26.55, 1),
        ("C9r.2r", 26.55, 26.82, -1),
        ("C10n.1n", 26.82, 27.03, 1),
        ("C10n.1r", 27.03, 27.15, -1),
        ("C10n.2n", 27.15, 27.37, 1),
        ("C10n.2r", 27.37, 27.57, -1),
        ("C10r.1n", 27.57, 27.77, 1),
        ("C10r.1r", 27.77, 27.97, -1),
        ("C10r.2n", 27.97, 28.12, 1),
        ("C10r.2r", 28.12, 28.28, -1),
        ("C11n.1n", 28.28, 28.51, 1),
        ("C11n.1r", 28.51, 28.75, -1),
        ("C11n.2n", 28.75, 28.92, 1),
        ("C11n.2r", 28.92, 29.15, -1),
        ("C11n.3n", 29.15, 29.35, 1),
        ("C11n.3r", 29.35, 29.46, -1),
        ("C11r.1n", 29.46, 29.66, 1),
        ("C11r.1r", 29.66, 29.86, -1),
        ("C11r.2n", 29.86, 30.01, 1),
        ("C11r.2r", 30.01, 30.20, -1),
        ("C12n.1n", 30.20, 30.38, 1),
        ("C12n.1r", 30.38, 30.48, -1),
        ("C12n.2n", 30.48, 30.67, 1),
        ("C12n.2r", 30.67, 30.90, -1),
        ("C12r.1n", 30.90, 31.04, 1),
        ("C12r.1r", 31.04, 31.20, -1),
        ("C12r.2n", 31.20, 31.34, 1),
        ("C12r.2r", 31.34, 31.48, -1),
        ("C13n.1n", 31.48, 31.65, 1),
        ("C13n.1r", 31.65, 31.83, -1),
        ("C13n.2n", 31.83, 31.99, 1),
        ("C13n.2r", 31.99, 32.16, -1),
        ("C13n.3n", 32.16, 32.33, 1),
        ("C13r.1n", 32.33, 32.48, 1),
        ("C13r.1r", 32.48, 32.63, -1),
        ("C13r.2n", 32.63, 32.75, 1),
        ("C13r.2r", 32.75, 32.93, -1),
        ("C14n.1n", 32.93, 33.07, 1),
        ("C14n.1r", 33.07, 33.24, -1),
        ("C14n.2n", 33.24, 33.38, 1),
        ("C14n.2r", 33.38, 33.54, -1),
        ("C15n.1n", 33.54, 33.71, 1),
        ("C15n.1r", 33.71, 33.86, -1),
        ("C15n.2n", 33.86, 34.05, 1),
        ("C15n.2r", 34.05, 34.25, -1),
        ("C16n.1n", 34.25, 34.42, 1),
        ("C16n.1r", 34.42, 34.62, -1),
        ("C16n.2n", 34.62, 34.76, 1),
        ("C16n.2r", 34.76, 34.94, -1),
        ("C16r.1n", 34.94, 35.08, 1),
        ("C16r.1r", 35.08, 35.26, -1),
        ("C16r.2n", 35.26, 35.42, 1),
        ("C16r.2r", 35.42, 35.67, -1),
        ("C17n.1n", 35.67, 35.83, 1),
        ("C17n.1r", 35.83, 36.00, -1),
        ("C17n.2n", 36.00, 36.18, 1),
        ("C17n.2r", 36.18, 36.32, -1),
        ("C17n.3n", 36.32, 36.48, 1),
        ("C17n.3r", 36.48, 36.62, -1),
        ("C18n.1n", 36.62, 36.79, 1),
        ("C18n.1r", 36.79, 36.95, -1),
        ("C18n.2n", 36.95, 37.10, 1),
        ("C18n.2r", 37.10, 37.27, -1),
        ("C18n.3n", 37.27, 37.40, 1),
        ("C18n.3r", 37.40, 37.60, -1),
        ("C19n", 37.60, 37.84, 1),
        ("C19r", 37.84, 38.00, -1),
        ("C20n", 38.00, 38.24, 1),
        ("C20r", 38.24, 38.43, -1),
        ("C21n.1n", 38.43, 38.60, 1),
        ("C21n.1r", 38.60, 38.75, -1),
        ("C21n.2n", 38.75, 38.95, 1),
        ("C21n.2r", 38.95, 39.14, -1),
        ("C21r.1n", 39.14, 39.30, 1),
        ("C21r.1r", 39.30, 39.47, -1),
        ("C21r.2n", 39.47, 39.63, 1),
        ("C21r.2r", 39.63, 39.82, -1),
        ("C22n", 39.82, 40.13, 1),
        ("C22r.1n", 40.13, 40.25, 1),
        ("C22r.1r", 40.25, 40.43, -1),
        ("C22r.2n", 40.43, 40.52, 1),
        ("C22r.2r", 40.52, 40.77, -1),
        ("C23n.1n", 40.77, 41.02, 1),
        ("C23n.1r", 41.02, 41.22, -1),
        ("C23n.2n", 41.22, 41.42, 1),
        ("C23n.2r", 41.42, 41.55, -1),
        ("C23n.3n", 41.55, 41.77, 1),
        ("C23r.1n", 41.77, 41.94, 1),
        ("C23r.1r", 41.94, 42.16, -1),
        ("C23r.2n", 42.16, 42.33, 1),
        ("C23r.2r", 42.33, 42.54, -1),
        ("C24n.1n", 42.54, 42.77, 1),
        ("C24n.1r", 42.77, 42.97, -1),
        ("C24n.2n", 42.97, 43.14, 1),
        ("C24n.2r", 43.14, 43.37, -1),
        ("C24n.3n", 43.37, 43.56, 1),
        ("C24r.1n", 43.56, 43.64, 1),
        ("C24r.1r", 43.64, 43.80, -1),
        ("C24r.2n", 43.80, 43.97, 1),
        ("C24r.2r", 43.97, 44.15, -1),
        ("C25n.1n", 44.15, 44.34, 1),
        ("C25n.1r", 44.34, 44.50, -1),
        ("C25n.2n", 44.50, 44.66, 1),
        ("C25n.2r", 44.66, 44.83, -1),
        ("C25n.3n", 44.83, 45.01, 1),
        ("C25n.3r", 45.01, 45.15, -1),
        ("C26n.1n", 45.15, 45.35, 1),
        ("C26n.1r", 45.35, 45.52, -1),
        ("C26n.2n", 45.52, 45.68, 1),
        ("C26n.2r", 45.68, 45.84, -1),
        ("C26r.1n", 45.84, 46.02, 1),
        ("C26r.1r", 46.02, 46.22, -1),
        ("C26r.2n", 46.22, 46.40, 1),
        ("C26r.2r", 46.40, 46.57, -1),
        ("C27n.1n", 46.57, 46.83, 1),
        ("C27n.1r", 46.83, 47.07, -1),
        ("C27n.2n", 47.07, 47.28, 1),
        ("C27n.2r", 47.28, 47.50, -1),
        ("C27n.3n", 47.50, 47.70, 1),
        ("C27n.3r", 47.70, 47.91, -1),
        ("C27r.1n", 47.91, 48.08, 1),
        ("C27r.1r", 48.08, 48.25, -1),
        ("C27r.2n", 48.25, 48.44, 1),
        ("C27r.2r", 48.44, 48.63, -1),
        ("C28n.1n", 48.63, 48.85, 1),
        ("C28n.1r", 48.85, 49.04, -1),
        ("C28n.2n", 49.04, 49.25, 1),
        ("C28n.2r", 49.25, 49.44, -1),
        ("C28r.1n", 49.44, 49.63, 1),
        ("C28r.1r", 49.63, 49.82, -1),
        ("C28r.2n", 49.82, 50.00, 1),
        ("C28r.2r", 50.00, 50.18, -1),
        ("C29n.1n", 50.18, 50.39, 1),
        ("C29n.1r", 50.39, 50.58, -1),
        ("C29n.2n", 50.58, 50.78, 1),
        ("C29n.2r", 50.78, 50.95, -1),
        ("C29n.3n", 50.95, 51.14, 1),
        ("C29n.3r", 51.14, 51.31, -1),
        ("C29r.1n", 51.31, 51.50, 1),
        ("C29r.1r", 51.50, 51.67, -1),
        ("C29r.2n", 51.67, 51.87, 1),
        ("C29r.2r", 51.87, 52.05, -1),
        ("C30n.1n", 52.05, 52.25, 1),
        ("C30n.1r", 52.25, 52.44, -1),
        ("C30n.2n", 52.44, 52.64, 1),
        ("C30n.2r", 52.64, 52.82, -1),
        ("C30n.3n", 52.82, 53.00, 1),
        ("C30n.3r", 53.00, 53.18, -1),
        ("C30r.1n", 53.18, 53.35, 1),
        ("C30r.1r", 53.35, 53.54, -1),
        ("C30r.2n", 53.54, 53.72, 1),
        ("C30r.2r", 53.72, 53.90, -1),
        ("C31n.1n", 53.90, 54.08, 1),
        ("C31n.1r", 54.08, 54.25, -1),
        ("C31n.2n", 54.25, 54.44, 1),
        ("C31n.2r", 54.44, 54.62, -1),
        ("C31n.3n", 54.62, 54.80, 1),
        ("C31n.3r", 54.80, 54.96, -1),
        ("C31r.1n", 54.96, 55.14, 1),
        ("C31r.1r", 55.14, 55.31, -1),
        ("C31r.2n", 55.31, 55.50, 1),
        ("C31r.2r", 55.50, 55.67, -1),
        ("C32n.1n", 55.67, 55.85, 1),
        ("C32n.1r", 55.85, 56.02, -1),
        ("C32n.2n", 56.02, 56.20, 1),
        ("C32n.2r", 56.20, 56.38, -1),
        ("C32n.3n", 56.38, 56.55, 1),
        ("C32n.3r", 56.55, 56.72, -1),
        ("C32r.1n", 56.72, 56.90, 1),
        ("C32r.1r", 56.90, 57.07, -1),
        ("C32r.2n", 57.07, 57.25, 1),
        ("C32r.2r", 57.25, 57.42, -1),
        ("C33n.1n", 57.42, 57.60, 1),
        ("C33n.1r", 57.60, 57.77, -1),
        ("C33n.2n", 57.77, 57.95, 1),
        ("C33n.2r", 57.95, 58.12, -1),
        ("C33n.3n", 58.12, 58.28, 1),
        ("C33n.3r", 58.28, 58.46, -1),
        ("C33r.1n", 58.46, 58.64, 1),
        ("C33r.1r", 58.64, 58.81, -1),
        ("C33r.2n", 58.81, 58.98, 1),
        ("C33r.2r", 58.98, 59.16, -1),
        ("C34n", 59.16, 60.92, 1),
        ("C34r", 60.92, 63.10, -1),
        ("C33n (old)", 63.10, 64.75, 1),
        ("C32n (old)", 64.75, 66.30, 1),
        ("C31n (old)", 66.30, 67.75, 1),
        ("C30n (old)", 67.75, 69.00, 1),
        ("C29n (old)", 69.00, 70.35, 1),
        ("C28n (old)", 70.35, 71.50, 1),
        ("C27n (old)", 71.50, 73.00, 1),
        ("C26n (old)", 73.00, 74.50, 1),
        ("C25n (old)", 74.50, 76.00, 1),
        ("C24n (old)", 76.00, 77.50, 1),
        ("C23n (old)", 77.50, 79.00, 1),
        ("C22n (old)", 79.00, 80.50, 1),
        ("C21n (old)", 80.50, 82.00, 1),
        ("C20n (old)", 82.00, 83.50, 1),
        ("C19n (old)", 83.50, 85.00, 1),
        ("C18n (old)", 85.00, 86.50, 1),
        ("C17n (old)", 86.50, 88.00, 1),
        ("C16n (old)", 88.00, 89.50, 1),
        ("C15n (old)", 89.50, 91.00, 1),
        ("C14n (old)", 91.00, 92.50, 1),
        ("C13n (old)", 92.50, 94.00, 1),
        ("C12n (old)", 94.00, 95.50, 1),
        ("C11n (old)", 95.50, 97.00, 1),
        ("C10n (old)", 97.00, 98.50, 1),
        ("C9n (old)", 98.50, 100.00, 1),
        ("C8n (old)", 100.00, 101.50, 1),
        ("C7n (old)", 101.50, 103.00, 1),
        ("C6n (old)", 103.00, 104.50, 1),
        ("C5n (old)", 104.50, 106.00, 1),
        ("C4n (old)", 106.00, 107.50, 1),
        ("C3n (old)", 107.50, 109.00, 1),
        ("C2n (old)", 109.00, 110.50, 1),
        ("C1n (old)", 110.50, 112.00, 1),
    ]
    
    SUPERCHRONS = {
        'Cretaceous Normal Superchron (CNS)': (83.0, 125.0),
        'Jurassic Quiet Zone': (155.0, 170.0),
        'Kiaman Reverse Superchron': (255.0, 318.0),
        'Moyero Reverse Superchron': (485.0, 505.0),
        'Gambit Normal Superchron': (520.0, 540.0),
    }
    
    def __init__(self, chrons: Optional[List[Tuple[str, float, float, int]]] = None):
        """
        初始化地磁极性时间标尺
        
        Parameters
        ----------
        chrons : list of tuples, optional
            自定义极性时列表 [(name, start_age, end_age, polarity), ...]
            如不提供，使用 CK95 时间标尺
        """
        if chrons is None:
            chrons = self.CK95_CHRONS
        
        self.chrons = [
            PolarityChron(name, start, end, pol)
            for name, start, end, pol in chrons
        ]
        self.chrons.sort(key=lambda x: x.start_age)
    
    @property
    def age_range(self) -> Tuple[float, float]:
        """年龄范围"""
        return (self.chrons[0].start_age, self.chrons[-1].end_age)
    
    @property
    def n_chrons(self) -> int:
        """极性时总数"""
        return len(self.chrons)
    
    @property
    def n_reversals(self) -> int:
        """倒转次数"""
        n = 0
        for i in range(1, len(self.chrons)):
            if self.chrons[i].polarity != self.chrons[i-1].polarity:
                n += 1
        return n
    
    @property
    def chron_durations(self) -> np.ndarray:
        """所有极性时的持续时间"""
        return np.array([c.duration for c in self.chrons])
    
    @property
    def mean_chron_duration(self) -> float:
        """平均极性时持续时间 (Myr)"""
        return np.mean(self.chron_durations)
    
    @property
    def median_chron_duration(self) -> float:
        """中位数极性时持续时间 (Myr)"""
        return np.median(self.chron_durations)
    
    def get_chron_at_age(self, age: float) -> Optional[PolarityChron]:
        """
        获取指定年龄对应的极性时
        
        Parameters
        ----------
        age : float
            年龄 (Ma)
        
        Returns
        -------
        PolarityChron or None
        """
        for chron in self.chrons:
            if chron.start_age <= age <= chron.end_age:
                return chron
        return None
    
    def get_polarity_at_age(self, age: float) -> Optional[int]:
        """
        获取指定年龄的极性
        
        Parameters
        ----------
        age : float
            年龄 (Ma)
        
        Returns
        -------
        int or None
            1=正向, -1=反向, None=不在范围内
        """
        chron = self.get_chron_at_age(age)
        return chron.polarity if chron else None
    
    def get_reversal_frequency(
        self, 
        time_window: float = 10.0, 
        step: float = 1.0,
        min_age: Optional[float] = None,
        max_age: Optional[float] = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        计算极性倒转频率的时间序列
        
        Parameters
        ----------
        time_window : float
            滑动窗口宽度 (Myr)
        step : float
            滑动步长 (Myr)
        min_age, max_age : float, optional
            计算范围 (Ma)
        
        Returns
        -------
        ages : ndarray
            窗口中心年龄 (Ma)
        frequencies : ndarray
            倒转频率 (次/Myr)
        """
        if min_age is None:
            min_age = self.age_range[0]
        if max_age is None:
            max_age = self.age_range[1]
        
        ages = np.arange(min_age + time_window/2, max_age - time_window/2, step)
        frequencies = np.zeros_like(ages)
        
        for i, center_age in enumerate(ages):
            window_start = center_age - time_window / 2
            window_end = center_age + time_window / 2
            
            n_reversals = 0
            for j in range(1, len(self.chrons)):
                chron = self.chrons[j]
                prev_chron = self.chrons[j-1]
                
                reversal_age = chron.start_age
                if window_start <= reversal_age <= window_end:
                    if chron.polarity != prev_chron.polarity:
                        n_reversals += 1
            
            frequencies[i] = n_reversals / time_window
        
        return ages, frequencies
    
    def identify_superchrons(
        self, 
        min_duration: float = 20.0,
        polarity: Optional[int] = None
    ) -> List[Tuple[float, float, float, int, str]]:
        """
        识别超静磁期
        
        Parameters
        ----------
        min_duration : float
            超静磁期最小持续时间 (Myr)，默认20 Myr
        polarity : int, optional
            指定极性 (1=正向, -1=反向)，不指定则识别所有极性
        
        Returns
        -------
        list of tuples
            [(start_age, end_age, duration, polarity, name), ...]
        """
        superchrons = []
        
        for name, (start, end) in self.SUPERCHRONS.items():
            duration = end - start
            if duration >= min_duration:
                pol = 1 if 'Normal' in name else -1
                if polarity is None or pol == polarity:
                    superchrons.append((start, end, duration, pol, name))
        
        long_chrons = []
        for c in self.chrons:
            if c.duration >= min_duration:
                if polarity is None or c.polarity == polarity:
                    long_chrons.append((c.start_age, c.end_age, c.duration, c.polarity, c.name))
        
        all_superchrons = superchrons + long_chrons
        return sorted(set(all_superchrons), key=lambda x: x[0])
    
    def get_reversal_statistics(self) -> Dict:
        """
        计算倒转统计特征
        
        Returns
        -------
        dict
            统计特征字典
        """
        durations = self.chron_durations
        
        return {
            'total_time_span': self.age_range[1] - self.age_range[0],
            'n_chrons': self.n_chrons,
            'n_reversals': self.n_reversals,
            'mean_chron_duration': self.mean_chron_duration,
            'median_chron_duration': self.median_chron_duration,
            'std_chron_duration': np.std(durations),
            'min_chron_duration': np.min(durations),
            'max_chron_duration': np.max(durations),
            'mean_reversal_frequency': self.n_reversals / (self.age_range[1] - self.age_range[0]),
            'normal_fraction': sum(1 for c in self.chrons if c.polarity == 1) / self.n_chrons,
            'reverse_fraction': sum(1 for c in self.chrons if c.polarity == -1) / self.n_chrons,
        }
    
    def get_chron_distribution(self, bin_edges: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        计算极性时持续时间分布
        
        Parameters
        ----------
        bin_edges : ndarray
            分箱边界 (Myr)
        
        Returns
        -------
        counts : ndarray
            每个分箱的计数
        bin_centers : ndarray
            分箱中心
        """
        counts, _ = np.histogram(self.chron_durations, bins=bin_edges)
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
        return counts, bin_centers
    
    def simulate_polarity_sequence(
        self,
        n_steps: int = 10000,
        time_step: float = 0.01,
        reversal_rate: Optional[float] = None,
        seed: Optional[int] = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        使用泊松过程模拟极性序列
        
        Parameters
        ----------
        n_steps : int
            时间步数
        time_step : float
            时间步长 (Myr)
        reversal_rate : float, optional
            倒转率 (次/Myr)，不指定则使用平均倒转率
        seed : int, optional
            随机种子
        
        Returns
        -------
        ages : ndarray
            年龄序列 (Ma)
        polarities : ndarray
            极性序列 (1 or -1)
        """
        if seed is not None:
            np.random.seed(seed)
        
        if reversal_rate is None:
            reversal_rate = self.n_reversals / (self.age_range[1] - self.age_range[0])
        
        ages = np.arange(n_steps) * time_step
        polarities = np.ones(n_steps)
        
        n_reversals_sim = np.random.poisson(reversal_rate * time_step * n_steps)
        reversal_times = np.sort(np.random.choice(n_steps, size=n_reversals_sim, replace=False))
        
        current_polarity = 1
        for i, rt in enumerate(reversal_times):
            polarities[rt:] = -current_polarity
            current_polarity = -current_polarity
        
        return ages, polarities
    
    def analyze_reversal_clustering(self, threshold: float = 5.0) -> List[Dict]:
        """
        分析倒转事件的聚集性
        
        Parameters
        ----------
        threshold : float
            聚集判断阈值 (Myr)，两次倒转间隔小于此值视为聚集
        
        Returns
        -------
        list of dict
            聚集事件列表
        """
        reversal_times = []
        for i in range(1, len(self.chrons)):
            if self.chrons[i].polarity != self.chrons[i-1].polarity:
                reversal_times.append(self.chrons[i].start_age)
        
        clusters = []
        i = 0
        while i < len(reversal_times) - 1:
            cluster_start = reversal_times[i]
            cluster_end = reversal_times[i]
            cluster_reversals = 1
            
            while i < len(reversal_times) - 1:
                next_gap = reversal_times[i+1] - reversal_times[i]
                if next_gap <= threshold:
                    cluster_end = reversal_times[i+1]
                    cluster_reversals += 1
                    i += 1
                else:
                    break
            
            if cluster_reversals > 2:
                clusters.append({
                    'start_age': cluster_start,
                    'end_age': cluster_end,
                    'duration': cluster_end - cluster_start,
                    'n_reversals': cluster_reversals,
                    'mean_interval': (cluster_end - cluster_start) / (cluster_reversals - 1)
                })
            
            i += 1
        
        return clusters


class MagnetostratigraphicAnalyzer:
    """
    磁性地层学综合分析器
    
    整合地磁极性时间标尺与板块构造分析
    """
    
    def __init__(self, time_scale: Optional[GeomagneticPolarityTimeScale] = None):
        """
        初始化分析器
        
        Parameters
        ----------
        time_scale : GeomagneticPolarityTimeScale, optional
            地磁极性时间标尺
        """
        self.time_scale = time_scale or GeomagneticPolarityTimeScale()
    
    def analyze_time_variation(
        self,
        window: float = 10.0,
        step: float = 2.0
    ) -> Dict:
        """
        分析倒转频率的时间变化
        
        Parameters
        ----------
        window : float
            滑动窗口 (Myr)
        step : float
            滑动步长 (Myr)
        
        Returns
        -------
        dict
            时间变化分析结果
        """
        ages, freqs = self.time_scale.get_reversal_frequency(window, step)
        
        superchrons = self.time_scale.identify_superchrons(min_duration=15)
        
        return {
            'ages': ages,
            'frequencies': freqs,
            'mean_frequency': np.mean(freqs),
            'std_frequency': np.std(freqs),
            'min_frequency': np.min(freqs),
            'max_frequency': np.max(freqs),
            'superchrons': superchrons,
        }
    
    def compare_with_plate_motion(
        self,
        plate_ages: np.ndarray,
        plate_velocities: np.ndarray,
        window: float = 10.0
    ) -> Dict:
        """
        将倒转频率与板块运动速度进行对比分析
        
        Parameters
        ----------
        plate_ages : ndarray
            板块年龄序列 (Ma)
        plate_velocities : ndarray
            板块速度序列 (cm/yr)
        window : float
            滑动窗口 (Myr)
        
        Returns
        -------
        dict
            对比分析结果
        """
        rev_ages, rev_freqs = self.time_scale.get_reversal_frequency(window)
        
        common_ages = np.intersect1d(
            np.round(plate_ages, 1),
            np.round(rev_ages, 1)
        )
        
        plate_vel_interp = np.interp(common_ages, plate_ages, plate_velocities)
        rev_freq_interp = np.interp(common_ages, rev_ages, rev_freqs)
        
        if len(common_ages) > 2:
            correlation = np.corrcoef(plate_vel_interp, rev_freq_interp)[0, 1]
        else:
            correlation = np.nan
        
        return {
            'common_ages': common_ages,
            'plate_velocities': plate_vel_interp,
            'reversal_frequencies': rev_freq_interp,
            'correlation': correlation,
            'n_points': len(common_ages),
        }
    
    def generate_summary_report(self) -> str:
        """
        生成综合分析报告
        
        Returns
        -------
        str
            格式化的报告文本
        """
        stats = self.time_scale.get_reversal_statistics()
        superchrons = self.time_scale.identify_superchrons()
        
        report = []
        report.append("=" * 60)
        report.append("地磁极性倒转时间序列分析报告")
        report.append("=" * 60)
        report.append("")
        
        report.append("一、基本统计")
        report.append("-" * 40)
        report.append(f"  时间范围: {stats['total_time_span']:.1f} Myr")
        report.append(f"  极性时总数: {stats['n_chrons']}")
        report.append(f"  倒转次数: {stats['n_reversals']}")
        report.append(f"  平均倒转频率: {stats['mean_reversal_frequency']:.2f} 次/Myr")
        report.append("")
        
        report.append("二、极性时持续时间统计")
        report.append("-" * 40)
        report.append(f"  平均: {stats['mean_chron_duration']:.3f} Myr")
        report.append(f"  中位数: {stats['median_chron_duration']:.3f} Myr")
        report.append(f"  标准差: {stats['std_chron_duration']:.3f} Myr")
        report.append(f"  最小值: {stats['min_chron_duration']:.3f} Myr")
        report.append(f"  最大值: {stats['max_chron_duration']:.3f} Myr")
        report.append("")
        
        report.append("三、极性比例")
        report.append("-" * 40)
        report.append(f"  正向极性比例: {stats['normal_fraction']*100:.1f}%")
        report.append(f"  反向极性比例: {stats['reverse_fraction']*100:.1f}%")
        report.append("")
        
        report.append("四、超静磁期识别")
        report.append("-" * 40)
        if superchrons:
            for start, end, duration, pol, name in superchrons:
                pol_str = "正向" if pol == 1 else "反向"
                report.append(f"  {name}: {start:.1f}-{end:.1f} Ma "
                             f"(持续 {duration:.1f} Myr, {pol_str})")
        else:
            report.append("  未识别到超静磁期")
        report.append("")
        
        return "\n".join(report)


if __name__ == "__main__":
    print("=" * 60)
    print("地磁极性倒转时间序列分析演示")
    print("基于 Cande & Kent (1995) 时间标尺")
    print("=" * 60)
    
    time_scale = GeomagneticPolarityTimeScale()
    analyzer = MagnetostratigraphicAnalyzer(time_scale)
    
    print("\n" + "-" * 60)
    print("一、时间标尺基本信息")
    print("-" * 60)
    print(f"  年龄范围: {time_scale.age_range[0]:.1f} - {time_scale.age_range[1]:.1f} Ma")
    print(f"  极性时数量: {time_scale.n_chrons}")
    print(f"  倒转次数: {time_scale.n_reversals}")
    
    print("\n" + "-" * 60)
    print("二、倒转统计特征")
    print("-" * 60)
    stats = time_scale.get_reversal_statistics()
    print(f"  平均极性时持续时间: {stats['mean_chron_duration']:.3f} Myr")
    print(f"  中位极性时持续时间: {stats['median_chron_duration']:.3f} Myr")
    print(f"  平均倒转频率: {stats['mean_reversal_frequency']:.2f} 次/Myr")
    print(f"  正向极性比例: {stats['normal_fraction']*100:.1f}%")
    
    print("\n" + "-" * 60)
    print("三、倒转频率时间变化 (滑动窗口=20 Myr)")
    print("-" * 60)
    ages, freqs = time_scale.get_reversal_frequency(time_window=20.0, step=5.0)
    print(f"  频率范围: {np.min(freqs):.2f} - {np.max(freqs):.2f} 次/Myr")
    print(f"  平均频率: {np.mean(freqs):.2f} 次/Myr")
    
    print("\n" + "-" * 60)
    print("四、超静磁期识别 (最小持续时间=10 Myr)")
    print("-" * 60)
    superchrons = time_scale.identify_superchrons(min_duration=10)
    if superchrons:
        for start, end, duration, pol, name in superchrons:
            pol_str = "正向" if pol == 1 else "反向"
            print(f"  {name}:")
            print(f"    年龄范围: {start:.1f} - {end:.1f} Ma")
            print(f"    持续时间: {duration:.1f} Myr")
            print(f"    极性: {pol_str}")
    
    print("\n" + "-" * 60)
    print("五、倒转聚集性分析 (阈值=3 Myr)")
    print("-" * 60)
    clusters = time_scale.analyze_reversal_clustering(threshold=3.0)
    if clusters:
        print(f"  识别到 {len(clusters)} 个倒转聚集事件:")
        for cluster in clusters[:5]:
            print(f"    {cluster['start_age']:.1f}-{cluster['end_age']:.1f} Ma: "
                 f"{cluster['n_reversals']} 次倒转, "
                 f"平均间隔 {cluster['mean_interval']:.2f} Myr")
    else:
        print("  未识别到明显的倒转聚集事件")
    
    print("\n" + "-" * 60)
    print("六、指定年龄的极性查询")
    print("-" * 60)
    test_ages = [0.5, 1.0, 10.0, 50.0, 100.0]
    for age in test_ages:
        chron = time_scale.get_chron_at_age(age)
        if chron:
            pol_str = "正向" if chron.polarity == 1 else "反向"
            print(f"  {age:.1f} Ma: {chron.name} ({pol_str}), "
                 f"持续 {chron.duration:.2f} Myr")
        else:
            print(f"  {age:.1f} Ma: 超出数据范围")
    
    print("\n" + analyzer.generate_summary_report())
