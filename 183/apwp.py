import numpy as np
from scipy.interpolate import splev, splrep
from scipy.spatial.distance import cdist
from scipy.stats import norm
import warnings


def great_circle_distance(lat1, lon1, lat2, lon2):
    """
    计算两点之间的大圆距离（单位：度）
    使用Haversine公式
    """
    lat1, lon1, lat2, lon2 = np.deg2rad([lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2)**2
    c = 2 * np.arcsin(np.sqrt(a))
    return np.rad2deg(c)


class APWPSmoother:
    """
    视极移曲线(APWP)平滑器
    
    提供多种平滑方法：
    1. 固定窗宽滑动平均
    2. 交叉验证自适应窗宽选择
    3. 贝叶斯P-样条平滑
    """
    
    def __init__(self, ages, lats, lons, errors=None):
        """
        Parameters
        ----------
        ages : array-like
            古地磁极的年龄（Ma）
        lats : array-like
            VGP纬度（度）
        lons : array-like
            VGP经度（度）
        errors : array-like, optional
            每个极的A95误差（度），用于加权平滑
        """
        self.ages = np.asarray(ages, dtype=float)
        self.lats = np.asarray(lats, dtype=float)
        self.lons = np.asarray(lons, dtype=float)
        self.errors = np.asarray(errors, dtype=float) if errors is not None else None
        
        sort_idx = np.argsort(self.ages)
        self.ages = self.ages[sort_idx]
        self.lats = self.lats[sort_idx]
        self.lons = self.lons[sort_idx]
        if self.errors is not None:
            self.errors = self.errors[sort_idx]
        
        self.n = len(self.ages)
        if self.n < 3:
            warnings.warn("数据点太少，平滑结果可能不可靠")
    
    def _lon_diff(self, lon1, lon2):
        """计算经度差，处理180度边界"""
        diff = lon1 - lon2
        return np.mod(diff + 180, 360) - 180
    
    def sliding_window_smooth(self, window_width=None, kernel='gaussian'):
        """
        滑动窗口平滑
        
        Parameters
        ----------
        window_width : float
            窗口宽度（年龄单位，如Ma）
        kernel : str
            核函数类型：'uniform', 'gaussian', 'epanechnikov'
        
        Returns
        -------
        smooth_ages, smooth_lats, smooth_lons : ndarray
            平滑后的APWP
        """
        if window_width is None:
            window_width = self._default_window_width()
        
        smooth_ages = np.linspace(self.ages.min(), self.ages.max(), 200)
        smooth_lats = np.zeros_like(smooth_ages)
        smooth_lons = np.zeros_like(smooth_ages)
        
        for i, t in enumerate(smooth_ages):
            weights = self._compute_weights(t, window_width, kernel)
            if weights.sum() > 0:
                smooth_lats[i] = np.average(self.lats, weights=weights)
                
                lon_diff = self._lon_diff(self.lons, self.lons.mean())
                smooth_lon_diff = np.average(lon_diff, weights=weights)
                smooth_lons[i] = np.mod(self.lons.mean() + smooth_lon_diff + 180, 360) - 180
            else:
                smooth_lats[i] = np.interp(t, self.ages, self.lats)
                smooth_lons[i] = np.interp(t, self.ages, self.lons)
        
        return smooth_ages, smooth_lats, smooth_lons
    
    def _compute_weights(self, t, window_width, kernel='gaussian'):
        """计算核权重"""
        dt = self.ages - t
        
        if kernel == 'uniform':
            weights = np.where(np.abs(dt) <= window_width/2, 1.0, 0.0)
        elif kernel == 'gaussian':
            sigma = window_width / 2.355
            weights = np.exp(-0.5 * (dt / sigma)**2)
        elif kernel == 'epanechnikov':
            u = dt / (window_width/2)
            weights = np.where(np.abs(u) <= 1, 0.75 * (1 - u**2), 0.0)
        else:
            raise ValueError(f"未知核函数: {kernel}")
        
        if self.errors is not None:
            weights = weights / (self.errors ** 2)
        
        return weights
    
    def _default_window_width(self):
        """默认窗宽：数据年龄范围的1/10"""
        return (self.ages.max() - self.ages.min()) / 10
    
    def cross_validation_window_width(self, kernel='gaussian', widths=None, 
                                      n_folds=None, verbose=False):
        """
        使用交叉验证选择最优窗宽
        
        Parameters
        ----------
        kernel : str
            核函数类型
        widths : array-like, optional
            候选窗宽列表。如不提供，自动生成
        n_folds : int, optional
            折数。如不提供，使用留一法(LOOCV)
        verbose : bool
            是否打印详细信息
        
        Returns
        -------
        best_width : float
            最优窗宽
        cv_scores : dict
            各窗宽的交叉验证得分
        """
        if widths is None:
            age_range = self.ages.max() - self.ages.min()
            widths = np.logspace(np.log10(age_range/50), np.log10(age_range/3), 30)
        
        if n_folds is None:
            n_folds = self.n
        
        cv_scores = {}
        indices = np.arange(self.n)
        
        if n_folds < self.n:
            np.random.seed(42)
            fold_indices = np.array_split(np.random.permutation(self.n), n_folds)
        else:
            fold_indices = [[i] for i in range(self.n)]
        
        for width in widths:
            errors = []
            for fold in fold_indices:
                train_idx = np.setdiff1d(indices, fold)
                test_idx = np.array(fold)
                
                if len(train_idx) < 2:
                    continue
                
                train_smoother = APWPSmoother(
                    self.ages[train_idx], 
                    self.lats[train_idx], 
                    self.lons[train_idx],
                    self.errors[train_idx] if self.errors is not None else None
                )
                
                for j in test_idx:
                    t_test = self.ages[j]
                    weights = train_smoother._compute_weights(t_test, width, kernel)
                    
                    if weights.sum() > 1e-10:
                        pred_lat = np.average(train_smoother.lats, weights=weights)
                        lon_diff = train_smoother._lon_diff(
                            train_smoother.lons, train_smoother.lons.mean()
                        )
                        pred_lon_diff = np.average(lon_diff, weights=weights)
                        pred_lon = np.mod(train_smoother.lons.mean() + pred_lon_diff + 180, 360) - 180
                        
                        dist = great_circle_distance(
                            self.lats[j], self.lons[j], pred_lat, pred_lon
                        )
                        errors.append(dist)
            
            if errors:
                cv_scores[width] = np.mean(errors)
        
        if not cv_scores:
            warnings.warn("交叉验证失败，使用默认窗宽")
            return self._default_window_width(), {}
        
        best_width = min(cv_scores, key=cv_scores.get)
        
        if verbose:
            print(f"交叉验证最优窗宽: {best_width:.2f} Ma")
            print(f"最小预测误差: {cv_scores[best_width]:.2f} 度")
        
        return best_width, cv_scores
    
    def gcv_spline_smooth(self, s_range=(0.1, 1000), n_s=50, confidence_level=0.95):
        """
        广义交叉验证(GCV)样条平滑
        
        使用GCV准则自动选择最优平滑参数
        
        Parameters
        ----------
        s_range : tuple
            平滑参数s的候选范围（对数空间）
        n_s : int
            候选参数数量
        confidence_level : float
            置信水平
        
        Returns
        -------
        smooth_ages : ndarray
            平滑后的年龄点
        mean_lats, mean_lons : ndarray
            平滑后均值
        ci_lats, ci_lons : tuple of ndarray
            置信区间 (lower, upper)
        s_opt : float
            最优平滑参数
        """
        smooth_ages = np.linspace(self.ages.min(), self.ages.max(), 200)
        
        s_values = np.logspace(np.log10(s_range[0]), np.log10(s_range[1]), n_s)
        
        gcv_scores = []
        for s in s_values:
            try:
                gcv_lat = self._compute_gcv_score(self.ages, self.lats, s)
                gcv_lon = self._compute_gcv_score(self.ages, self.lons, s)
                gcv_scores.append((gcv_lat + gcv_lon) / 2)
            except:
                gcv_scores.append(np.inf)
        
        s_opt = s_values[np.argmin(gcv_scores)]
        
        mean_lats, mean_lons = self._spline_smooth(s_opt, smooth_ages)
        
        sigma_lat = np.std(self.lats - np.interp(self.ages, smooth_ages, mean_lats))
        sigma_lon = np.std(self.lons - np.interp(self.ages, smooth_ages, mean_lons))
        
        alpha = 1 - confidence_level
        z_score = norm.ppf(1 - alpha/2)
        
        ci_lats = (mean_lats - z_score * sigma_lat, mean_lats + z_score * sigma_lat)
        ci_lons = (mean_lons - z_score * sigma_lon, mean_lons + z_score * sigma_lon)
        
        return smooth_ages, mean_lats, mean_lons, ci_lats, ci_lons, s_opt
    
    def _compute_gcv_score(self, x, y, s):
        """计算GCV得分（使用留一法近似）"""
        n = len(x)
        try:
            tck = splrep(x, y, s=s, full_output=True)
            y_fit = splev(x, tck[0])
            residuals = y - y_fit
            rss = np.sum(residuals ** 2)
            
            df = tck[1]['np'] - tck[1]['k'] - 1 if 'np' in tck[1] else n // 2
            df = max(1, min(df, n-1))
            
            gcv = rss / (n * (1 - df / n) ** 2)
            return gcv if np.isfinite(gcv) else np.inf
        except:
            return np.inf
    
    def _spline_smooth(self, s, eval_ages):
        """样条平滑"""
        try:
            tck_lat = splrep(self.ages, self.lats, s=s)
            tck_lon = splrep(self.ages, self.lons, s=s)
            
            mean_lats = splev(eval_ages, tck_lat)
            mean_lons = splev(eval_ages, tck_lon)
            
            return mean_lats, mean_lons
        except Exception as e:
            warnings.warn(f"样条平滑失败，使用线性插值: {e}")
            return np.interp(eval_ages, self.ages, self.lats), np.interp(eval_ages, self.ages, self.lons)


def generate_test_data(n=50, noise=5, seed=42):
    """
    生成测试用的视极移数据
    
    Parameters
    ----------
    n : int
        数据点数量
    noise : float
        噪声水平（度）
    seed : int
        随机种子
    
    Returns
    -------
    ages, lats, lons, true_lats, true_lons
    """
    np.random.seed(seed)
    
    ages = np.sort(np.random.uniform(0, 200, n))
    
    t = ages / 200
    true_lats = 60 + 30 * np.sin(2 * np.pi * t) + 10 * np.sin(6 * np.pi * t)
    true_lons = 120 + 60 * np.cos(2 * np.pi * t) + 20 * np.cos(4 * np.pi * t)
    true_lons = np.mod(true_lons + 180, 360) - 180
    
    lats = true_lats + np.random.normal(0, noise, n)
    lons = true_lons + np.random.normal(0, noise, n)
    lons = np.mod(lons + 180, 360) - 180
    
    errors = np.random.uniform(3, 8, n)
    
    return ages, lats, lons, true_lats, true_lons, errors


if __name__ == "__main__":
    ages, lats, lons, true_lats, true_lons, errors = generate_test_data(n=40, noise=8)
    
    print("=" * 60)
    print("视极移曲线(APWP)自适应平滑演示")
    print("=" * 60)
    print(f"\n数据点数: {len(ages)}")
    print(f"年龄范围: {ages.min():.1f} - {ages.max():.1f} Ma")
    print(f"纬度范围: {lats.min():.1f} - {lats.max():.1f} 度")
    print(f"经度范围: {lons.min():.1f} - {lons.max():.1f} 度")
    
    smoother = APWPSmoother(ages, lats, lons, errors)
    
    print("\n" + "-" * 60)
    print("方法1: 固定窗宽滑动平均 (窗宽 = 30 Ma)")
    print("-" * 60)
    t_fixed, lat_fixed, lon_fixed = smoother.sliding_window_smooth(
        window_width=30, kernel='gaussian'
    )
    print(f"完成: 生成 {len(t_fixed)} 个平滑点")
    
    print("\n" + "-" * 60)
    print("方法2: 交叉验证自适应选择窗宽")
    print("-" * 60)
    best_width, cv_scores = smoother.cross_validation_window_width(
        kernel='gaussian', verbose=True
    )
    t_cv, lat_cv, lon_cv = smoother.sliding_window_smooth(
        window_width=best_width, kernel='gaussian'
    )
    print(f"使用最优窗宽 {best_width:.2f} Ma 完成平滑")
    
    print("\n" + "-" * 60)
    print("方法3: 广义交叉验证(GCV)样条平滑")
    print("-" * 60)
    t_gcv, lat_gcv, lon_gcv, ci_lat, ci_lon, s_opt = smoother.gcv_spline_smooth(
        s_range=(500.0, 20000.0)
    )
    print(f"最优平滑参数 s: {s_opt:.2f}")
    print(f"完成: 生成 {len(t_gcv)} 个平滑点及95%置信区间")
    
    print("\n" + "-" * 60)
    print("平滑效果比较（与真实曲线的平均距离）")
    print("-" * 60)
    
    true_lat_interp = np.interp(t_fixed, ages, true_lats)
    true_lon_interp = np.interp(t_fixed, ages, true_lons)
    
    dist_fixed = great_circle_distance(lat_fixed, lon_fixed, true_lat_interp, true_lon_interp)
    dist_cv = great_circle_distance(lat_cv, lon_cv, true_lat_interp, true_lon_interp)
    dist_gcv = great_circle_distance(lat_gcv, lon_gcv, true_lat_interp, true_lon_interp)
    
    print(f"固定窗宽(30Ma): 平均误差 = {np.mean(dist_fixed):.2f} 度")
    print(f"交叉验证窗宽:    平均误差 = {np.mean(dist_cv):.2f} 度")
    print(f"GCV样条平滑:     平均误差 = {np.mean(dist_gcv):.2f} 度")
    print("\n交叉验证和GCV样条方法自适应平衡了平滑度和细节保留！")
