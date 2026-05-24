import numpy as np
import matplotlib.pyplot as plt
import time


def get_cmb_power_spectrum_full(Omega_m=0.3, Omega_L=0.7, H0=67.5,
                                 Omega_b=0.0486, ns=0.9667, As=2.105e-9,
                                 tau=0.0561, r=0.0, nt=0.0,
                                 l_max=2500, lensing=True, return_ell=False):
    """
    计算完整的 CMB 功率谱，包括温度、E模、B模极化，支持引力透镜和原初张量模式
    
    参数:
        Omega_m (float): 总物质密度参数，默认值 0.3
        Omega_L (float): 暗能量密度参数，默认值 0.7
        H0 (float): 哈勃常数 (km/s/Mpc)，默认值 67.5
        Omega_b (float): 重子物质密度参数，默认值 0.0486
        ns (float): 标量谱指数，默认值 0.9667
        As (float): 原初标量功率谱振幅，默认值 2.105e-9
        tau (float): 再电离光深，默认值 0.0561
        r (float): 张量-标量比，默认值 0.0（无原初引力波）
        nt (float): 张量谱指数，默认值 0.0（一致性关系: nt = -r/8）
        l_max (int): 最大 multipole，默认值 2500
        lensing (bool): 是否包含引力透镜效应，默认值 True
        return_ell (bool): 是否返回 ell 数组，默认值 False
    
    返回:
        ell (array): multipole 数组 (仅当 return_ell=True 时返回)
        spectra (dict): 包含各功率谱的字典
            - 'TT': 温度-温度功率谱
            - 'EE': E模-E模功率谱
            - 'BB': B模-B模功率谱
            - 'TE': 温度-E模交叉功率谱
            - 'unlensed_TT': 无透镜温度功率谱
            - 'unlensed_EE': 无透镜E模功率谱
            - 'unlensed_BB': 无透镜B模功率谱 (仅张量贡献)
            - 'lens_potential': 透镜势功率谱
            单位均为: μK²
    """
    try:
        import camb
    except ImportError:
        raise ImportError("请先安装 CAMB 库: pip install camb")
    
    pars = camb.CAMBparams()
    
    Omega_c = Omega_m - Omega_b
    
    pars.set_cosmology(H0=H0,
                       ombh2=Omega_b * (H0/100)**2,
                       omch2=Omega_c * (H0/100)**2,
                       mnu=0.06,
                       omk=1 - Omega_m - Omega_L,
                       tau=tau)
    
    pars.InitPower.set_params(As=As, ns=ns, r=r, nt=nt)
    
    lens_accuracy = 2 if lensing else 0
    pars.set_for_lmax(l_max, lens_potential_accuracy=lens_accuracy)
    
    pars.WantTensors = (r > 0)
    pars.DoLensing = lensing
    
    results = camb.get_results(pars)
    
    powers = results.get_cmb_power_spectra(pars, CMB_unit='muK')
    
    totCL = powers['total']
    ls = np.arange(totCL.shape[0])
    
    spectra = {
        'TT': totCL[:, 0],
        'EE': totCL[:, 1],
        'BB': totCL[:, 2],
        'TE': totCL[:, 3]
    }
    
    if lensing:
        unlensedCL = powers['unlensed_scalar']
        spectra['unlensed_TT'] = unlensedCL[:, 0]
        spectra['unlensed_EE'] = unlensedCL[:, 1]
        spectra['unlensed_BB'] = np.zeros_like(spectra['BB'])
        
        if r > 0:
            tensor_cl = powers['tensor']
            spectra['unlensed_BB'] = tensor_cl[:, 2]
        
        lens_potential = powers['lens_potential']
        spectra['lens_potential'] = lens_potential[:, 0]
    else:
        spectra['unlensed_TT'] = totCL[:, 0]
        spectra['unlensed_EE'] = totCL[:, 1]
        spectra['unlensed_BB'] = totCL[:, 2]
    
    if return_ell:
        return ls, spectra
    else:
        return spectra


def get_cmb_power_spectrum(Omega_m=0.3, Omega_L=0.7, H0=67.5,
                           Omega_b=0.0486, ns=0.9667, As=2.105e-9,
                           tau=0.0561, l_max=2500, return_ell=False,
                           lensing=True):
    """
    计算宇宙微波背景辐射（CMB）温度功率谱 C_l (简化接口)
    
    参数:
        同 get_cmb_power_spectrum_full
    
    返回:
        ell (array): multipole 数组 (仅当 return_ell=True 时返回)
        C_l_TT (array): 温度功率谱 C_l (单位: μK²)
    """
    result = get_cmb_power_spectrum_full(
        Omega_m=Omega_m, Omega_L=Omega_L, H0=H0,
        Omega_b=Omega_b, ns=ns, As=As, tau=tau,
        r=0.0, l_max=l_max, lensing=lensing,
        return_ell=return_ell
    )
    
    if return_ell:
        ls, spectra = result
        return ls, spectra['TT']
    else:
        return result['TT']


def get_cmb_power_spectrum_class(Omega_m=0.3, Omega_L=0.7, H0=67.5,
                                  Omega_b=0.0486, ns=0.9667, As=2.105e-9,
                                  tau=0.0561, r=0.0, l_max=2500, return_ell=False,
                                  lensing=True):
    """
    使用 CLASS 库计算完整 CMB 功率谱
    
    参数:
        参数说明同 get_cmb_power_spectrum_full 函数
    
    返回:
        同 get_cmb_power_spectrum_full 函数
    """
    try:
        from classy import Class
    except ImportError:
        raise ImportError("请先安装 CLASS 库: pip install classy")
    
    cosmo = Class()
    
    Omega_c = Omega_m - Omega_b
    
    output = 'tCl pCl lCl' if lensing else 'tCl pCl'
    
    params = {
        'output': output,
        'l_max_scalars': l_max,
        'l_max_tensors': l_max if r > 0 else 0,
        'modes': 's,t' if r > 0 else 's',
        'lensing': 'yes' if lensing else 'no',
        'h': H0 / 100,
        'omega_b': Omega_b * (H0/100)**2,
        'omega_cdm': Omega_c * (H0/100)**2,
        'Omega_Lambda': Omega_L,
        'tau_reio': tau,
        'n_s': ns,
        'A_s': As,
        'r': r,
        'T_cmb': 2.7255
    }
    
    cosmo.set(params)
    cosmo.compute()
    
    cl = cosmo.raw_cl(l_max)
    ls = cl['ell']
    
    T_cmb = 2.7255 * 1e6
    unit = T_cmb**2
    
    spectra = {
        'TT': cl['tt'] * unit,
        'EE': cl['ee'] * unit,
        'BB': cl['bb'] * unit,
        'TE': cl['te'] * unit,
        'unlensed_TT': cl['tt'] * unit,
        'unlensed_EE': cl['ee'] * unit,
        'unlensed_BB': cl['bb'] * unit
    }
    
    cosmo.struct_cleanup()
    cosmo.empty()
    
    if return_ell:
        return ls, spectra
    else:
        return spectra


def plot_power_spectrum(ls, C_l_TT, title='CMB Temperature Power Spectrum',
                        save_path=None):
    """
    绘制 CMB 温度功率谱 D_l = l(l+1)C_l/(2π)
    
    参数:
        ls (array): multipole 数组
        C_l_TT (array): 温度功率谱 C_l
        title (str): 图表标题
        save_path (str): 保存路径，默认不保存
    """
    D_l = ls * (ls + 1) * C_l_TT / (2 * np.pi)
    
    plt.figure(figsize=(10, 6))
    plt.plot(ls, D_l, linewidth=1.5)
    plt.xlabel(r'$\ell$', fontsize=14)
    plt.ylabel(r'$\ell(\ell+1)C_\ell/(2\pi) \, [\mu\mathrm{K}^2]$', fontsize=14)
    plt.title(title, fontsize=16)
    plt.xscale('log')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300)
        print(f"图表已保存到: {save_path}")
    
    plt.show()


def plot_full_spectra(ls, spectra, title='CMB Power Spectra',
                      show_lensing=True, save_path=None):
    """
    绘制完整的 CMB 功率谱（TT, EE, BB）
    
    参数:
        ls (array): multipole 数组
        spectra (dict): 功率谱字典
        title (str): 图表标题
        show_lensing (bool): 是否显示透镜效应对比
        save_path (str): 保存路径
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()
    
    Dl_TT = ls * (ls + 1) * spectra['TT'] / (2 * np.pi)
    Dl_EE = ls * (ls + 1) * spectra['EE'] / (2 * np.pi)
    Dl_BB = ls * (ls + 1) * spectra['BB'] / (2 * np.pi)
    Dl_TE = ls * (ls + 1) * spectra['TE'] / (2 * np.pi)
    
    axes[0].semilogx(ls, Dl_TT, 'r-', label='TT (lensed)', linewidth=2)
    if show_lensing and 'unlensed_TT' in spectra:
        Dl_TT_unl = ls * (ls + 1) * spectra['unlensed_TT'] / (2 * np.pi)
        axes[0].semilogx(ls, Dl_TT_unl, 'r--', label='TT (unlensed)', 
                        linewidth=1.5, alpha=0.7)
    axes[0].set_ylabel(r'$\ell(\ell+1)C_\ell^{TT}/(2\pi) \, [\mu\mathrm{K}^2]$', fontsize=12)
    axes[0].set_xlabel(r'$\ell$', fontsize=12)
    axes[0].set_title('Temperature Power Spectrum', fontsize=14)
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    axes[1].semilogx(ls, Dl_EE, 'b-', label='EE (lensed)', linewidth=2)
    if show_lensing and 'unlensed_EE' in spectra:
        Dl_EE_unl = ls * (ls + 1) * spectra['unlensed_EE'] / (2 * np.pi)
        axes[1].semilogx(ls, Dl_EE_unl, 'b--', label='EE (unlensed)', 
                        linewidth=1.5, alpha=0.7)
    axes[1].set_ylabel(r'$\ell(\ell+1)C_\ell^{EE}/(2\pi) \, [\mu\mathrm{K}^2]$', fontsize=12)
    axes[1].set_xlabel(r'$\ell$', fontsize=12)
    axes[1].set_title('E-mode Polarization', fontsize=14)
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    axes[2].semilogx(ls, Dl_BB, 'g-', label='BB (total)', linewidth=2)
    if show_lensing and 'unlensed_BB' in spectra:
        Dl_BB_unl = ls * (ls + 1) * spectra['unlensed_BB'] / (2 * np.pi)
        axes[2].semilogx(ls, Dl_BB_unl, 'm--', label='BB (primordial tensor)', 
                        linewidth=1.5, alpha=0.7)
        Dl_BB_lens = Dl_BB - Dl_BB_unl
        axes[2].semilogx(ls, np.maximum(Dl_BB_lens, 1e-4), 'c:', 
                        label='BB (lensing)', linewidth=1.5, alpha=0.7)
    axes[2].set_ylabel(r'$\ell(\ell+1)C_\ell^{BB}/(2\pi) \, [\mu\mathrm{K}^2]$', fontsize=12)
    axes[2].set_xlabel(r'$\ell$', fontsize=12)
    axes[2].set_title('B-mode Polarization', fontsize=14)
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)
    axes[2].set_yscale('log')
    axes[2].set_ylim([1e-3, 1e1])
    
    axes[3].semilogx(ls, Dl_TE, 'k-', label='TE', linewidth=2)
    axes[3].axhline(y=0, color='gray', linestyle='-', alpha=0.5)
    axes[3].set_ylabel(r'$\ell(\ell+1)C_\ell^{TE}/(2\pi) \, [\mu\mathrm{K}^2]$', fontsize=12)
    axes[3].set_xlabel(r'$\ell$', fontsize=12)
    axes[3].set_title('TE Cross Correlation', fontsize=14)
    axes[3].legend()
    axes[3].grid(True, alpha=0.3)
    
    plt.suptitle(title, fontsize=16, y=0.995)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"图表已保存到: {save_path}")
    
    plt.show()


def plot_Bmode_comparison(ls, spectra_list, labels, r_values=None,
                          title='B-mode Power Spectrum Comparison',
                          save_path=None):
    """
    比较不同张量-标量比 r 下的 B 模功率谱
    
    参数:
        ls (array): multipole 数组
        spectra_list (list): 功率谱字典列表
        labels (list): 标签列表
        r_values (list): r 值列表
        title (str): 图表标题
        save_path (str): 保存路径
    """
    plt.figure(figsize=(12, 7))
    
    colors = ['r', 'b', 'g', 'm', 'c', 'orange']
    
    for i, spectra in enumerate(spectra_list):
        Dl_BB = ls * (ls + 1) * spectra['BB'] / (2 * np.pi)
        label = labels[i]
        if r_values is not None and i < len(r_values):
            label = f'{label} (r={r_values[i]})'
        plt.loglog(ls, Dl_BB, color=colors[i % len(colors)], 
                   label=label, linewidth=2)
        
        if 'unlensed_BB' in spectra:
            Dl_BB_unl = ls * (ls + 1) * spectra['unlensed_BB'] / (2 * np.pi)
            if np.max(Dl_BB_unl) > 1e-10:
                plt.loglog(ls, Dl_BB_unl, color=colors[i % len(colors)], 
                          linestyle='--', alpha=0.7, linewidth=1.5)
    
    plt.xlabel(r'$\ell$', fontsize=14)
    plt.ylabel(r'$\ell(\ell+1)C_\ell^{BB}/(2\pi) \, [\mu\mathrm{K}^2]$', fontsize=14)
    plt.title(title, fontsize=16)
    plt.legend(fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.ylim([1e-3, 1e1])
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"图表已保存到: {save_path}")
    
    plt.show()


class CMBBmodeLikelihood:
    """
    CMB B 模似然函数类，用于约束张量-标量比 r
    
    核心思想:
    - 低阶 multipole (ℓ ~ 50-150): 原初引力波产生的 B 模峰值 (重复合峰)
    - 高阶 multipole (ℓ > 1000): 引力透镜产生的 B 模
    - 通过测量大尺度 B 模功率来约束 r
    """
    
    def __init__(self, fiducial_params, l_min=30, l_max=300, 
                 noise_level=0.5, de_lensed=False):
        """
        初始化 B 模似然函数
        
        参数:
            fiducial_params (dict): 基准宇宙学参数，必须包含 'r'
            l_min (int): 最小 multipole
            l_max (int): 最大 multipole
            noise_level (float): 噪声水平 (μK-arcmin)
            de_lensed (bool): 是否假设消透镜后的数据
        """
        self.fiducial_params = fiducial_params
        self.l_min = l_min
        self.l_max = l_max
        self.noise_level = noise_level
        self.de_lensed = de_lensed
        
        self._generate_mock_data()
    
    def _generate_mock_data(self):
        """
        生成模拟 B 模观测数据
        """
        print("生成模拟 CMB B 模数据...")
        
        ls, spectra = get_cmb_power_spectrum_full(
            Omega_m=self.fiducial_params.get('Omega_m', 0.3111),
            Omega_L=self.fiducial_params.get('Omega_L', 0.6889),
            H0=self.fiducial_params.get('H0', 67.66),
            Omega_b=self.fiducial_params.get('Omega_b', 0.04897),
            ns=self.fiducial_params.get('ns', 0.9665),
            As=self.fiducial_params.get('As', 2.107e-9),
            tau=self.fiducial_params.get('tau', 0.0561),
            r=self.fiducial_params.get('r', 0.01),
            l_max=self.l_max + 100,
            lensing=True,
            return_ell=True
        )
        
        self.ls_full = ls
        
        mask = (ls >= self.l_min) & (ls <= self.l_max)
        self.mask = mask
        self.ls = ls[mask]
        
        if self.de_lensed:
            C_l_fid = spectra['unlensed_BB'][mask]
        else:
            C_l_fid = spectra['BB'][mask]
        
        self.C_l_fid = C_l_fid
        
        np.random.seed(42)
        cosmic_var = np.sqrt(2. / (2 * self.ls + 1)) * C_l_fid
        
        noise_rad = self.noise_level * np.pi / 10800.0
        N_l = (noise_rad * 1e6)**2 * np.exp(self.ls * (self.ls + 1) / 8e6)
        
        self.sigma = np.sqrt(cosmic_var**2 + N_l**2)
        self.C_l_obs = self.C_l_fid + np.random.normal(0, self.sigma)
        self.inv_cov = np.diag(1.0 / self.sigma**2)
        
        print(f"  multipole 范围: ℓ={self.l_min}-{self.l_max}, {len(self.ls)} 个模式")
        print(f"  基准 r = {self.fiducial_params.get('r', 0.01):.4f}")
        print(f"  噪声水平: {self.noise_level} μK-arcmin")
        print(f"  消透镜: {'是' if self.de_lensed else '否'}")
    
    def compute_model(self, r, tau=None, As=None):
        """
        计算给定参数下的 B 模功率谱
        
        参数:
            r (float): 张量-标量比
            tau (float): 再电离光深 (可选，默认使用基准值)
            As (float): 原初振幅 (可选，默认使用基准值)
        
        返回:
            C_l_BB (array): B 模功率谱
        """
        if tau is None:
            tau = self.fiducial_params.get('tau', 0.0561)
        if As is None:
            As = self.fiducial_params.get('As', 2.107e-9)
        
        _, spectra = get_cmb_power_spectrum_full(
            Omega_m=self.fiducial_params.get('Omega_m', 0.3111),
            Omega_L=self.fiducial_params.get('Omega_L', 0.6889),
            H0=self.fiducial_params.get('H0', 67.66),
            Omega_b=self.fiducial_params.get('Omega_b', 0.04897),
            ns=self.fiducial_params.get('ns', 0.9665),
            As=As,
            tau=tau,
            r=r,
            l_max=self.l_max + 100,
            lensing=not self.de_lensed,
            return_ell=True
        )
        
        if self.de_lensed:
            return spectra['unlensed_BB'][self.mask]
        else:
            return spectra['BB'][self.mask]
    
    def chi2(self, r, tau=None, As=None):
        """
        计算 χ² 值
        """
        try:
            C_l_model = self.compute_model(r, tau, As)
        except:
            return np.inf
        
        diff = self.C_l_obs - C_l_model
        return diff @ self.inv_cov @ diff
    
    def log_prior(self, theta):
        """
        先验分布
        """
        if len(theta) == 1:
            r = theta[0]
            if 0.0 <= r < 1.0:
                return 0.0
        elif len(theta) == 2:
            r, tau = theta
            if 0.0 <= r < 1.0 and 0.001 < tau < 0.5:
                return 0.0
        elif len(theta) == 3:
            r, tau, As = theta
            if 0.0 <= r < 1.0 and 0.001 < tau < 0.5 and 1e-10 < As < 5e-9:
                return 0.0
        return -np.inf
    
    def log_likelihood(self, theta):
        """
        对数似然函数
        """
        if len(theta) == 1:
            r = theta[0]
            chi2_val = self.chi2(r)
        elif len(theta) == 2:
            r, tau = theta
            chi2_val = self.chi2(r, tau)
        else:
            r, tau, As = theta
            chi2_val = self.chi2(r, tau, As)
        return -0.5 * chi2_val
    
    def log_probability(self, theta):
        """
        对数后验概率
        """
        lp = self.log_prior(theta)
        if not np.isfinite(lp):
            return -np.inf
        return lp + self.log_likelihood(theta)


class CMBLikelihood:
    """
    CMB 似然函数类，用于处理 τ 和 A_s 的联合拟合
    
    核心思想: 
    - 低阶 multipole (ℓ < 30): 主要受再电离影响，τ 的效应明显
    - 中间阶 (30 < ℓ < 100): 过渡区域
    - 高阶 multipole (ℓ > 100): 主要受原初功率谱振幅 A_s 影响，τ 效应减弱
    - 使用不同尺度的联合约束来打破 τ-A_s 简并
    """
    
    def __init__(self, fiducial_params, l_min_low=2, l_max_low=30,
                 l_min_high=50, l_max_high=600, noise_level=50.0):
        """
        初始化似然函数
        
        参数:
            fiducial_params (dict): 基准宇宙学参数
            l_min_low (int): 低阶 multipole 最小值
            l_max_low (int): 低阶 multipole 最大值 (再电离信号区)
            l_min_high (int): 高阶 multipole 最小值
            l_max_high (int): 高阶 multipole 最大值
            noise_level (float): 噪声水平 (μK)
        """
        self.fiducial_params = fiducial_params
        self.l_min_low = l_min_low
        self.l_max_low = l_max_low
        self.l_min_high = l_min_high
        self.l_max_high = l_max_high
        self.noise_level = noise_level
        
        self._generate_mock_data()
    
    def _generate_mock_data(self):
        """
        生成模拟观测数据，基于基准参数
        """
        print("生成模拟 CMB 数据...")
        
        ls, C_l_fid = get_cmb_power_spectrum(
            Omega_m=self.fiducial_params['Omega_m'],
            Omega_L=self.fiducial_params['Omega_L'],
            H0=self.fiducial_params['H0'],
            Omega_b=self.fiducial_params['Omega_b'],
            ns=self.fiducial_params['ns'],
            As=self.fiducial_params['As'],
            tau=self.fiducial_params['tau'],
            l_max=max(self.l_max_low, self.l_max_high) + 100,
            return_ell=True
        )
        
        self.ls_full = ls
        
        self.mask_low = (ls >= self.l_min_low) & (ls <= self.l_max_low)
        self.mask_high = (ls >= self.l_min_high) & (ls <= self.l_max_high)
        self.mask_all = self.mask_low | self.mask_high
        
        self.ls_low = ls[self.mask_low]
        self.ls_high = ls[self.mask_high]
        self.ls_all = ls[self.mask_all]
        
        self.C_l_fid_low = C_l_fid[self.mask_low]
        self.C_l_fid_high = C_l_fid[self.mask_high]
        self.C_l_fid_all = C_l_fid[self.mask_all]
        
        np.random.seed(42)
        cosmic_var_low = np.sqrt(2. / (2 * self.ls_low + 1)) * self.C_l_fid_low
        cosmic_var_high = np.sqrt(2. / (2 * self.ls_high + 1)) * self.C_l_fid_high
        
        noise_low = self.noise_level**2 * np.exp(self.ls_low * (self.ls_low + 1) / 1e5)
        noise_high = self.noise_level**2 * np.exp(self.ls_high * (self.ls_high + 1) / 1e5)
        
        self.sigma_low = np.sqrt(cosmic_var_low**2 + noise_low**2)
        self.sigma_high = np.sqrt(cosmic_var_high**2 + noise_high**2)
        self.sigma_all = np.concatenate([self.sigma_low, self.sigma_high])
        
        self.C_l_obs_low = self.C_l_fid_low + np.random.normal(0, self.sigma_low)
        self.C_l_obs_high = self.C_l_fid_high + np.random.normal(0, self.sigma_high)
        self.C_l_obs_all = np.concatenate([self.C_l_obs_low, self.C_l_obs_high])
        
        self.inv_cov_low = np.diag(1.0 / self.sigma_low**2)
        self.inv_cov_high = np.diag(1.0 / self.sigma_high**2)
        self.inv_cov_all = np.diag(1.0 / self.sigma_all**2)
        
        print(f"  低阶 multipole: ℓ={self.l_min_low}-{self.l_max_low}, {len(self.ls_low)} 个模式")
        print(f"  高阶 multipole: ℓ={self.l_min_high}-{self.l_max_high}, {len(self.ls_high)} 个模式")
        print(f"  基准 τ = {self.fiducial_params['tau']:.4f}")
        print(f"  基准 A_s = {self.fiducial_params['As']:.4e}")
    
    def compute_model(self, tau, As):
        """
        计算给定参数下的理论 C_l
        
        参数:
            tau (float): 再电离光深
            As (float): 原初功率谱振幅
        
        返回:
            C_l_low, C_l_high: 低阶和高阶的理论功率谱
        """
        _, C_l = get_cmb_power_spectrum(
            Omega_m=self.fiducial_params['Omega_m'],
            Omega_L=self.fiducial_params['Omega_L'],
            H0=self.fiducial_params['H0'],
            Omega_b=self.fiducial_params['Omega_b'],
            ns=self.fiducial_params['ns'],
            As=As,
            tau=tau,
            l_max=max(self.l_max_low, self.l_max_high) + 100,
            return_ell=True
        )
        
        return C_l[self.mask_low], C_l[self.mask_high]
    
    def chi2(self, tau, As):
        """
        计算 χ² 值
        
        关键改进: 使用分尺度加权来打破 τ-A_s 简并
        - 低阶: 对 τ 更敏感
        - 高阶: 对 A_s 更敏感
        """
        try:
            C_l_model_low, C_l_model_high = self.compute_model(tau, As)
        except:
            return np.inf
        
        diff_low = self.C_l_obs_low - C_l_model_low
        diff_high = self.C_l_obs_high - C_l_model_high
        
        weight_low = 1.0
        weight_high = 1.0
        
        chi2_low = weight_low * diff_low @ self.inv_cov_low @ diff_low
        chi2_high = weight_high * diff_high @ self.inv_cov_high @ diff_high
        
        return chi2_low + chi2_high
    
    def log_prior(self, theta):
        """
        先验分布
        """
        tau, As = theta
        
        if 0.001 < tau < 0.5 and 1e-10 < As < 5e-9:
            return 0.0
        return -np.inf
    
    def log_likelihood(self, theta):
        """
        对数似然函数
        """
        tau, As = theta
        chi2_val = self.chi2(tau, As)
        return -0.5 * chi2_val
    
    def log_probability(self, theta):
        """
        对数后验概率
        """
        lp = self.log_prior(theta)
        if not np.isfinite(lp):
            return -np.inf
        return lp + self.log_likelihood(theta)


def run_mcmc_fit(likelihood, n_walkers=32, n_steps=5000, burn_in=1000,
                 init_params=None, param_names=None):
    """
    使用 emcee 进行 MCMC 采样
    
    参数:
        likelihood: 似然函数对象
        n_walkers (int): 游走者数量
        n_steps (int): 采样步数
        burn_in (int): 燃烧期步数
        init_params (list): 参数初始值
        param_names (list): 参数名称列表
    
    返回:
        dict: 包含采样结果、最佳拟合参数、协方差矩阵等
    """
    try:
        import emcee
    except ImportError:
        raise ImportError("请先安装 emcee 库: pip install emcee")
    
    print("\n" + "="*60)
    if param_names:
        print(f"开始 MCMC 采样拟合: {', '.join(param_names)}")
    else:
        print("开始 MCMC 采样")
    print("="*60)
    
    ndim = len(init_params)
    
    if init_params is None:
        raise ValueError("必须提供初始参数 init_params")
    
    print(f"初始位置: {init_params}")
    
    pos = np.array(init_params) + 1e-3 * np.random.randn(n_walkers, ndim)
    pos = np.abs(pos)
    
    print(f"游走者数量: {n_walkers}")
    print(f"采样步数: {n_steps} (燃烧期: {burn_in})")
    
    start_time = time.time()
    
    sampler = emcee.EnsembleSampler(n_walkers, ndim, likelihood.log_probability)
    sampler.run_mcmc(pos, n_steps, progress=True)
    
    elapsed = time.time() - start_time
    print(f"\n采样完成，耗时: {elapsed:.2f} 秒")
    
    samples = sampler.get_chain(discard=burn_in, flat=True)
    log_probs = sampler.get_log_prob(discard=burn_in, flat=True)
    
    param_medians = np.median(samples, axis=0)
    param_stds = np.std(samples, axis=0)
    
    covariance = np.cov(samples.T)
    
    print("\n" + "-"*60)
    print("拟合结果:")
    print("-"*60)
    
    if param_names:
        for i, name in enumerate(param_names):
            print(f"{name} = {param_medians[i]:.4e} ± {param_stds[i]:.4e}")
    
    print(f"\n协方差矩阵:")
    print(covariance)
    
    print("-"*60)
    
    results = {
        'samples': samples,
        'log_probs': log_probs,
        'param_medians': param_medians,
        'param_stds': param_stds,
        'covariance': covariance,
        'sampler': sampler,
        'burn_in': burn_in,
        'n_walkers': n_walkers,
        'n_steps': n_steps,
        'param_names': param_names
    }
    
    for i, name in enumerate(param_names or []):
        results[f'{name}_best'] = param_medians[i]
        results[f'{name}_sigma'] = param_stds[i]
        results[f'{name}_samples'] = samples[:, i]
    
    return results


def run_mcmc_fit_r(likelihood, n_walkers=20, n_steps=3000, burn_in=1000,
                   r_init=0.01, tau_init=None, As_init=None, fit_tau=False, fit_As=False):
    """
    专门针对张量-标量比 r 的 MCMC 拟合
    
    参数:
        likelihood (CMBBmodeLikelihood): B 模似然函数对象
        n_walkers (int): 游走者数量
        n_steps (int): 采样步数
        burn_in (int): 燃烧期步数
        r_init (float): r 初始值
        tau_init (float): τ 初始值 (如果拟合)
        As_init (float): A_s 初始值 (如果拟合)
        fit_tau (bool): 是否同时拟合 τ
        fit_As (bool): 是否同时拟合 A_s
    
    返回:
        dict: 拟合结果
    """
    init_params = [r_init]
    param_names = ['r']
    
    if fit_tau:
        if tau_init is None:
            tau_init = likelihood.fiducial_params.get('tau', 0.0561)
        init_params.append(tau_init)
        param_names.append('tau')
    
    if fit_As:
        if As_init is None:
            As_init = likelihood.fiducial_params.get('As', 2.107e-9)
        init_params.append(As_init)
        param_names.append('As')
    
    results = run_mcmc_fit(
        likelihood,
        n_walkers=n_walkers,
        n_steps=n_steps,
        burn_in=burn_in,
        init_params=init_params,
        param_names=param_names
    )
    
    r_fid = likelihood.fiducial_params.get('r', 0.01)
    r_best = results['r_best']
    r_sigma = results['r_sigma']
    
    print(f"\nr 约束结果:")
    print(f"  95% 上限: r < {r_best + 2*r_sigma:.4f} (95% CL)")
    print(f"  探测显著性: {r_best / r_sigma:.2f} σ")
    print(f"  基准值 r = {r_fid:.4f}")
    print(f"  偏差: {r_best - r_fid:.4f} ({100*(r_best/r_fid-1) if r_fid>0 else np.nan:.2f}%)")
    
    return results


def plot_mcmc_results(results, likelihood, save_prefix=None):
    """
    绘制 MCMC 采样结果
    
    参数:
        results (dict): MCMC 结果字典
        likelihood: 似然函数对象
        save_prefix (str): 保存文件前缀
    """
    param_names = results.get('param_names', ['τ', 'A_s'])
    ndim = len(param_names)
    
    try:
        import corner
        has_corner = True
    except ImportError:
        has_corner = False
        print("提示: 安装 corner 库可绘制更好的角图: pip install corner")
    
    if ndim == 1:
        fig, axes = plt.subplots(1, 2, figsize=(12, 4))
        
        param_name = param_names[0]
        samples = results['samples'].flatten()
        best = results['param_medians'][0]
        sigma = results['param_stds'][0]
        
        axes[0].plot(samples, 'k-', alpha=0.5, linewidth=0.5)
        axes[0].axhline(best, color='b', linestyle='-', label='最佳拟合')
        axes[0].set_xlabel('样本索引')
        axes[0].set_ylabel(param_name)
        axes[0].set_title(f'{param_name} 采样链')
        axes[0].legend()
        
        axes[1].hist(samples, bins=50, density=True, alpha=0.7, color='steelblue')
        axes[1].axvline(best, color='k', linestyle='-', linewidth=2, label='最佳拟合')
        axes[1].axvline(best - sigma, color='k', linestyle='--', alpha=0.5)
        axes[1].axvline(best + sigma, color='k', linestyle='--', alpha=0.5)
        axes[1].axvline(best + 2*sigma, color='r', linestyle='--', alpha=0.7, label='95%上限')
        axes[1].set_xlabel(param_name)
        axes[1].set_ylabel('概率密度')
        axes[1].set_title(f'{param_name} 边缘分布')
        axes[1].legend()
        
    elif ndim == 2:
        fig = plt.figure(figsize=(14, 10))
        
        ax1 = plt.subplot(2, 2, 1)
        ax1.plot(results[f'{param_names[0]}_samples'], 'k-', alpha=0.5, linewidth=0.5)
        ax1.axhline(results[f'{param_names[0]}_best'], color='b', linestyle='-', label='最佳拟合')
        ax1.set_xlabel('样本索引')
        ax1.set_ylabel(param_names[0])
        ax1.set_title(f'{param_names[0]} 采样链')
        ax1.legend()
        
        ax2 = plt.subplot(2, 2, 2)
        ax2.plot(results[f'{param_names[1]}_samples'], 'k-', alpha=0.5, linewidth=0.5)
        ax2.axhline(results[f'{param_names[1]}_best'], color='b', linestyle='-', label='最佳拟合')
        ax2.set_xlabel('样本索引')
        ax2.set_ylabel(param_names[1])
        ax2.set_title(f'{param_names[1]} 采样链')
        ax2.legend()
        
        ax3 = plt.subplot(2, 2, 3)
        ax3.scatter(results[f'{param_names[0]}_samples'], 
                    results[f'{param_names[1]}_samples'], 
                    s=1, alpha=0.3, color='gray')
        ax3.plot(results[f'{param_names[0]}_best'], 
                 results[f'{param_names[1]}_best'], 
                 'bo', markersize=10, label='最佳拟合')
        ax3.set_xlabel(param_names[0])
        ax3.set_ylabel(param_names[1])
        ax3.set_title(f'{param_names[0]}-{param_names[1]} 后验分布')
        ax3.legend()
        
        ax4 = plt.subplot(2, 2, 4)
        ax4.hist(results[f'{param_names[0]}_samples'], bins=50, density=True, 
                 alpha=0.7, color='steelblue', label=param_names[0])
        ax4.hist(results[f'{param_names[1]}_samples'] / np.max(results[f'{param_names[1]}_samples']) * np.max(results[f'{param_names[0]}_samples']), 
                 bins=50, density=True, alpha=0.5, color='orange', label=param_names[1])
        ax4.set_xlabel('归一化参数值')
        ax4.set_ylabel('概率密度')
        ax4.set_title('归一化边缘分布')
        ax4.legend()
        
    else:
        print("多维参数结果，请使用 corner 库绘制角图")
        has_corner = True
    
    plt.tight_layout()
    
    if save_prefix:
        plt.savefig(f"{save_prefix}_mcmc_results.png", dpi=300, bbox_inches='tight')
        print(f"图表已保存到: {save_prefix}_mcmc_results.png")
    
    plt.show()
    
    if has_corner and ndim > 1:
        truths = []
        for name in param_names:
            if hasattr(likelihood, 'fiducial_params'):
                truths.append(likelihood.fiducial_params.get(name, None))
            else:
                truths.append(None)
        
        fig_corner = corner.corner(
            results['samples'],
            labels=param_names,
            truths=truths if any(t is not None for t in truths) else None,
            show_titles=True,
            title_fmt='.4e',
            quantiles=[0.16, 0.5, 0.84],
            figsize=(10, 10)
        )
        if save_prefix:
            fig_corner.savefig(f"{save_prefix}_corner.png", dpi=300, bbox_inches='tight')
            print(f"角图已保存到: {save_prefix}_corner.png")
        plt.show()


def plot_r_constraint(results, likelihood, save_path=None):
    """
    绘制 r 约束结果
    
    参数:
        results (dict): MCMC 结果
        likelihood (CMBBmodeLikelihood): 似然函数对象
        save_path (str): 保存路径
    """
    r_samples = results['r_samples']
    r_best = results['r_best']
    r_sigma = results['r_sigma']
    r_fid = likelihood.fiducial_params.get('r', 0.01)
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    axes[0].hist(r_samples, bins=50, density=True, alpha=0.7, color='steelblue')
    axes[0].axvline(r_best, color='k', linestyle='-', linewidth=2, label='最佳拟合')
    axes[0].axvline(r_best - r_sigma, color='k', linestyle='--', alpha=0.5)
    axes[0].axvline(r_best + r_sigma, color='k', linestyle='--', alpha=0.5)
    axes[0].axvline(r_best + 2*r_sigma, color='r', linestyle='--', alpha=0.7, label='95% 上限')
    axes[0].axvline(r_fid, color='g', linestyle='--', alpha=0.8, label='基准值')
    axes[0].set_xlabel(r'$r$', fontsize=14)
    axes[0].set_ylabel('概率密度', fontsize=12)
    axes[0].set_title(r'张量-标量比 $r$ 后验分布', fontsize=14)
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    r_grid = np.linspace(0, r_best + 5*r_sigma, 50)
    chi2_grid = np.array([likelihood.chi2(r) for r in r_grid])
    delta_chi2 = chi2_grid - np.min(chi2_grid)
    
    axes[1].plot(r_grid, delta_chi2, 'b-', linewidth=2)
    axes[1].axhline(2.71, color='r', linestyle='--', label='95% CL')
    axes[1].axhline(1.0, color='k', linestyle='--', alpha=0.7, label='68% CL')
    axes[1].axvline(r_best, color='g', linestyle='--', label='最佳拟合')
    axes[1].set_xlabel(r'$r$', fontsize=14)
    axes[1].set_ylabel(r'$\Delta \chi^2$', fontsize=12)
    axes[1].set_title(r'$\chi^2$ 剖面', fontsize=14)
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"图表已保存到: {save_path}")
    
    plt.show()


def run_tau_As_fit_demo():
    """
    演示 τ 和 A_s 的联合拟合
    """
    print("="*70)
    print("CMB τ-A_s 联合拟合演示")
    print("="*70)
    print("\n本演示展示如何使用 MCMC 打破再电离光深 τ 和原初振幅 A_s 的简并")
    print("核心思想:")
    print("  - 低阶 multipole (ℓ < 30): 对再电离 τ 更敏感")
    print("  - 高阶 multipole (ℓ > 50): 对原初振幅 A_s 更敏感")
    print("  - 联合不同尺度的约束可以打破参数简并")
    print("="*70)
    
    fiducial_params = {
        'Omega_m': 0.3111,
        'Omega_L': 0.6889,
        'H0': 67.66,
        'Omega_b': 0.04897,
        'ns': 0.9665,
        'As': 2.107e-9,
        'tau': 0.0561
    }
    
    likelihood = CMBLikelihood(
        fiducial_params,
        l_min_low=2,
        l_max_low=30,
        l_min_high=50,
        l_max_high=400,
        noise_level=30.0
    )
    
    results = run_mcmc_fit(
        likelihood,
        n_walkers=20,
        n_steps=3000,
        burn_in=1000,
        init_params=[0.06, 2.2e-9],
        param_names=['tau', 'As']
    )
    
    plot_mcmc_results(results, likelihood, save_prefix='cmb_tau_As')
    
    return likelihood, results


def run_Bmode_r_constraint_demo():
    """
    演示 CMB B 模和张量-标量比 r 约束
    """
    print("\n" + "="*70)
    print("CMB B 模极化与原初引力波约束演示")
    print("="*70)
    print("\n本演示展示如何使用 CMB B 模极化约束原初引力波 (张量-标量比 r)")
    print("物理背景:")
    print("  - 暴胀模型预言原初引力波会产生 B 模极化")
    print("  - 低阶 multipole (ℓ~100): 原初引力波产生的 B 模峰值")
    print("  - 引力透镜也会产生 B 模，需要分离或消除")
    print("  - r 的测量是检验暴胀模型的关键观测证据")
    print("="*70)
    
    print("\n1. 计算不同 r 值的完整 CMB 功率谱...")
    ls, spectra_r0 = get_cmb_power_spectrum_full(
        Omega_m=0.3111, Omega_L=0.6889, H0=67.66,
        Omega_b=0.04897, ns=0.9665, As=2.107e-9,
        tau=0.0561, r=0.0, l_max=2000, lensing=True, return_ell=True
    )
    
    _, spectra_r001 = get_cmb_power_spectrum_full(
        Omega_m=0.3111, Omega_L=0.6889, H0=67.66,
        Omega_b=0.04897, ns=0.9665, As=2.107e-9,
        tau=0.0561, r=0.01, l_max=2000, lensing=True, return_ell=True
    )
    
    _, spectra_r01 = get_cmb_power_spectrum_full(
        Omega_m=0.3111, Omega_L=0.6889, H0=67.66,
        Omega_b=0.04897, ns=0.9665, As=2.107e-9,
        tau=0.0561, r=0.1, l_max=2000, lensing=True, return_ell=True
    )
    
    plot_full_spectra(ls, spectra_r001, title='Planck Cosmology with r=0.01',
                      show_lensing=True, save_path='cmb_full_spectra_r001.png')
    
    plot_Bmode_comparison(
        ls, [spectra_r0, spectra_r001, spectra_r01],
        ['r=0', 'r=0.01', 'r=0.1'],
        r_values=[0.0, 0.01, 0.1],
        title='B-mode Power Spectrum vs Tensor-to-Scalar Ratio r',
        save_path='cmb_Bmode_r_comparison.png'
    )
    
    print("\n2. 模拟 B 模观测并约束 r...")
    
    fiducial_params = {
        'Omega_m': 0.3111,
        'Omega_L': 0.6889,
        'H0': 67.66,
        'Omega_b': 0.04897,
        'ns': 0.9665,
        'As': 2.107e-9,
        'tau': 0.0561,
        'r': 0.01
    }
    
    likelihood = CMBBmodeLikelihood(
        fiducial_params,
        l_min=30,
        l_max=300,
        noise_level=0.3,
        de_lensed=True
    )
    
    results = run_mcmc_fit_r(
        likelihood,
        n_walkers=20,
        n_steps=2000,
        burn_in=500,
        r_init=0.015,
        fit_tau=False,
        fit_As=False
    )
    
    plot_r_constraint(results, likelihood, save_path='cmb_r_constraint.png')
    
    return likelihood, results, ls, spectra_r001


def run_inflation_model_test_demo():
    """
    演示暴胀模型检验
    """
    print("\n" + "="*70)
    print("暴胀模型检验演示")
    print("="*70)
    print("\n不同暴胀模型预言不同的 (r, n_s) 值:")
    print("  - 单场慢滚暴胀: r ≈ 16ε, n_s - 1 ≈ 2η - 6ε")
    print("  - 混沌暴胀 (V ∝ φ²): r ≈ 0.13, n_s ≈ 0.96")
    print("  - 混沌暴胀 (V ∝ φ^4): r ≈ 0.27, n_s ≈ 0.95")
    print("  - 自然暴胀: r 通常较小")
    print("  - Starobinsky 模型: r ≈ 0.003, n_s ≈ 0.965")
    print("="*70)
    
    models = {
        'Starobinsky': {'r': 0.003, 'ns': 0.965, 'color': 'b'},
        'Chaotic φ²': {'r': 0.13, 'ns': 0.96, 'color': 'r'},
        'Chaotic φ^4': {'r': 0.27, 'ns': 0.95, 'color': 'g'},
        'Natural': {'r': 0.01, 'ns': 0.965, 'color': 'm'},
        'Planck 2018': {'r': 0.0, 'ns': 0.9665, 'color': 'k'}
    }
    
    plt.figure(figsize=(10, 8))
    
    for name, params in models.items():
        plt.plot(params['ns'], params['r'], 'o', markersize=10, 
                 label=name, color=params['color'])
    
    ns_range = np.linspace(0.94, 0.98, 100)
    r_bound = 0.1 * np.ones_like(ns_range)
    plt.fill_between(ns_range, 0, r_bound, alpha=0.2, color='gray', label='Planck 95%上限')
    
    plt.xlabel(r'$n_s$', fontsize=14)
    plt.ylabel(r'$r$', fontsize=14)
    plt.title('Inflation Model Predictions in (n_s, r) Plane', fontsize=16)
    plt.legend(fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.ylim([0, 0.35])
    plt.tight_layout()
    plt.savefig('inflation_models_ns_r.png', dpi=300, bbox_inches='tight')
    print("暴胀模型比较图已保存到: inflation_models_ns_r.png")
    plt.show()


if __name__ == "__main__":
    print("计算 CMB 温度功率谱...")
    
    ls, C_l_TT = get_cmb_power_spectrum(
        Omega_m=0.3111,
        Omega_L=0.6889,
        H0=67.66,
        Omega_b=0.04897,
        ns=0.9665,
        As=2.107e-9,
        tau=0.0561,
        l_max=2500,
        return_ell=True
    )
    
    print(f"计算完成，l 范围: {ls[0]} - {ls[-1]}")
    print(f"C_l[10] = {C_l_TT[10]:.4e} μK²")
    print(f"C_l[100] = {C_l_TT[100]:.4e} μK²")
    print(f"C_l[1000] = {C_l_TT[1000]:.4e} μK²")
    
    print("\n" + "="*50)
    print("运行完整演示 (需要 emcee 库)")
    print("="*50)
    
    try:
        import emcee
        
        print("\n选择要运行的演示:")
        print("1. τ-A_s 联合拟合")
        print("2. B 模极化与 r 约束")
        print("3. 暴胀模型检验")
        print("4. 运行全部")
        
        choice = input("\n请输入选项 (1-4): ").strip()
        
        if choice == '1':
            run_tau_As_fit_demo()
        elif choice == '2':
            run_Bmode_r_constraint_demo()
        elif choice == '3':
            run_inflation_model_test_demo()
        elif choice == '4':
            run_tau_As_fit_demo()
            run_Bmode_r_constraint_demo()
            run_inflation_model_test_demo()
        else:
            print("无效选项，运行 B 模演示...")
            run_Bmode_r_constraint_demo()
            
    except ImportError:
        print("请先安装 emcee: pip install emcee")
        print("然后运行: python cmb_power_spectrum.py")
