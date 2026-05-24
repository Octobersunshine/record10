import numpy as np
from propagation import propagate, back_propagate
from phase_correction import PhaseReference, PhaseCorrector


class GerchbergSaxton:
    """
    Gerchberg-Saxton (GS) 相位恢复算法
    
    用于从衍射强度图重建复振幅分布
    """
    
    def __init__(self, wavelength, pixel_size, distance, 
                 method='angular_spectrum', 
                 support=None, 
                 positivity=True,
                 phase_corrector=None,
                 phase_reference_type=None):
        """
        初始化GS算法
        
        参数:
            wavelength: 波长
            pixel_size: 像素大小
            distance: 传播距离
            method: 传播方法 ('angular_spectrum', 'fresnel', 'fraunhofer')
            support: 支撑区域（物体所在区域的掩码），None表示全区域
            positivity: 是否强制振幅为正
            phase_corrector: PhaseCorrector对象，用于相位校正
            phase_reference: 相位基准类型，用于自动创建相位校正器
                None表示不进行相位校正
                'support' - 支撑域相位基准
                'point' - 中心点相位基准
                'amplitude' - 振幅加权相位基准
                'region' - 中心区域相位基准
        """
        self.wavelength = wavelength
        self.pixel_size = pixel_size
        self.distance = distance
        self.method = method
        self.support = support
        self.positivity = positivity
        self.errors = []
        self.phase_corrector = phase_corrector
        self.phase_reference_type = phase_reference_type
        self._init_phase_corrector()
        
    def _init_phase_corrector(self):
        """
        初始化相位校正器
        """
        if self.phase_corrector is not None:
            return
        
        if self.phase_reference_type is None:
            self.phase_corrector = None
            return
        
        reference = None
        if self.phase_reference_type == 'support' and self.support is not None:
            reference = PhaseReference('support', support=self.support, phase=0.0)
        elif self.phase_reference_type == 'point':
            reference = PhaseReference('point', phase=0.0)
        elif self.phase_reference_type == 'amplitude':
            reference = PhaseReference('amplitude_weighted', threshold=0.3)
        elif self.phase_reference_type == 'region':
            reference = PhaseReference('region', size=5, phase=0.0)
        
        if reference is not None:
            self.phase_corrector = PhaseCorrector(
                global_phase=True,
                phase_tilt=False,
                reference=reference
            )
        
    def _apply_phase_correction(self, field):
        """
        应用相位校正
        """
        if self.phase_corrector is not None:
            return self.phase_corrector.apply(field)
        return field
        
    def apply_object_constraint(self, field):
        """
        应用物体平面约束
        """
        if self.support is not None:
            field = field * self.support
            
        if self.positivity:
            amplitude = np.abs(field)
            phase = np.angle(field)
            amplitude = np.maximum(amplitude, 0)
            field = amplitude * np.exp(1j * phase)
            
        return field
    
    def apply_diffraction_constraint(self, field, measured_intensity):
        """
        应用衍射平面约束（替换振幅）
        """
        phase = np.angle(field)
        amplitude = np.sqrt(np.maximum(measured_intensity, 0))
        return amplitude * np.exp(1j * phase)
    
    def reconstruct(self, measured_intensity, 
                    initial_guess=None, 
                    num_iterations=100, 
                    verbose=True):
        """
        执行相位恢复
        
        参数:
            measured_intensity: 测得的衍射强度图
            initial_guess: 初始猜测的复振幅，None表示随机初始化
            num_iterations: 迭代次数
            verbose: 是否打印进度
        
        返回:
            重建的复振幅分布
        """
        ny, nx = measured_intensity.shape
        
        if initial_guess is None:
            random_phase = np.random.rand(ny, nx) * 2 * np.pi
            initial_amplitude = np.sqrt(np.mean(measured_intensity))
            object_field = initial_amplitude * np.exp(1j * random_phase)
        else:
            object_field = initial_guess.copy()
        
        self.errors = []
        
        for iteration in range(num_iterations):
            diffraction_field, _ = propagate(
                object_field, self.wavelength, self.pixel_size, 
                self.distance, self.method
            )
            
            error = np.mean((np.abs(diffraction_field)**2 - measured_intensity)**2)
            self.errors.append(error)
            
            if verbose and (iteration + 1) % 10 == 0:
                print(f"Iteration {iteration + 1}/{num_iterations}, Error: {error:.6e}")
            
            diffraction_constrained = self.apply_diffraction_constraint(
                diffraction_field, measured_intensity
            )
            
            object_field, _ = back_propagate(
                diffraction_constrained, self.wavelength, self.pixel_size,
                self.distance, self.method
            )
            
            object_field = self.apply_object_constraint(object_field)
            
            object_field = self._apply_phase_correction(object_field)
        
        return object_field
    
    def get_errors(self):
        """
        获取迭代误差历史
        """
        return np.array(self.errors)


class HybridInputOutput(GerchbergSaxton):
    """
    Hybrid Input-Output (HIO) 算法
    GS算法的改进版本，具有更好的收敛性能
    """
    
    def __init__(self, wavelength, pixel_size, distance,
                 method='angular_spectrum',
                 support=None,
                 positivity=True,
                 beta=0.9,
                 phase_corrector=None,
                 phase_reference_type=None):
        """
        初始化HIO算法
        
        参数:
            beta: HIO反馈参数，通常在0.5-0.99之间
            phase_corrector: PhaseCorrector对象，用于相位校正
            phase_reference_type: 相位基准类型
        """
        super().__init__(wavelength, pixel_size, distance, method, support, positivity,
                         phase_corrector, phase_reference_type)
        self.beta = beta
        
    def apply_object_constraint(self, field, previous_field=None):
        """
        HIO的物体平面约束（包含反馈项）
        """
        if self.support is not None:
            support_mask = self.support
        else:
            support_mask = np.ones_like(np.abs(field), dtype=bool)
        
        outside_support = ~support_mask
        
        if previous_field is not None:
            field[outside_support] = previous_field[outside_support] - self.beta * field[outside_support]
        else:
            field[outside_support] = 0
        
        if self.positivity:
            amplitude = np.abs(field)
            phase = np.angle(field)
            amplitude[support_mask] = np.maximum(amplitude[support_mask], 0)
            field = amplitude * np.exp(1j * phase)
            
        return field
    
    def reconstruct(self, measured_intensity,
                    initial_guess=None,
                    num_iterations=100,
                    verbose=True):
        """
        执行HIO相位恢复
        """
        ny, nx = measured_intensity.shape
        
        if initial_guess is None:
            random_phase = np.random.rand(ny, nx) * 2 * np.pi
            initial_amplitude = np.sqrt(np.mean(measured_intensity))
            object_field = initial_amplitude * np.exp(1j * random_phase)
        else:
            object_field = initial_guess.copy()
        
        self.errors = []
        previous_object = object_field.copy()
        
        for iteration in range(num_iterations):
            diffraction_field, _ = propagate(
                object_field, self.wavelength, self.pixel_size,
                self.distance, self.method
            )
            
            error = np.mean((np.abs(diffraction_field)**2 - measured_intensity)**2)
            self.errors.append(error)
            
            if verbose and (iteration + 1) % 10 == 0:
                print(f"Iteration {iteration + 1}/{num_iterations}, Error: {error:.6e}")
            
            diffraction_constrained = self.apply_diffraction_constraint(
                diffraction_field, measured_intensity
            )
            
            object_field_new, _ = back_propagate(
                diffraction_constrained, self.wavelength, self.pixel_size,
                self.distance, self.method
            )
            
            object_field = self.apply_object_constraint(object_field_new, previous_object)
            previous_object = object_field_new.copy()
            
            object_field = self._apply_phase_correction(object_field)
        
        return object_field


class ErrorReduction(GerchbergSaxton):
    """
    Error Reduction (ER) 算法
    与GS算法类似，但严格应用约束
    """
    
    def apply_object_constraint(self, field):
        """
        ER的物体平面约束
        """
        if self.support is not None:
            field = field * self.support
        else:
            field = field
        
        if self.positivity:
            amplitude = np.abs(field)
            phase = np.angle(field)
            amplitude = np.maximum(amplitude, 0)
            field = amplitude * np.exp(1j * phase)
            
        return field
