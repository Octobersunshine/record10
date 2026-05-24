import numpy as np
from propagation import propagate, back_propagate
from phase_correction import PhaseReference, PhaseCorrector


class DifferenceMap:
    """
    Difference Map (DM) 相位恢复算法
    
    一种基于投影的迭代相位恢复算法，通常比GS/HIO具有更好的收敛性能
    """
    
    def __init__(self, wavelength, pixel_size, distance,
                 method='angular_spectrum',
                 support=None,
                 positivity=True,
                 beta=0.9,
                 phase_corrector=None,
                 phase_reference_type=None):
        """
        初始化DM算法
        
        参数:
            wavelength: 波长
            pixel_size: 像素大小
            distance: 传播距离
            method: 传播方法
            support: 支撑区域掩码
            positivity: 是否强制振幅为正
            beta: DM算法参数，通常取0.9
            phase_corrector: PhaseCorrector对象，用于相位校正
            phase_reference_type: 相位基准类型
        """
        self.wavelength = wavelength
        self.pixel_size = pixel_size
        self.distance = distance
        self.method = method
        self.support = support
        self.positivity = positivity
        self.beta = beta
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
        
    def P_object(self, field):
        """
        物体平面约束投影
        """
        result = field.copy()
        
        if self.support is not None:
            result = result * self.support
            
        if self.positivity:
            amplitude = np.abs(result)
            phase = np.angle(result)
            amplitude = np.maximum(amplitude, 0)
            result = amplitude * np.exp(1j * phase)
            
        return result
    
    def P_diffraction(self, field, measured_intensity):
        """
        衍射平面约束投影
        """
        diffraction_field, _ = propagate(
            field, self.wavelength, self.pixel_size,
            self.distance, self.method
        )
        
        phase = np.angle(diffraction_field)
        amplitude = np.sqrt(np.maximum(measured_intensity, 0))
        diffraction_constrained = amplitude * np.exp(1j * phase)
        
        result, _ = back_propagate(
            diffraction_constrained, self.wavelength, self.pixel_size,
            self.distance, self.method
        )
        
        return result
    
    def R_object(self, field):
        """
        物体平面反射算子
        """
        return 2 * self.P_object(field) - field
    
    def R_diffraction(self, field, measured_intensity):
        """
        衍射平面反射算子
        """
        return 2 * self.P_diffraction(field, measured_intensity) - field
    
    def reconstruct(self, measured_intensity,
                    initial_guess=None,
                    num_iterations=100,
                    verbose=True):
        """
        执行DM相位恢复
        
        参数:
            measured_intensity: 测得的衍射强度图
            initial_guess: 初始猜测
            num_iterations: 迭代次数
            verbose: 是否打印进度
        
        返回:
            重建的复振幅分布
        """
        ny, nx = measured_intensity.shape
        
        if initial_guess is None:
            random_phase = np.random.rand(ny, nx) * 2 * np.pi
            initial_amplitude = np.sqrt(np.mean(measured_intensity))
            x = initial_amplitude * np.exp(1j * random_phase)
        else:
            x = initial_guess.copy()
        
        self.errors = []
        
        for iteration in range(num_iterations):
            x_prev = x.copy()
            
            Ro_x = self.R_object(x)
            x = x + self.beta * (self.P_diffraction(Ro_x, measured_intensity) - self.P_object(x))
            
            Rd_x = self.R_diffraction(x, measured_intensity)
            x = x + self.beta * (self.P_object(Rd_x) - self.P_diffraction(x, measured_intensity))
            
            x = self._apply_phase_correction(x)
            
            diffraction_field, _ = propagate(
                x, self.wavelength, self.pixel_size,
                self.distance, self.method
            )
            error = np.mean((np.abs(diffraction_field)**2 - measured_intensity)**2)
            self.errors.append(error)
            
            if verbose and (iteration + 1) % 10 == 0:
                print(f"Iteration {iteration + 1}/{num_iterations}, Error: {error:.6e}")
        
        return self.P_object(x)
    
    def get_errors(self):
        return np.array(self.errors)


class RelaxedAveragedAlternatingProjections:
    """
    Relaxed Averaged Alternating Projections (RAAR) 算法
    
    结合了HIO和ER的优点，具有良好的收敛性能
    """
    
    def __init__(self, wavelength, pixel_size, distance,
                 method='angular_spectrum',
                 support=None,
                 positivity=True,
                 beta=0.8,
                 phase_corrector=None,
                 phase_reference_type=None):
        """
        初始化RAAR算法
        
        参数:
            beta: RAAR参数，0 < beta < 1
                  beta -> 0 接近HIO
                  beta -> 1 接近ER
            phase_corrector: PhaseCorrector对象，用于相位校正
            phase_reference_type: 相位基准类型
        """
        self.wavelength = wavelength
        self.pixel_size = pixel_size
        self.distance = distance
        self.method = method
        self.support = support
        self.positivity = positivity
        self.beta = beta
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
        
    def P_object(self, field):
        """
        物体平面约束投影
        """
        result = field.copy()
        
        if self.support is not None:
            result = result * self.support
            
        if self.positivity:
            amplitude = np.abs(result)
            phase = np.angle(result)
            amplitude = np.maximum(amplitude, 0)
            result = amplitude * np.exp(1j * phase)
            
        return result
    
    def P_diffraction(self, field, measured_intensity):
        """
        衍射平面约束投影（返回到物体平面）
        """
        diffraction_field, _ = propagate(
            field, self.wavelength, self.pixel_size,
            self.distance, self.method
        )
        
        phase = np.angle(diffraction_field)
        amplitude = np.sqrt(np.maximum(measured_intensity, 0))
        diffraction_constrained = amplitude * np.exp(1j * phase)
        
        result, _ = back_propagate(
            diffraction_constrained, self.wavelength, self.pixel_size,
            self.distance, self.method
        )
        
        return result
    
    def reconstruct(self, measured_intensity,
                    initial_guess=None,
                    num_iterations=100,
                    verbose=True):
        """
        执行RAAR相位恢复
        """
        ny, nx = measured_intensity.shape
        
        if initial_guess is None:
            random_phase = np.random.rand(ny, nx) * 2 * np.pi
            initial_amplitude = np.sqrt(np.mean(measured_intensity))
            x = initial_amplitude * np.exp(1j * random_phase)
        else:
            x = initial_guess.copy()
        
        self.errors = []
        x_prev = x.copy()
        
        for iteration in range(num_iterations):
            y = self.P_diffraction(x, measured_intensity)
            
            x_new = self.beta * self.P_object(2 * y - x) + (1 - self.beta) * y
            
            x_prev = x.copy()
            x = x_new
            
            x = self._apply_phase_correction(x)
            
            diffraction_field, _ = propagate(
                x, self.wavelength, self.pixel_size,
                self.distance, self.method
            )
            error = np.mean((np.abs(diffraction_field)**2 - measured_intensity)**2)
            self.errors.append(error)
            
            if verbose and (iteration + 1) % 10 == 0:
                print(f"Iteration {iteration + 1}/{num_iterations}, Error: {error:.6e}")
        
        return self.P_object(x)
    
    def get_errors(self):
        return np.array(self.errors)


class HybridProjections:
    """
    混合投影算法 - 在迭代过程中从HIO切换到ER
    """
    
    def __init__(self, wavelength, pixel_size, distance,
                 method='angular_spectrum',
                 support=None,
                 positivity=True,
                 beta=0.9,
                 switch_iteration=50,
                 phase_corrector=None,
                 phase_reference_type=None):
        """
        初始化混合投影算法
        
        参数:
            beta: HIO阶段的beta参数
            switch_iteration: 从HIO切换到ER的迭代次数
            phase_corrector: PhaseCorrector对象，用于相位校正
            phase_reference_type: 相位基准类型
        """
        self.wavelength = wavelength
        self.pixel_size = pixel_size
        self.distance = distance
        self.method = method
        self.support = support
        self.positivity = positivity
        self.beta = beta
        self.switch_iteration = switch_iteration
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
        
    def P_object(self, field):
        result = field.copy()
        
        if self.support is not None:
            result = result * self.support
            
        if self.positivity:
            amplitude = np.abs(result)
            phase = np.angle(result)
            amplitude = np.maximum(amplitude, 0)
            result = amplitude * np.exp(1j * phase)
            
        return result
    
    def P_diffraction(self, field, measured_intensity):
        diffraction_field, _ = propagate(
            field, self.wavelength, self.pixel_size,
            self.distance, self.method
        )
        
        phase = np.angle(diffraction_field)
        amplitude = np.sqrt(np.maximum(measured_intensity, 0))
        diffraction_constrained = amplitude * np.exp(1j * phase)
        
        result, _ = back_propagate(
            diffraction_constrained, self.wavelength, self.pixel_size,
            self.distance, self.method
        )
        
        return result
    
    def reconstruct(self, measured_intensity,
                    initial_guess=None,
                    num_iterations=100,
                    verbose=True):
        """
        执行混合投影相位恢复
        前switch_iteration次使用HIO，之后使用ER
        """
        ny, nx = measured_intensity.shape
        
        if initial_guess is None:
            random_phase = np.random.rand(ny, nx) * 2 * np.pi
            initial_amplitude = np.sqrt(np.mean(measured_intensity))
            x = initial_amplitude * np.exp(1j * random_phase)
        else:
            x = initial_guess.copy()
        
        self.errors = []
        x_prev = x.copy()
        
        for iteration in range(num_iterations):
            y = self.P_diffraction(x, measured_intensity)
            
            if iteration < self.switch_iteration:
                if self.support is not None:
                    support_mask = self.support
                else:
                    support_mask = np.ones_like(np.abs(x), dtype=bool)
                
                outside_support = ~support_mask
                
                x_new = y.copy()
                x_new[outside_support] = x[outside_support] - self.beta * y[outside_support]
                
                if self.positivity:
                    amplitude = np.abs(x_new)
                    phase = np.angle(x_new)
                    amplitude[support_mask] = np.maximum(amplitude[support_mask], 0)
                    x_new = amplitude * np.exp(1j * phase)
            else:
                x_new = self.P_object(y)
            
            x_prev = x.copy()
            x = x_new
            
            x = self._apply_phase_correction(x)
            
            diffraction_field, _ = propagate(
                x, self.wavelength, self.pixel_size,
                self.distance, self.method
            )
            error = np.mean((np.abs(diffraction_field)**2 - measured_intensity)**2)
            self.errors.append(error)
            
            if verbose and (iteration + 1) % 10 == 0:
                print(f"Iteration {iteration + 1}/{num_iterations}, Error: {error:.6e}")
        
        return self.P_object(x)
    
    def get_errors(self):
        return np.array(self.errors)
