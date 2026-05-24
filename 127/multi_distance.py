import numpy as np
from propagation import propagate, back_propagate
from phase_correction import PhaseReference, PhaseCorrector


class MultiDistanceGS:
    """
    多距离GS相位恢复算法
    
    利用多个不同传播距离的衍射强度图进行相位恢复
    """
    
    def __init__(self, wavelength, pixel_size, distances,
                 method='angular_spectrum',
                 support=None,
                 positivity=True,
                 weights=None,
                 phase_corrector=None,
                 phase_reference_type=None):
        """
        初始化多距离GS算法
        
        参数:
            wavelength: 波长
            pixel_size: 像素大小
            distances: 传播距离列表 [d1, d2, ...]
            method: 传播方法
            support: 支撑区域掩码
            positivity: 是否强制振幅为正
            weights: 各距离的权重，None表示等权重
            phase_corrector: PhaseCorrector对象，用于相位校正
            phase_reference_type: 相位基准类型
        """
        self.wavelength = wavelength
        self.pixel_size = pixel_size
        self.distances = distances
        self.method = method
        self.support = support
        self.positivity = positivity
        
        if weights is None:
            self.weights = np.ones(len(distances)) / len(distances)
        else:
            self.weights = np.array(weights) / np.sum(weights)
            
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
        result = field.copy()
        
        if self.support is not None:
            result = result * self.support
            
        if self.positivity:
            amplitude = np.abs(result)
            phase = np.angle(result)
            amplitude = np.maximum(amplitude, 0)
            result = amplitude * np.exp(1j * phase)
            
        return result
    
    def apply_diffraction_constraint(self, field, measured_intensity, distance):
        """
        应用衍射平面约束并返回物体平面
        """
        diffraction_field, _ = propagate(
            field, self.wavelength, self.pixel_size,
            distance, self.method
        )
        
        phase = np.angle(diffraction_field)
        amplitude = np.sqrt(np.maximum(measured_intensity, 0))
        diffraction_constrained = amplitude * np.exp(1j * phase)
        
        result, _ = back_propagate(
            diffraction_constrained, self.wavelength, self.pixel_size,
            distance, self.method
        )
        
        return result
    
    def reconstruct(self, measured_intensities,
                    initial_guess=None,
                    num_iterations=100,
                    verbose=True):
        """
        执行多距离相位恢复
        
        参数:
            measured_intensities: 衍射强度图列表 [I1, I2, ...]
            initial_guess: 初始猜测
            num_iterations: 迭代次数
            verbose: 是否打印进度
        
        返回:
            重建的复振幅分布
        """
        assert len(measured_intensities) == len(self.distances), \
            "强度图数量必须与距离数量相等"
        
        ny, nx = measured_intensities[0].shape
        
        if initial_guess is None:
            random_phase = np.random.rand(ny, nx) * 2 * np.pi
            initial_amplitude = np.sqrt(np.mean(measured_intensities[0]))
            object_field = initial_amplitude * np.exp(1j * random_phase)
        else:
            object_field = initial_guess.copy()
        
        self.errors = []
        
        for iteration in range(num_iterations):
            updated_fields = []
            
            for idx, (intensity, distance) in enumerate(zip(measured_intensities, self.distances)):
                constrained = self.apply_diffraction_constraint(
                    object_field, intensity, distance
                )
                updated_fields.append(constrained)
            
            object_field = np.zeros_like(object_field, dtype=complex)
            for idx, field in enumerate(updated_fields):
                object_field += self.weights[idx] * field
            
            object_field = self.apply_object_constraint(object_field)
            
            object_field = self._apply_phase_correction(object_field)
            
            total_error = 0
            for idx, (intensity, distance) in enumerate(zip(measured_intensities, self.distances)):
                diffraction_field, _ = propagate(
                    object_field, self.wavelength, self.pixel_size,
                    distance, self.method
                )
                error = np.mean((np.abs(diffraction_field)**2 - intensity)**2)
                total_error += self.weights[idx] * error
            
            self.errors.append(total_error)
            
            if verbose and (iteration + 1) % 10 == 0:
                print(f"Iteration {iteration + 1}/{num_iterations}, Weighted Error: {total_error:.6e}")
        
        return object_field
    
    def get_errors(self):
        return np.array(self.errors)


class MultiDistanceHIO:
    """
    多距离HIO相位恢复算法
    """
    
    def __init__(self, wavelength, pixel_size, distances,
                 method='angular_spectrum',
                 support=None,
                 positivity=True,
                 beta=0.9,
                 weights=None,
                 phase_corrector=None,
                 phase_reference_type=None):
        self.wavelength = wavelength
        self.pixel_size = pixel_size
        self.distances = distances
        self.method = method
        self.support = support
        self.positivity = positivity
        self.beta = beta
        
        if weights is None:
            self.weights = np.ones(len(distances)) / len(distances)
        else:
            self.weights = np.array(weights) / np.sum(weights)
            
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
        
    def apply_object_constraint(self, field, previous_field=None):
        """
        HIO物体平面约束
        """
        if self.support is not None:
            support_mask = self.support
        else:
            support_mask = np.ones_like(np.abs(field), dtype=bool)
        
        outside_support = ~support_mask
        
        result = field.copy()
        
        if previous_field is not None:
            result[outside_support] = previous_field[outside_support] - self.beta * field[outside_support]
        else:
            result[outside_support] = 0
        
        if self.positivity:
            amplitude = np.abs(result)
            phase = np.angle(result)
            amplitude[support_mask] = np.maximum(amplitude[support_mask], 0)
            result = amplitude * np.exp(1j * phase)
            
        return result
    
    def apply_diffraction_constraint(self, field, measured_intensity, distance):
        diffraction_field, _ = propagate(
            field, self.wavelength, self.pixel_size,
            distance, self.method
        )
        
        phase = np.angle(diffraction_field)
        amplitude = np.sqrt(np.maximum(measured_intensity, 0))
        diffraction_constrained = amplitude * np.exp(1j * phase)
        
        result, _ = back_propagate(
            diffraction_constrained, self.wavelength, self.pixel_size,
            distance, self.method
        )
        
        return result
    
    def reconstruct(self, measured_intensities,
                    initial_guess=None,
                    num_iterations=100,
                    verbose=True):
        assert len(measured_intensities) == len(self.distances)
        
        ny, nx = measured_intensities[0].shape
        
        if initial_guess is None:
            random_phase = np.random.rand(ny, nx) * 2 * np.pi
            initial_amplitude = np.sqrt(np.mean(measured_intensities[0]))
            object_field = initial_amplitude * np.exp(1j * random_phase)
        else:
            object_field = initial_guess.copy()
        
        self.errors = []
        previous_object = object_field.copy()
        
        for iteration in range(num_iterations):
            updated_fields = []
            
            for idx, (intensity, distance) in enumerate(zip(measured_intensities, self.distances)):
                constrained = self.apply_diffraction_constraint(
                    object_field, intensity, distance
                )
                updated_fields.append(constrained)
            
            avg_field = np.zeros_like(object_field, dtype=complex)
            for idx, field in enumerate(updated_fields):
                avg_field += self.weights[idx] * field
            
            object_field = self.apply_object_constraint(avg_field, previous_object)
            previous_object = avg_field.copy()
            
            object_field = self._apply_phase_correction(object_field)
            
            total_error = 0
            for idx, (intensity, distance) in enumerate(zip(measured_intensities, self.distances)):
                diffraction_field, _ = propagate(
                    object_field, self.wavelength, self.pixel_size,
                    distance, self.method
                )
                error = np.mean((np.abs(diffraction_field)**2 - intensity)**2)
                total_error += self.weights[idx] * error
            
            self.errors.append(total_error)
            
            if verbose and (iteration + 1) % 10 == 0:
                print(f"Iteration {iteration + 1}/{num_iterations}, Weighted Error: {total_error:.6e}")
        
        return object_field
    
    def get_errors(self):
        return np.array(self.errors)


class MultiAngleGS:
    """
    多角度GS相位恢复算法
    
    利用多个不同入射角度的衍射强度图进行相位恢复
    """
    
    def __init__(self, wavelength, pixel_size, distance,
                 angles=None,
                 method='angular_spectrum',
                 support=None,
                 positivity=True,
                 weights=None,
                 phase_corrector=None,
                 phase_reference_type=None):
        """
        初始化多角度GS算法
        
        参数:
            wavelength: 波长
            pixel_size: 像素大小
            distance: 传播距离
            angles: 入射角度列表（弧度） [(kx1, ky1), (kx2, ky2), ...]
                    如果为None，使用零角度
            method: 传播方法
            support: 支撑区域掩码
            positivity: 是否强制振幅为正
            weights: 各角度的权重
            phase_corrector: PhaseCorrector对象，用于相位校正
            phase_reference_type: 相位基准类型
        """
        self.wavelength = wavelength
        self.pixel_size = pixel_size
        self.distance = distance
        self.method = method
        self.support = support
        self.positivity = positivity
        
        if angles is None:
            self.angles = [(0, 0)]
        else:
            self.angles = angles
        
        if weights is None:
            self.weights = np.ones(len(self.angles)) / len(self.angles)
        else:
            self.weights = np.array(weights) / np.sum(weights)
            
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
        
    def apply_illumination(self, field, angle):
        """
        应用倾斜照明（添加线性相位）
        """
        kx, ky = angle
        ny, nx = field.shape
        
        x = np.linspace(-nx/2, nx/2, nx) * self.pixel_size
        y = np.linspace(-ny/2, ny/2, ny) * self.pixel_size
        X, Y = np.meshgrid(x, y)
        
        phase_factor = np.exp(1j * (kx * X + ky * Y))
        
        return field * phase_factor
    
    def apply_object_constraint(self, field):
        result = field.copy()
        
        if self.support is not None:
            result = result * self.support
            
        if self.positivity:
            amplitude = np.abs(result)
            phase = np.angle(result)
            amplitude = np.maximum(amplitude, 0)
            result = amplitude * np.exp(1j * phase)
            
        return result
    
    def apply_diffraction_constraint(self, field, measured_intensity, angle):
        """
        应用衍射平面约束（考虑倾斜照明）
        """
        illuminated = self.apply_illumination(field, angle)
        
        diffraction_field, _ = propagate(
            illuminated, self.wavelength, self.pixel_size,
            self.distance, self.method
        )
        
        phase = np.angle(diffraction_field)
        amplitude = np.sqrt(np.maximum(measured_intensity, 0))
        diffraction_constrained = amplitude * np.exp(1j * phase)
        
        back_prop, _ = back_propagate(
            diffraction_constrained, self.wavelength, self.pixel_size,
            self.distance, self.method
        )
        
        result = back_prop * np.conj(self.apply_illumination(np.ones_like(field), angle))
        
        return result
    
    def reconstruct(self, measured_intensities,
                    initial_guess=None,
                    num_iterations=100,
                    verbose=True):
        """
        执行多角度相位恢复
        """
        assert len(measured_intensities) == len(self.angles)
        
        ny, nx = measured_intensities[0].shape
        
        if initial_guess is None:
            random_phase = np.random.rand(ny, nx) * 2 * np.pi
            initial_amplitude = np.sqrt(np.mean(measured_intensities[0]))
            object_field = initial_amplitude * np.exp(1j * random_phase)
        else:
            object_field = initial_guess.copy()
        
        self.errors = []
        
        for iteration in range(num_iterations):
            updated_fields = []
            
            for idx, (intensity, angle) in enumerate(zip(measured_intensities, self.angles)):
                constrained = self.apply_diffraction_constraint(
                    object_field, intensity, angle
                )
                updated_fields.append(constrained)
            
            object_field = np.zeros_like(object_field, dtype=complex)
            for idx, field in enumerate(updated_fields):
                object_field += self.weights[idx] * field
            
            object_field = self.apply_object_constraint(object_field)
            
            object_field = self._apply_phase_correction(object_field)
            
            total_error = 0
            for idx, (intensity, angle) in enumerate(zip(measured_intensities, self.angles)):
                illuminated = self.apply_illumination(object_field, angle)
                diffraction_field, _ = propagate(
                    illuminated, self.wavelength, self.pixel_size,
                    self.distance, self.method
                )
                error = np.mean((np.abs(diffraction_field)**2 - intensity)**2)
                total_error += self.weights[idx] * error
            
            self.errors.append(total_error)
            
            if verbose and (iteration + 1) % 10 == 0:
                print(f"Iteration {iteration + 1}/{num_iterations}, Weighted Error: {total_error:.6e}")
        
        return object_field
    
    def get_errors(self):
        return np.array(self.errors)


class GeneralizedProjection:
    """
    广义投影算法 - 支持任意数量的约束
    
    可以组合多距离、多角度等多种约束
    """
    
    def __init__(self, wavelength, pixel_size,
                 method='angular_spectrum',
                 support=None,
                 positivity=True,
                 beta=0.9,
                 phase_corrector=None,
                 phase_reference_type=None):
        self.wavelength = wavelength
        self.pixel_size = pixel_size
        self.method = method
        self.support = support
        self.positivity = positivity
        self.beta = beta
        self.constraints = []
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
        
    def add_distance_constraint(self, distance, measured_intensity, weight=1.0):
        """
        添加距离约束
        """
        self.constraints.append({
            'type': 'distance',
            'distance': distance,
            'intensity': measured_intensity,
            'weight': weight
        })
        
    def add_angle_constraint(self, distance, angle, measured_intensity, weight=1.0):
        """
        添加角度约束
        """
        self.constraints.append({
            'type': 'angle',
            'distance': distance,
            'angle': angle,
            'intensity': measured_intensity,
            'weight': weight
        })
        
    def apply_object_constraint(self, field):
        result = field.copy()
        
        if self.support is not None:
            result = result * self.support
            
        if self.positivity:
            amplitude = np.abs(result)
            phase = np.angle(result)
            amplitude = np.maximum(amplitude, 0)
            result = amplitude * np.exp(1j * phase)
            
        return result
    
    def apply_constraint(self, field, constraint):
        """
        应用单个约束
        """
        if constraint['type'] == 'distance':
            return self._apply_distance_constraint(field, constraint)
        elif constraint['type'] == 'angle':
            return self._apply_angle_constraint(field, constraint)
        else:
            return field
    
    def _apply_distance_constraint(self, field, constraint):
        distance = constraint['distance']
        intensity = constraint['intensity']
        
        diffraction_field, _ = propagate(
            field, self.wavelength, self.pixel_size,
            distance, self.method
        )
        
        phase = np.angle(diffraction_field)
        amplitude = np.sqrt(np.maximum(intensity, 0))
        diffraction_constrained = amplitude * np.exp(1j * phase)
        
        result, _ = back_propagate(
            diffraction_constrained, self.wavelength, self.pixel_size,
            distance, self.method
        )
        
        return result
    
    def _apply_angle_constraint(self, field, constraint):
        distance = constraint['distance']
        angle = constraint['angle']
        intensity = constraint['intensity']
        kx, ky = angle
        
        ny, nx = field.shape
        x = np.linspace(-nx/2, nx/2, nx) * self.pixel_size
        y = np.linspace(-ny/2, ny/2, ny) * self.pixel_size
        X, Y = np.meshgrid(x, y)
        
        phase_factor = np.exp(1j * (kx * X + ky * Y))
        illuminated = field * phase_factor
        
        diffraction_field, _ = propagate(
            illuminated, self.wavelength, self.pixel_size,
            distance, self.method
        )
        
        phase = np.angle(diffraction_field)
        amplitude = np.sqrt(np.maximum(intensity, 0))
        diffraction_constrained = amplitude * np.exp(1j * phase)
        
        back_prop, _ = back_propagate(
            diffraction_constrained, self.wavelength, self.pixel_size,
            distance, self.method
        )
        
        result = back_prop * np.conj(phase_factor)
        
        return result
    
    def reconstruct(self, initial_guess=None, num_iterations=100, verbose=True, mode='HIO'):
        """
        执行相位恢复
        
        参数:
            mode: 'GS', 'HIO', 或 'RAAR'
        """
        if len(self.constraints) == 0:
            raise ValueError("没有添加任何约束")
        
        ny, nx = self.constraints[0]['intensity'].shape
        
        if initial_guess is None:
            random_phase = np.random.rand(ny, nx) * 2 * np.pi
            initial_amplitude = np.sqrt(np.mean(self.constraints[0]['intensity']))
            object_field = initial_amplitude * np.exp(1j * random_phase)
        else:
            object_field = initial_guess.copy()
        
        self.errors = []
        previous_object = object_field.copy()
        
        total_weight = sum(c['weight'] for c in self.constraints)
        
        for iteration in range(num_iterations):
            weighted_sum = np.zeros_like(object_field, dtype=complex)
            
            for constraint in self.constraints:
                constrained = self.apply_constraint(object_field, constraint)
                weighted_sum += constraint['weight'] * constrained
            
            avg_field = weighted_sum / total_weight
            
            if mode == 'GS':
                object_field = self.apply_object_constraint(avg_field)
            elif mode == 'HIO':
                if self.support is not None:
                    support_mask = self.support
                else:
                    support_mask = np.ones_like(np.abs(avg_field), dtype=bool)
                outside_support = ~support_mask
                
                object_field = avg_field.copy()
                object_field[outside_support] = previous_object[outside_support] - self.beta * avg_field[outside_support]
                
                if self.positivity:
                    amplitude = np.abs(object_field)
                    phase = np.angle(object_field)
                    amplitude[support_mask] = np.maximum(amplitude[support_mask], 0)
                    object_field = amplitude * np.exp(1j * phase)
                previous_object = avg_field.copy()
            elif mode == 'RAAR':
                beta = self.beta
                object_field = beta * self.apply_object_constraint(2 * avg_field - previous_object) + (1 - beta) * avg_field
                previous_object = avg_field.copy()
            
            object_field = self._apply_phase_correction(object_field)
            
            total_error = 0
            for constraint in self.constraints:
                if constraint['type'] == 'distance':
                    diffraction_field, _ = propagate(
                        object_field, self.wavelength, self.pixel_size,
                        constraint['distance'], self.method
                    )
                elif constraint['type'] == 'angle':
                    kx, ky = constraint['angle']
                    x = np.linspace(-nx/2, nx/2, nx) * self.pixel_size
                    y = np.linspace(-ny/2, ny/2, ny) * self.pixel_size
                    X, Y = np.meshgrid(x, y)
                    phase_factor = np.exp(1j * (kx * X + ky * Y))
                    illuminated = object_field * phase_factor
                    diffraction_field, _ = propagate(
                        illuminated, self.wavelength, self.pixel_size,
                        constraint['distance'], self.method
                    )
                
                error = np.mean((np.abs(diffraction_field)**2 - constraint['intensity'])**2)
                total_error += constraint['weight'] * error
            
            total_error /= total_weight
            self.errors.append(total_error)
            
            if verbose and (iteration + 1) % 10 == 0:
                print(f"Iteration {iteration + 1}/{num_iterations}, Error: {total_error:.6e}")
        
        return self.apply_object_constraint(object_field)
    
    def get_errors(self):
        return np.array(self.errors)
