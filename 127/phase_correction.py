import numpy as np


class PhaseReference:
    """
    相位基准类，用于定义相位校正的参考点/区域
    """
    
    def __init__(self, reference_type='point', **kwargs):
        """
        初始化相位基准
        
        参数:
            reference_type: 基准类型
                'point' - 单点相位基准
                'region' - 区域相位基准
                'support' - 支撑域相位基准
                'amplitude_weighted' - 振幅加权相位基准
                'zero_phase' - 零相位约束基准
        """
        self.reference_type = reference_type
        self.params = kwargs
        
    def apply(self, field):
        """
        应用相位校正
        
        参数:
            field: 复振幅场
            
        返回:
            校正后的复振幅场
        """
        if self.reference_type == 'point':
            return self._point_reference(field)
        elif self.reference_type == 'region':
            return self._region_reference(field)
        elif self.reference_type == 'support':
            return self._support_reference(field)
        elif self.reference_type == 'amplitude_weighted':
            return self._amplitude_weighted_reference(field)
        elif self.reference_type == 'zero_phase':
            return self._zero_phase_reference(field)
        else:
            return field
    
    def _point_reference(self, field):
        """
        单点相位基准：固定某个像素的相位
        """
        x = self.params.get('x', field.shape[1] // 2)
        y = self.params.get('y', field.shape[0] // 2)
        target_phase = self.params.get('phase', 0.0)
        
        current_phase = np.angle(field[y, x])
        phase_offset = target_phase - current_phase
        
        return field * np.exp(1j * phase_offset)
    
    def _region_reference(self, field):
        """
        区域相位基准：固定某个区域的平均相位
        """
        region_mask = self.params.get('mask', None)
        target_phase = self.params.get('phase', 0.0)
        
        if region_mask is None:
            ny, nx = field.shape
            cy, cx = ny // 2, nx // 2
            region_size = self.params.get('size', 5)
            region_mask = np.zeros((ny, nx), dtype=bool)
            region_mask[cy-region_size:cy+region_size, cx-region_size:cx+region_size] = True
        
        region_phases = np.angle(field[region_mask])
        region_amplitude = np.abs(field[region_mask])
        
        if np.sum(region_amplitude) > 0:
            avg_phase = np.sum(region_phases * region_amplitude) / np.sum(region_amplitude)
        else:
            avg_phase = np.mean(region_phases)
        
        phase_offset = target_phase - avg_phase
        
        return field * np.exp(1j * phase_offset)
    
    def _support_reference(self, field):
        """
        支撑域相位基准：固定支撑域内的平均相位
        """
        support = self.params.get('support', None)
        
        if support is None:
            return field
        
        target_phase = self.params.get('phase', 0.0)
        
        support_amplitude = np.abs(field[support])
        support_phases = np.angle(field[support])
        
        if np.sum(support_amplitude) > 0:
            avg_phase = np.sum(support_phases * support_amplitude) / np.sum(support_amplitude)
        else:
            avg_phase = np.mean(support_phases)
        
        phase_offset = target_phase - avg_phase
        
        return field * np.exp(1j * phase_offset)
    
    def _amplitude_weighted_reference(self, field):
        """
        振幅加权相位基准：以强振幅区域的相位为参考
        """
        amplitude = np.abs(field)
        threshold = self.params.get('threshold', 0.5)
        
        high_amplitude_mask = amplitude > threshold * np.max(amplitude)
        
        if np.any(high_amplitude_mask):
            phases = np.angle(field[high_amplitude_mask])
            weights = amplitude[high_amplitude_mask]
            avg_phase = np.sum(phases * weights) / np.sum(weights)
            
            phase_offset = -avg_phase
            return field * np.exp(1j * phase_offset)
        
        return field
    
    def _zero_phase_reference(self, field):
        """
        零相位约束：在指定区域强制相位为零
        """
        zero_mask = self.params.get('mask', None)
        
        if zero_mask is None:
            return field
        
        result = field.copy()
        amplitude = np.abs(result[zero_mask])
        result[zero_mask] = amplitude
        
        return result


def correct_global_phase(field, reference_field=None, method='amplitude_correlation'):
    """
    全局相位校正（相对于参考场）
    
    参数:
        field: 待校正的复振幅场
        reference_field: 参考复振幅场（可选）
        method: 校正方法
            'amplitude_correlation' - 振幅加权相位相关
            'max_amplitude' - 最大振幅点相位对齐
            'center_point' - 中心点相位对齐
            'mean_phase' - 平均相位对齐
    
    返回:
        校正后的复振幅场
    """
    if reference_field is None:
        return field
    
    if method == 'amplitude_correlation':
        phase_diff = np.angle(field * np.conj(reference_field))
        amplitude = np.abs(field) * np.abs(reference_field)
        
        if np.sum(amplitude) > 0:
            avg_phase_diff = np.sum(phase_diff * amplitude) / np.sum(amplitude)
        else:
            avg_phase_diff = np.mean(phase_diff)
        
        return field * np.exp(-1j * avg_phase_diff)
    
    elif method == 'max_amplitude':
        idx = np.unravel_index(np.argmax(np.abs(reference_field)), reference_field.shape)
        phase_diff = np.angle(field[idx]) - np.angle(reference_field[idx])
        return field * np.exp(-1j * phase_diff)
    
    elif method == 'center_point':
        ny, nx = field.shape
        cy, cx = ny // 2, nx // 2
        phase_diff = np.angle(field[cy, cx]) - np.angle(reference_field[cy, cx])
        return field * np.exp(-1j * phase_diff)
    
    elif method == 'mean_phase':
        phase_diff = np.mean(np.angle(field * np.conj(reference_field)))
        return field * np.exp(-1j * phase_diff)
    
    else:
        return field


def remove_phase_tilt(field):
    """
    移除线性相位倾斜（整体倾斜）
    
    参数:
        field: 复振幅场
    
    返回:
        移除倾斜后的复振幅场
    """
    phase = np.angle(field)
    amplitude = np.abs(field)
    
    ny, nx = phase.shape
    x = np.arange(nx) - nx // 2
    y = np.arange(ny) - ny // 2
    X, Y = np.meshgrid(x, y)
    
    weights = amplitude ** 2
    weights_sum = np.sum(weights)
    
    if weights_sum > 0:
        a = np.sum(phase * weights * X) / np.sum(weights * X ** 2) if np.sum(weights * X ** 2) > 0 else 0
        b = np.sum(phase * weights * Y) / np.sum(weights * Y ** 2) if np.sum(weights * Y ** 2) > 0 else 0
        
        phase_tilt = a * X + b * Y
        phase_corrected = phase - phase_tilt
        
        return amplitude * np.exp(1j * phase_corrected)
    
    return field


class PhaseCorrector:
    """
    相位校正器，组合多种相位校正方法
    """
    
    def __init__(self, 
                 global_phase=True,
                 phase_tilt=False,
                 reference=None,
                 global_phase_method='amplitude_weighted'):
        """
        初始化相位校正器
        
        参数:
            global_phase: 是否进行全局相位校正
            phase_tilt: 是否移除相位倾斜
            reference: PhaseReference对象或参考场
            global_phase_method: 全局相位校正方法
        """
        self.global_phase = global_phase
        self.phase_tilt = phase_tilt
        self.reference = reference
        self.global_phase_method = global_phase_method
        
    def apply(self, field, reference_field=None):
        """
        应用相位校正
        
        参数:
            field: 待校正的复振幅场
            reference_field: 参考场（用于相对校正）
            
        返回:
            校正后的复振幅场
        """
        result = field.copy()
        
        if self.phase_tilt:
            result = remove_phase_tilt(result)
        
        if self.global_phase:
            if self.reference is not None and isinstance(self.reference, PhaseReference):
                result = self.reference.apply(result)
            elif reference_field is not None:
                result = correct_global_phase(result, reference_field, self.global_phase_method)
        
        return result
