import numpy as np
from numpy.fft import fft2, ifft2, fftshift, ifftshift


def angular_spectrum(field, wavelength, pixel_size, distance):
    """
    角谱传播方法 (Angular Spectrum Method)
    
    参数:
        field: 输入复振幅场
        wavelength: 波长
        pixel_size: 像素大小
        distance: 传播距离
    
    返回:
        传播后的复振幅场
    """
    ny, nx = field.shape
    k = 2 * np.pi / wavelength
    
    fx = np.fft.fftfreq(nx, pixel_size)
    fy = np.fft.fftfreq(ny, pixel_size)
    FX, FY = np.meshgrid(fx, fy)
    
    k_squared = k ** 2
    kx = 2 * np.pi * FX
    ky = 2 * np.pi * FY
    
    sqrt_term = np.sqrt(np.maximum(k_squared - kx**2 - ky**2, 0))
    H = np.exp(1j * distance * sqrt_term)
    
    field_ft = fft2(field)
    field_prop_ft = field_ft * H
    field_prop = ifft2(field_prop_ft)
    
    return field_prop


def fresnel_propagation(field, wavelength, pixel_size, distance):
    """
    菲涅尔衍射传播 (Fresnel Diffraction)
    
    参数:
        field: 输入复振幅场
        wavelength: 波长
        pixel_size: 像素大小
        distance: 传播距离
    
    返回:
        传播后的复振幅场
    """
    ny, nx = field.shape
    k = 2 * np.pi / wavelength
    
    x = np.linspace(-nx/2, nx/2, nx) * pixel_size
    y = np.linspace(-ny/2, ny/2, ny) * pixel_size
    X, Y = np.meshgrid(x, y)
    
    quad_phase = np.exp(1j * k * (X**2 + Y**2) / (2 * distance))
    
    field1 = field * quad_phase
    
    ft_size = nx * pixel_size**2 / (wavelength * distance)
    ft_pixel = wavelength * distance / (nx * pixel_size)
    
    field_ft = fftshift(fft2(ifftshift(field1))) * (pixel_size**2)
    
    x2 = np.linspace(-nx/2, nx/2, nx) * ft_pixel
    y2 = np.linspace(-ny/2, ny/2, ny) * ft_pixel
    X2, Y2 = np.meshgrid(x2, y2)
    
    quad_phase2 = np.exp(1j * k * (X2**2 + Y2**2) / (2 * distance))
    
    field_prop = quad_phase2 * field_ft * np.exp(1j * k * distance) / (1j * wavelength * distance)
    
    return field_prop, ft_pixel


def fraunhofer_propagation(field, wavelength, pixel_size, distance):
    """
    夫琅禾费衍射传播 (Fraunhofer Diffraction)
    
    参数:
        field: 输入复振幅场
        wavelength: 波长
        pixel_size: 像素大小
        distance: 传播距离
    
    返回:
        传播后的复振幅场, 输出像素大小
    """
    ny, nx = field.shape
    k = 2 * np.pi / wavelength
    
    x = np.linspace(-nx/2, nx/2, nx) * pixel_size
    y = np.linspace(-ny/2, ny/2, ny) * pixel_size
    X, Y = np.meshgrid(x, y)
    
    lens_phase = np.exp(-1j * k * (X**2 + Y**2) / (2 * distance))
    
    field_lens = field * lens_phase
    
    ft_pixel = wavelength * distance / (nx * pixel_size)
    
    field_ft = fftshift(fft2(ifftshift(field_lens))) * (pixel_size**2)
    
    field_prop = np.exp(1j * k * distance) / (1j * wavelength * distance) * field_ft
    
    return field_prop, ft_pixel


def propagate(field, wavelength, pixel_size, distance, method='angular_spectrum'):
    """
    统一的传播函数接口
    
    参数:
        field: 输入复振幅场
        wavelength: 波长
        pixel_size: 像素大小
        distance: 传播距离
        method: 传播方法 ('angular_spectrum', 'fresnel', 'fraunhofer')
    
    返回:
        传播后的复振幅场, 输出像素大小
    """
    if method == 'angular_spectrum':
        return angular_spectrum(field, wavelength, pixel_size, distance), pixel_size
    elif method == 'fresnel':
        return fresnel_propagation(field, wavelength, pixel_size, distance)
    elif method == 'fraunhofer':
        return fraunhofer_propagation(field, wavelength, pixel_size, distance)
    else:
        raise ValueError(f"未知的传播方法: {method}")


def back_propagate(field, wavelength, pixel_size, distance, method='angular_spectrum'):
    """
    反向传播（距离取负值）
    
    参数:
        field: 输入复振幅场
        wavelength: 波长
        pixel_size: 像素大小
        distance: 传播距离（将取负值反向传播）
        method: 传播方法
    
    返回:
        反向传播后的复振幅场, 输出像素大小
    """
    return propagate(field, wavelength, pixel_size, -distance, method)
