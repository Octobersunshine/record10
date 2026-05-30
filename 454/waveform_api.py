import wave
import struct
import os
import tempfile
import numpy as np
import base64
import io
from typing import Tuple, List, Optional, Dict, Union
from dataclasses import dataclass, field


SUPPORTED_FORMATS = {'.wav', '.mp3', '.m4a', '.aac', '.flac', '.ogg', '.wma', '.opus'}
FFMPEG_AVAILABLE = None


def check_ffmpeg() -> bool:
    global FFMPEG_AVAILABLE
    if FFMPEG_AVAILABLE is not None:
        return FFMPEG_AVAILABLE
    
    try:
        from pydub.utils import which
        FFMPEG_AVAILABLE = which("ffmpeg") is not None
    except Exception:
        FFMPEG_AVAILABLE = False
    
    return FFMPEG_AVAILABLE


@dataclass
class WaveformResult:
    points: List[float]
    svg_path: str
    base64_image: str
    sample_rate: int
    duration: float
    num_samples: int
    num_channels: int
    width: int
    height: int
    downsample_factor: int
    channel_strategy: str
    original_format: str = '.wav'
    spectogram_base64: str = ''
    spectogram_svg: str = ''
    frequency_bins: List[float] = field(default_factory=list)
    time_bins: List[float] = field(default_factory=list)


@dataclass
class AudioSegmentResult:
    audio_data: bytes
    sample_rate: int
    num_channels: int
    sample_width: int
    start_time: float
    end_time: float
    duration: float
    format: str = 'wav'
    base64_data: str = ''


def convert_to_wav(file_path: str) -> Tuple[str, str]:
    ext = os.path.splitext(file_path)[1].lower()
    
    if ext == '.wav':
        return file_path, ext
    
    if not check_ffmpeg():
        raise RuntimeError(
            "需要安装ffmpeg才能处理非WAV格式音频。\n"
            "Windows: 从 https://ffmpeg.org/download.html 下载并添加到PATH\n"
            "Mac: brew install ffmpeg\n"
            "Linux: sudo apt-get install ffmpeg"
        )
    
    try:
        from pydub import AudioSegment
    except ImportError:
        raise ImportError("需要安装pydub库: pip install pydub")
    
    audio = AudioSegment.from_file(file_path, format=ext.lstrip('.'))
    
    temp_fd, temp_path = tempfile.mkstemp(suffix='.wav')
    os.close(temp_fd)
    
    audio.export(temp_path, format='wav')
    
    return temp_path, ext


def read_wav(file_path: str, channel_strategy: str = 'average') -> Tuple[np.ndarray, int, float, int]:
    ext = os.path.splitext(file_path)[1].lower()
    
    if ext != '.wav':
        file_path, _ = convert_to_wav(file_path)
        temp_file = file_path
    else:
        temp_file = None
    
    try:
        with wave.open(file_path, 'rb') as wav_file:
            num_channels = wav_file.getnchannels()
            sample_width = wav_file.getsampwidth()
            sample_rate = wav_file.getframerate()
            num_frames = wav_file.getnframes()
            duration = num_frames / sample_rate
            raw_data = wav_file.readframes(num_frames)
        
        fmt = f"<{num_frames * num_channels}"
        if sample_width == 1:
            fmt += "b"
            dtype = np.int8
        elif sample_width == 2:
            fmt += "h"
            dtype = np.int16
        elif sample_width == 4:
            fmt += "i"
            dtype = np.int32
        else:
            raise ValueError(f"不支持的采样位宽: {sample_width}")
        
        samples = struct.unpack(fmt, raw_data)
        samples = np.array(samples, dtype=dtype)
        
        if num_channels > 1:
            samples = samples.reshape(-1, num_channels)
            if channel_strategy == 'average':
                samples = np.mean(samples, axis=1)
            elif channel_strategy == 'left':
                samples = samples[:, 0]
            elif channel_strategy == 'right':
                samples = samples[:, 1] if num_channels >= 2 else samples[:, 0]
            elif channel_strategy == 'mix':
                if num_channels >= 2:
                    samples = 0.5 * samples[:, 0] + 0.5 * samples[:, 1]
                else:
                    samples = samples[:, 0]
            else:
                raise ValueError(f"不支持的通道策略: {channel_strategy}，支持的策略: average, left, right, mix")
        
        samples = samples.astype(np.float64)
        max_val = float(np.iinfo(dtype).max)
        samples = samples / max_val
        
        return samples, sample_rate, duration, num_channels
    finally:
        if temp_file and os.path.exists(temp_file):
            try:
                os.unlink(temp_file)
            except:
                pass


def downsample(samples: np.ndarray, target_points: int = 1000, method: str = 'peak') -> np.ndarray:
    if len(samples) <= target_points:
        return samples
    
    if method == 'peak':
        num_samples = len(samples)
        block_size = num_samples // target_points
        reshaped = samples[:block_size * target_points].reshape(-1, block_size)
        max_vals = np.max(reshaped, axis=1)
        min_vals = np.min(reshaped, axis=1)
        result = np.zeros(target_points * 2)
        result[0::2] = max_vals
        result[1::2] = min_vals
        return result
    elif method == 'average':
        num_samples = len(samples)
        block_size = num_samples // target_points
        reshaped = samples[:block_size * target_points].reshape(-1, block_size)
        return np.mean(reshaped, axis=1)
    elif method == 'rms':
        num_samples = len(samples)
        block_size = num_samples // target_points
        reshaped = samples[:block_size * target_points].reshape(-1, block_size)
        return np.sqrt(np.mean(reshaped ** 2, axis=1))
    else:
        raise ValueError(f"不支持的降采样方法: {method}")


def adaptive_downsample(samples: np.ndarray, duration: float, width: int = 800, 
                        max_points_per_pixel: float = 2.0, min_points: int = 100, 
                        max_points: int = 2000, method: str = 'peak') -> Tuple[np.ndarray, int]:
    num_samples = len(samples)
    
    optimal_points = int(width * max_points_per_pixel)
    optimal_points = max(min_points, min(optimal_points, max_points))
    
    if method == 'peak':
        optimal_points = optimal_points // 2
    
    if num_samples <= optimal_points:
        downsample_factor = 1
        return samples, downsample_factor
    
    downsample_factor = num_samples // optimal_points
    downsample_factor = max(1, downsample_factor)
    
    actual_target_points = num_samples // downsample_factor
    
    if method == 'peak':
        actual_target_points = actual_target_points // 2
    
    downsampled = downsample(samples, actual_target_points, method)
    
    return downsampled, downsample_factor


def normalize_points(points: np.ndarray, height: int = 200) -> np.ndarray:
    max_abs = np.max(np.abs(points))
    if max_abs == 0:
        return points
    normalized = points / max_abs
    return normalized * (height / 2)


def generate_svg_path(points: np.ndarray, width: int = 800, height: int = 200, 
                      stroke_color: str = '#4a90d9', stroke_width: int = 1,
                      fill_color: Optional[str] = 'rgba(74, 144, 217, 0.3)') -> str:
    normalized = normalize_points(points, height)
    center_y = height / 2
    num_points = len(normalized)
    step_x = width / (num_points - 1) if num_points > 1 else width
    
    path_parts = []
    for i, y in enumerate(normalized):
        x = i * step_x
        y_pos = center_y - y
        if i == 0:
            path_parts.append(f"M {x:.2f} {y_pos:.2f}")
        else:
            path_parts.append(f"L {x:.2f} {y_pos:.2f}")
    
    if fill_color:
        for i in range(num_points - 1, -1, -1):
            x = i * step_x
            y = normalized[i]
            y_pos = center_y + y
            path_parts.append(f"L {x:.2f} {y_pos:.2f}")
        path_parts.append("Z")
    
    path_str = ' '.join(path_parts)
    
    svg = f'''<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">
        <rect width="100%" height="100%" fill="transparent"/>
        <path d="{path_str}" fill="{fill_color or 'none'}" stroke="{stroke_color}" stroke-width="{stroke_width}" stroke-linejoin="round"/>
    </svg>'''
    
    return svg


def generate_base64_image(points: np.ndarray, width: int = 800, height: int = 200,
                          bg_color: Tuple[int, int, int] = (255, 255, 255),
                          line_color: Tuple[int, int, int] = (74, 144, 217),
                          fill_color: Optional[Tuple[int, int, int, int]] = (74, 144, 217, 77)) -> str:
    try:
        from PIL import Image, ImageDraw
    except ImportError:
        raise ImportError("需要安装Pillow库才能生成Base64图片: pip install pillow")
    
    normalized = normalize_points(points, height)
    center_y = height / 2
    num_points = len(normalized)
    step_x = width / (num_points - 1) if num_points > 1 else width
    
    img = Image.new('RGB', (width, height), bg_color)
    draw = ImageDraw.Draw(img, 'RGBA')
    
    upper_points = []
    lower_points = []
    for i, y in enumerate(normalized):
        x = i * step_x
        upper_points.append((x, center_y - y))
        lower_points.append((x, center_y + y))
    
    if fill_color and len(upper_points) > 1:
        fill_points = upper_points + list(reversed(lower_points))
        draw.polygon(fill_points, fill=fill_color)
    
    if len(upper_points) > 1:
        draw.line(upper_points, fill=line_color, width=1)
    
    buffered = io.BytesIO()
    img.save(buffered, format='PNG')
    img_str = base64.b64encode(buffered.getvalue()).decode('utf-8')
    
    return f"data:image/png;base64,{img_str}"


def compute_spectrogram(samples: np.ndarray, sample_rate: int, 
                        n_fft: int = 2048, hop_length: int = 512,
                        window: str = 'hann') -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    window_func = {
        'hann': np.hanning,
        'hamming': np.hamming,
        'blackman': np.blackman,
        'rect': np.ones
    }.get(window, np.hanning)(n_fft)
    
    num_samples = len(samples)
    num_frames = 1 + (num_samples - n_fft) // hop_length
    
    if num_frames <= 0:
        num_frames = 1
        hop_length = num_samples
    
    spec = np.zeros((n_fft // 2 + 1, num_frames), dtype=np.float64)
    
    for i in range(num_frames):
        start = i * hop_length
        end = start + n_fft
        if end > num_samples:
            frame = np.zeros(n_fft)
            frame[:num_samples - start] = samples[start:]
        else:
            frame = samples[start:end]
        
        frame = frame * window_func
        fft_result = np.fft.rfft(frame)
        spec[:, i] = np.abs(fft_result) ** 2
    
    spec = 10 * np.log10(spec + 1e-10)
    
    freqs = np.fft.rfftfreq(n_fft, 1.0 / sample_rate)
    times = np.arange(num_frames) * hop_length / sample_rate
    
    return spec, freqs, times


def generate_spectrogram_image(spec: np.ndarray, freqs: np.ndarray, times: np.ndarray,
                               width: int = 800, height: int = 300,
                               cmap: str = 'viridis') -> Tuple[str, str]:
    try:
        from PIL import Image
    except ImportError:
        raise ImportError("需要安装Pillow库才能生成频谱图: pip install pillow")
    
    spec_norm = (spec - np.min(spec)) / (np.max(spec) - np.min(spec) + 1e-10)
    
    colormaps = {
        'viridis': [(68, 1, 84), (59, 82, 139), (33, 145, 140), (94, 201, 98), (253, 231, 37)],
        'plasma': [(13, 8, 135), (75, 3, 161), (127, 3, 168), (175, 33, 139), (217, 73, 98), (247, 122, 56), (251, 182, 31), (240, 249, 33)],
        'inferno': [(0, 0, 4), (25, 12, 67), (67, 16, 113), (109, 25, 137), (152, 38, 143), (194, 53, 133), (231, 76, 108), (251, 110, 77), (255, 152, 46), (246, 200, 30), (229, 250, 49)],
        'magma': [(0, 0, 4), (24, 11, 77), (62, 13, 133), (102, 23, 149), (140, 41, 145), (177, 61, 128), (210, 85, 105), (236, 117, 79), (252, 159, 52), (251, 208, 33), (226, 253, 57)],
        'hot': [(0, 0, 0), (127, 0, 0), (255, 0, 0), (255, 127, 0), (255, 255, 0), (255, 255, 255)],
        'coolwarm': [(59, 76, 192), (100, 135, 222), (144, 185, 240), (190, 220, 247), (232, 232, 232), (247, 202, 190), (239, 158, 133), (222, 113, 88), (180, 4, 38)]
    }
    
    colors = colormaps.get(cmap, colormaps['viridis'])
    
    def interpolate_color(value):
        if value <= 0:
            return colors[0]
        if value >= 1:
            return colors[-1]
        
        pos = value * (len(colors) - 1)
        idx = int(pos)
        frac = pos - idx
        
        c1 = colors[idx]
        c2 = colors[min(idx + 1, len(colors) - 1)]
        
        return (
            int(c1[0] + (c2[0] - c1[0]) * frac),
            int(c1[1] + (c2[1] - c1[1]) * frac),
            int(c1[2] + (c2[2] - c1[2]) * frac)
        )
    
    spec_resized = np.zeros((height, width, 3), dtype=np.uint8)
    
    for y in range(height):
        spec_y = int((1 - y / height) * spec_norm.shape[0])
        spec_y = max(0, min(spec_y, spec_norm.shape[0] - 1))
        for x in range(width):
            spec_x = int(x / width * spec_norm.shape[1])
            spec_x = max(0, min(spec_x, spec_norm.shape[1] - 1))
            val = spec_norm[spec_y, spec_x]
            spec_resized[y, x] = interpolate_color(val)
    
    img = Image.fromarray(spec_resized)
    buffered = io.BytesIO()
    img.save(buffered, format='PNG')
    img_str = base64.b64encode(buffered.getvalue()).decode('utf-8')
    
    svg_parts = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">']
    svg_parts.append(f'<rect width="100%" height="100%" fill="rgb{colors[0]}"/>')
    
    for y in range(0, height, 2):
        for x in range(0, width, 2):
            spec_y = int((1 - y / height) * spec_norm.shape[0])
            spec_y = max(0, min(spec_y, spec_norm.shape[0] - 1))
            spec_x = int(x / width * spec_norm.shape[1])
            spec_x = max(0, min(spec_x, spec_norm.shape[1] - 1))
            val = spec_norm[spec_y, spec_x]
            r, g, b = interpolate_color(val)
            svg_parts.append(f'<rect x="{x}" y="{y}" width="2" height="2" fill="rgb({r},{g},{b})"/>')
    
    svg_parts.append('</svg>')
    svg_str = ''.join(svg_parts)
    
    return f"data:image/png;base64,{img_str}", svg_str


def generate_spectrogram(file_path: str, width: int = 800, height: int = 300,
                         n_fft: int = 2048, hop_length: int = 512,
                         window: str = 'hann', cmap: str = 'viridis',
                         channel_strategy: str = 'average',
                         include_svg: bool = True) -> Dict[str, Union[str, List[float]]]:
    samples, sample_rate, duration, num_channels = read_wav(file_path, channel_strategy)
    
    spec, freqs, times = compute_spectrogram(
        samples, sample_rate, n_fft, hop_length, window
    )
    
    base64_img, svg_str = generate_spectrogram_image(
        spec, freqs, times, width, height, cmap
    )
    
    result = {
        'base64_image': base64_img,
        'frequency_bins': freqs.tolist(),
        'time_bins': times.tolist(),
        'sample_rate': sample_rate,
        'duration': duration,
        'num_channels': num_channels,
        'width': width,
        'height': height,
        'n_fft': n_fft,
        'hop_length': hop_length
    }
    
    if include_svg:
        result['svg'] = svg_str
    
    return result


def get_time_from_position(x_pos: float, width: int, duration: float) -> float:
    return max(0.0, min(duration, (x_pos / width) * duration))


def get_position_from_time(time: float, width: int, duration: float) -> float:
    return max(0.0, min(width, (time / duration) * width))


def segment_audio(file_path: str, start_time: float, end_time: float,
                  output_format: str = 'wav',
                  channel_strategy: str = 'average') -> AudioSegmentResult:
    if start_time < 0:
        start_time = 0
    if end_time <= start_time:
        raise ValueError(f"结束时间必须大于开始时间: {end_time} <= {start_time}")
    
    ext = os.path.splitext(file_path)[1].lower()
    
    if ext != '.wav' and not check_ffmpeg():
        raise RuntimeError("需要安装ffmpeg才能进行音频分段截取")
    
    try:
        from pydub import AudioSegment
    except ImportError:
        if ext != '.wav':
            raise ImportError("需要安装pydub库才能处理音频分段: pip install pydub")
    
    if ext == '.wav':
        samples, sample_rate, duration, num_channels = read_wav(file_path, channel_strategy)
        
        start_sample = int(start_time * sample_rate)
        end_sample = int(end_time * sample_rate)
        start_sample = max(0, min(start_sample, len(samples)))
        end_sample = max(start_sample, min(end_sample, len(samples)))
        
        segment_samples = samples[start_sample:end_sample]
        
        if len(segment_samples) == 0:
            raise ValueError("截取的音频段为空，请检查时间范围")
        
        max_val = 32767
        int_samples = (segment_samples * max_val).astype(np.int16)
        
        buffer = io.BytesIO()
        with wave.open(buffer, 'wb') as wav_file:
            wav_file.setnchannels(1)
            wav_file.setsampwidth(2)
            wav_file.setframerate(sample_rate)
            wav_file.writeframes(int_samples.tobytes())
        
        audio_data = buffer.getvalue()
        actual_duration = len(segment_samples) / sample_rate
        base64_data = base64.b64encode(audio_data).decode('utf-8')
        
        return AudioSegmentResult(
            audio_data=audio_data,
            sample_rate=sample_rate,
            num_channels=1,
            sample_width=2,
            start_time=start_time,
            end_time=end_time,
            duration=actual_duration,
            format='wav',
            base64_data=f"data:audio/wav;base64,{base64_data}"
        )
    else:
        audio = AudioSegment.from_file(file_path, format=ext.lstrip('.'))
        
        if end_time * 1000 > len(audio):
            end_time = len(audio) / 1000.0
        
        if start_time * 1000 >= len(audio):
            raise ValueError("开始时间超出音频长度")
        
        segment = audio[start_time * 1000:end_time * 1000]
        
        if output_format == 'mp3':
            buffer = io.BytesIO()
            segment.export(buffer, format='mp3')
            audio_data = buffer.getvalue()
            fmt = 'mp3'
            mime = 'audio/mpeg'
        elif output_format == 'wav':
            buffer = io.BytesIO()
            segment.export(buffer, format='wav')
            audio_data = buffer.getvalue()
            fmt = 'wav'
            mime = 'audio/wav'
        else:
            raise ValueError(f"不支持的输出格式: {output_format}，支持: wav, mp3")
        
        base64_data = base64.b64encode(audio_data).decode('utf-8')
        
        return AudioSegmentResult(
            audio_data=audio_data,
            sample_rate=segment.frame_rate,
            num_channels=segment.channels,
            sample_width=segment.sample_width,
            start_time=start_time,
            end_time=start_time + len(segment) / 1000.0,
            duration=len(segment) / 1000.0,
            format=fmt,
            base64_data=f"data:{mime};base64,{base64_data}"
        )


def generate_waveform(file_path: str, num_points: Optional[int] = None, 
                      width: int = 800, height: int = 200,
                      downsample_method: str = 'peak',
                      channel_strategy: str = 'average',
                      use_adaptive_downsample: bool = True,
                      max_points_per_pixel: float = 2.0,
                      min_points: int = 100,
                      max_points: int = 2000,
                      svg_stroke_color: str = '#4a90d9',
                      svg_stroke_width: int = 1,
                      svg_fill_color: Optional[str] = 'rgba(74, 144, 217, 0.3)',
                      image_bg_color: Tuple[int, int, int] = (255, 255, 255),
                      image_line_color: Tuple[int, int, int] = (74, 144, 217),
                      image_fill_color: Optional[Tuple[int, int, int, int]] = (74, 144, 217, 77),
                      include_image: bool = True,
                      include_spectrogram: bool = False,
                      spectrogram_width: int = 800,
                      spectrogram_height: int = 300,
                      n_fft: int = 2048,
                      hop_length: int = 512,
                      spectrogram_cmap: str = 'viridis') -> WaveformResult:
    original_ext = os.path.splitext(file_path)[1].lower()
    
    samples, sample_rate, duration, num_channels = read_wav(file_path, channel_strategy)
    
    if use_adaptive_downsample and num_points is None:
        downsampled, downsample_factor = adaptive_downsample(
            samples, duration, width,
            max_points_per_pixel, min_points, max_points,
            downsample_method
        )
    else:
        target_points = num_points if num_points is not None else 1000
        downsampled = downsample(samples, target_points, downsample_method)
        downsample_factor = len(samples) // len(downsampled)
        if downsample_method == 'peak':
            downsample_factor = downsample_factor * 2
    
    points_list = downsampled.tolist()
    
    svg_path = generate_svg_path(
        downsampled, width, height,
        svg_stroke_color, svg_stroke_width, svg_fill_color
    )
    
    base64_image = ""
    if include_image:
        base64_image = generate_base64_image(
            downsampled, width, height,
            image_bg_color, image_line_color, image_fill_color
        )
    
    spectogram_base64 = ""
    spectogram_svg = ""
    frequency_bins = []
    time_bins = []
    
    if include_spectrogram:
        try:
            spec_result = generate_spectrogram(
                file_path,
                width=spectrogram_width,
                height=spectrogram_height,
                n_fft=n_fft,
                hop_length=hop_length,
                cmap=spectrogram_cmap,
                channel_strategy=channel_strategy,
                include_svg=True
            )
            spectogram_base64 = spec_result['base64_image']
            spectogram_svg = spec_result.get('svg', '')
            frequency_bins = spec_result['frequency_bins']
            time_bins = spec_result['time_bins']
        except Exception as e:
            print(f"警告: 频谱图生成失败: {e}")
    
    return WaveformResult(
        points=points_list,
        svg_path=svg_path,
        base64_image=base64_image,
        sample_rate=sample_rate,
        duration=duration,
        num_samples=len(samples),
        num_channels=num_channels,
        width=width,
        height=height,
        downsample_factor=downsample_factor,
        channel_strategy=channel_strategy,
        original_format=original_ext,
        spectogram_base64=spectogram_base64,
        spectogram_svg=spectogram_svg,
        frequency_bins=frequency_bins,
        time_bins=time_bins
    )


def waveform_to_dict(result: WaveformResult) -> Dict[str, Union[List[float], int, float, str]]:
    data = {
        'points': result.points,
        'svg_path': result.svg_path,
        'base64_image': result.base64_image,
        'sample_rate': result.sample_rate,
        'duration': result.duration,
        'num_samples': result.num_samples,
        'num_channels': result.num_channels,
        'width': result.width,
        'height': result.height,
        'downsample_factor': result.downsample_factor,
        'channel_strategy': result.channel_strategy,
        'original_format': result.original_format,
        'num_output_points': len(result.points)
    }
    
    if result.spectogram_base64:
        data['spectogram_base64'] = result.spectogram_base64
    if result.spectogram_svg:
        data['spectogram_svg'] = result.spectogram_svg
    if result.frequency_bins:
        data['frequency_bins'] = result.frequency_bins
    if result.time_bins:
        data['time_bins'] = result.time_bins
    
    return data


def audio_segment_to_dict(seg: AudioSegmentResult) -> Dict[str, Union[bytes, int, float, str]]:
    return {
        'sample_rate': seg.sample_rate,
        'num_channels': seg.num_channels,
        'sample_width': seg.sample_width,
        'start_time': seg.start_time,
        'end_time': seg.end_time,
        'duration': seg.duration,
        'format': seg.format,
        'base64_data': seg.base64_data
    }


if __name__ == '__main__':
    import sys
    import json
    import argparse
    
    parser = argparse.ArgumentParser(description='音频波形图生成工具')
    parser.add_argument('file_path', help='音频文件路径 (支持WAV, MP3, M4A, FLAC等格式)')
    parser.add_argument('--num_points', type=int, default=None, help='输出点数（默认使用自适应降采样）')
    parser.add_argument('--width', type=int, default=800, help='波形图宽度，默认800')
    parser.add_argument('--height', type=int, default=200, help='波形图高度，默认200')
    parser.add_argument('--method', type=str, default='peak', choices=['peak', 'average', 'rms'],
                        help='降采样方法: peak(默认), average, rms')
    parser.add_argument('--channel', type=str, default='average', choices=['average', 'left', 'right', 'mix'],
                        help='通道合并策略: average(默认), left, right, mix')
    parser.add_argument('--no-adaptive', action='store_true', help='禁用自适应降采样')
    parser.add_argument('--max-points-per-pixel', type=float, default=2.0,
                        help='每像素最大点数，默认2.0')
    parser.add_argument('--min-points', type=int, default=100, help='最小输出点数，默认100')
    parser.add_argument('--max-points', type=int, default=2000, help='最大输出点数，默认2000')
    parser.add_argument('--no-image', action='store_true', help='不生成Base64图片')
    parser.add_argument('--svg-stroke-color', type=str, default='#4a90d9', help='SVG描边颜色')
    parser.add_argument('--svg-stroke-width', type=int, default=1, help='SVG描边宽度')
    parser.add_argument('--svg-fill-color', type=str, default='rgba(74, 144, 217, 0.3)', help='SVG填充颜色')
    
    parser.add_argument('--spectrogram', action='store_true', help='生成频谱图')
    parser.add_argument('--spec-width', type=int, default=800, help='频谱图宽度，默认800')
    parser.add_argument('--spec-height', type=int, default=300, help='频谱图高度，默认300')
    parser.add_argument('--n-fft', type=int, default=2048, help='FFT窗口大小，默认2048')
    parser.add_argument('--hop-length', type=int, default=512, help='帧移大小，默认512')
    parser.add_argument('--cmap', type=str, default='viridis', 
                        choices=['viridis', 'plasma', 'inferno', 'magma', 'hot', 'coolwarm'],
                        help='频谱图颜色映射，默认viridis')
    
    parser.add_argument('--segment', nargs=2, type=float, metavar=('START', 'END'),
                        help='截取指定时间段的音频 (秒)，例如: --segment 1.5 3.0')
    parser.add_argument('--output-format', type=str, default='wav', choices=['wav', 'mp3'],
                        help='音频截取输出格式，默认wav')
    parser.add_argument('--output', type=str, help='音频截取输出文件路径')
    
    args = parser.parse_args()
    
    try:
        if args.segment:
            start, end = args.segment
            seg_result = segment_audio(
                args.file_path, start, end,
                output_format=args.output_format,
                channel_strategy=args.channel
            )
            
            if args.output:
                with open(args.output, 'wb') as f:
                    f.write(seg_result.audio_data)
                print(f"音频已保存到: {args.output}")
            
            print(json.dumps(audio_segment_to_dict(seg_result), indent=2, ensure_ascii=False))
        else:
            result = generate_waveform(
                args.file_path,
                num_points=args.num_points,
                width=args.width,
                height=args.height,
                downsample_method=args.method,
                channel_strategy=args.channel,
                use_adaptive_downsample=not args.no_adaptive,
                max_points_per_pixel=args.max_points_per_pixel,
                min_points=args.min_points,
                max_points=args.max_points,
                svg_stroke_color=args.svg_stroke_color,
                svg_stroke_width=args.svg_stroke_width,
                svg_fill_color=args.svg_fill_color if args.svg_fill_color.lower() != 'none' else None,
                include_image=not args.no_image,
                include_spectrogram=args.spectrogram,
                spectrogram_width=args.spec_width,
                spectrogram_height=args.spec_height,
                n_fft=args.n_fft,
                hop_length=args.hop_length,
                spectrogram_cmap=args.cmap
            )
            
            result_dict = waveform_to_dict(result)
            print(json.dumps(result_dict, indent=2, ensure_ascii=False))
    except Exception as e:
        print(f"错误: {e}", file=sys.stderr)
        sys.exit(1)
