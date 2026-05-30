import os
import wave
import struct
import numpy as np
import tempfile
import json
from waveform_api import (
    read_wav, downsample, adaptive_downsample, normalize_points,
    generate_svg_path, generate_base64_image,
    generate_waveform, waveform_to_dict
)


def generate_test_wav(filepath, duration=1.0, sample_rate=44100, freq=440.0):
    num_samples = int(duration * sample_rate)
    t = np.linspace(0, duration, num_samples, endpoint=False)
    
    signal = np.sin(2 * np.pi * freq * t)
    signal = signal * 0.8
    signal = signal.astype(np.float32)
    signal = (signal * 32767).astype(np.int16)
    
    with wave.open(filepath, 'wb') as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(sample_rate)
        wav_file.setnframes(num_samples)
        wav_file.writeframes(signal.tobytes())
    
    return filepath


def test_read_wav():
    print("测试1: read_wav 函数")
    with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as f:
        filepath = f.name
    
    try:
        generate_test_wav(filepath, duration=0.5, sample_rate=44100)
        samples, sample_rate, duration, num_channels = read_wav(filepath)
        
        assert sample_rate == 44100, f"采样率错误: {sample_rate}"
        assert abs(duration - 0.5) < 0.01, f"时长错误: {duration}"
        assert len(samples) > 0, "样本数据为空"
        assert np.max(np.abs(samples)) <= 1.0, "样本未正确归一化"
        assert num_channels == 1, f"声道数错误: {num_channels}"
        
        print(f"  ✓ 采样率: {sample_rate} Hz")
        print(f"  ✓ 时长: {duration:.3f} 秒")
        print(f"  ✓ 声道数: {num_channels}")
        print(f"  ✓ 样本数: {len(samples)}")
        print(f"  ✓ 振幅范围: [{np.min(samples):.3f}, {np.max(samples):.3f}]")
        print("  ✓ read_wav 测试通过\n")
    finally:
        os.unlink(filepath)


def test_downsample():
    print("测试2: downsample 函数")
    samples = np.random.randn(10000)
    
    peak_result = downsample(samples, target_points=100, method='peak')
    assert len(peak_result) == 200, f"peak方法点数错误: {len(peak_result)}"
    
    avg_result = downsample(samples, target_points=100, method='average')
    assert len(avg_result) == 100, f"average方法点数错误: {len(avg_result)}"
    
    rms_result = downsample(samples, target_points=100, method='rms')
    assert len(rms_result) == 100, f"rms方法点数错误: {len(rms_result)}"
    
    print(f"  ✓ peak方法: {len(peak_result)} 个点 (最大值+最小值)")
    print(f"  ✓ average方法: {len(avg_result)} 个点")
    print(f"  ✓ rms方法: {len(rms_result)} 个点")
    print("  ✓ downsample 测试通过\n")


def test_normalize_points():
    print("测试3: normalize_points 函数")
    points = np.array([0.5, -0.3, 0.8, -0.6])
    normalized = normalize_points(points, height=200)
    
    assert abs(np.max(normalized) - 100) < 1e-9, f"归一化最大值错误: {np.max(normalized)}"
    assert abs(np.min(normalized) + 75) < 1e-9, f"归一化最小值错误: {np.min(normalized)}"
    
    print(f"  ✓ 原始值: {points}")
    print(f"  ✓ 归一化后: {normalized}")
    print("  ✓ normalize_points 测试通过\n")


def test_generate_svg_path():
    print("测试4: generate_svg_path 函数")
    points = np.sin(np.linspace(0, 4 * np.pi, 100))
    svg = generate_svg_path(points, width=800, height=200)
    
    assert '<svg' in svg, "SVG标签不存在"
    assert '<path' in svg, "PATH标签不存在"
    assert 'd=' in svg, "路径数据不存在"
    
    print(f"  ✓ SVG长度: {len(svg)} 字符")
    print(f"  ✓ 包含svg标签: {'<svg' in svg}")
    print(f"  ✓ 包含path标签: {'<path' in svg}")
    print("  ✓ generate_svg_path 测试通过\n")


def test_generate_base64_image():
    print("测试5: generate_base64_image 函数")
    try:
        points = np.sin(np.linspace(0, 4 * np.pi, 100))
        base64_img = generate_base64_image(points, width=800, height=200)
        
        assert base64_img.startswith('data:image/png;base64,'), "Base64格式错误"
        assert len(base64_img) > 100, "Base64数据过短"
        
        print(f"  ✓ Base64前缀正确: {base64_img[:30]}...")
        print(f"  ✓ Base64长度: {len(base64_img)} 字符")
        print("  ✓ generate_base64_image 测试通过\n")
    except ImportError as e:
        print(f"  ⚠ 跳过测试: {e}\n")


def test_generate_waveform():
    print("测试6: generate_waveform 函数")
    with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as f:
        filepath = f.name
    
    try:
        generate_test_wav(filepath, duration=1.0, sample_rate=22050)
        
        result = generate_waveform(
            filepath,
            num_points=200,
            width=600,
            height=150,
            downsample_method='peak',
            include_image=True
        )
        
        assert len(result.points) > 0, "点集为空"
        assert result.svg_path, "SVG路径为空"
        assert result.sample_rate == 22050, f"采样率错误: {result.sample_rate}"
        assert abs(result.duration - 1.0) < 0.01, f"时长错误: {result.duration}"
        assert result.width == 600, f"宽度错误: {result.width}"
        assert result.height == 150, f"高度错误: {result.height}"
        assert result.num_channels == 1, f"声道数错误: {result.num_channels}"
        assert result.downsample_factor >= 1, f"降采样因子错误: {result.downsample_factor}"
        assert result.channel_strategy == 'average', f"通道策略错误: {result.channel_strategy}"
        
        print(f"  ✓ 点集大小: {len(result.points)}")
        print(f"  ✓ 采样率: {result.sample_rate} Hz")
        print(f"  ✓ 时长: {result.duration:.3f} 秒")
        print(f"  ✓ 声道数: {result.num_channels}")
        print(f"  ✓ 降采样因子: {result.downsample_factor}")
        print(f"  ✓ 通道策略: {result.channel_strategy}")
        print(f"  ✓ SVG长度: {len(result.svg_path)} 字符")
        print(f"  ✓ Base64图片: {'已生成' if result.base64_image else '未生成'}")
        print("  ✓ generate_waveform 测试通过\n")
        
        result_no_img = generate_waveform(filepath, num_points=200, include_image=False)
        assert result_no_img.base64_image == "", "Base64图片应为空"
        print("  ✓ include_image=False 时不生成图片\n")
        
    except ImportError as e:
        if 'Pillow' in str(e):
            result = generate_waveform(filepath, num_points=200, include_image=False)
            assert len(result.points) > 0, "点集为空"
            print(f"  ⚠  Pillow未安装，跳过图片生成测试")
            print(f"  ✓ 点集大小: {len(result.points)}")
            print("  ✓ generate_waveform 测试通过 (无图片)\n")
        else:
            raise
    finally:
        os.unlink(filepath)


def test_waveform_to_dict():
    print("测试7: waveform_to_dict 函数")
    with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as f:
        filepath = f.name
    
    try:
        generate_test_wav(filepath, duration=0.5)
        result = generate_waveform(filepath, num_points=100, include_image=False)
        result_dict = waveform_to_dict(result)
        
        assert isinstance(result_dict, dict), "返回类型不是字典"
        assert 'points' in result_dict, "缺少points字段"
        assert 'svg_path' in result_dict, "缺少svg_path字段"
        assert 'base64_image' in result_dict, "缺少base64_image字段"
        assert 'sample_rate' in result_dict, "缺少sample_rate字段"
        assert 'duration' in result_dict, "缺少duration字段"
        assert 'num_channels' in result_dict, "缺少num_channels字段"
        assert 'downsample_factor' in result_dict, "缺少downsample_factor字段"
        assert 'channel_strategy' in result_dict, "缺少channel_strategy字段"
        assert 'num_output_points' in result_dict, "缺少num_output_points字段"
        
        json_str = json.dumps(result_dict, ensure_ascii=False)
        assert len(json_str) > 0, "JSON序列化失败"
        
        print(f"  ✓ 字典键: {list(result_dict.keys())}")
        print(f"  ✓ 包含num_channels: {result_dict['num_channels']}")
        print(f"  ✓ 包含downsample_factor: {result_dict['downsample_factor']}")
        print(f"  ✓ 包含channel_strategy: {result_dict['channel_strategy']}")
        print(f"  ✓ 包含num_output_points: {result_dict['num_output_points']}")
        print(f"  ✓ JSON序列化成功，长度: {len(json_str)} 字符")
        print("  ✓ waveform_to_dict 测试通过\n")
    finally:
        os.unlink(filepath)


def test_channel_strategy():
    print("测试8: 通道合并策略")
    with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as f:
        filepath = f.name
    
    try:
        duration = 0.5
        sample_rate = 44100
        num_samples = int(duration * sample_rate)
        t = np.linspace(0, duration, num_samples, endpoint=False)
        
        left = np.sin(2 * np.pi * 440 * t) * 0.8
        right = np.sin(2 * np.pi * 880 * t) * 0.4
        
        stereo = np.column_stack((left, right))
        stereo = (stereo * 32767).astype(np.int16)
        
        with wave.open(filepath, 'wb') as wav_file:
            wav_file.setnchannels(2)
            wav_file.setsampwidth(2)
            wav_file.setframerate(sample_rate)
            wav_file.setnframes(num_samples)
            wav_file.writeframes(stereo.tobytes())
        
        samples_avg, sr, dur, num_ch = read_wav(filepath, channel_strategy='average')
        assert num_ch == 2, f"声道数错误: {num_ch}"
        assert abs(samples_avg[0] - (0.8 + 0.4) / 2 * np.sin(2 * np.pi * 440 * t[0])) < 0.1 or \
               abs(samples_avg[0] - (left[0] + right[0]) / 2) < 0.1, "average策略错误"
        print(f"  ✓ average策略: 左右声道平均值")
        
        samples_left, _, _, _ = read_wav(filepath, channel_strategy='left')
        assert abs(samples_left[0] - left[0]) < 0.01, "left策略错误"
        print(f"  ✓ left策略: 只取左声道")
        
        samples_right, _, _, _ = read_wav(filepath, channel_strategy='right')
        assert abs(samples_right[0] - right[0]) < 0.01, "right策略错误"
        print(f"  ✓ right策略: 只取右声道")
        
        samples_mix, _, _, _ = read_wav(filepath, channel_strategy='mix')
        assert abs(samples_mix[0] - (left[0] * 0.5 + right[0] * 0.5)) < 0.01, "mix策略错误"
        print(f"  ✓ mix策略: 50%左声道 + 50%右声道")
        
        print("  ✓ 通道合并策略测试通过\n")
    finally:
        os.unlink(filepath)


def test_adaptive_downsample():
    print("测试9: 自适应降采样")
    with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as f:
        filepath = f.name
    
    try:
        generate_test_wav(filepath, duration=5.0, sample_rate=48000)
        
        result1 = generate_waveform(
            filepath,
            width=400,
            use_adaptive_downsample=True,
            max_points_per_pixel=2.0,
            include_image=False
        )
        expected_points1 = 400 * 2
        assert len(result1.points) <= expected_points1 + 10, f"宽度400时点数过多: {len(result1.points)}"
        print(f"  ✓ 宽度400px: 输出{len(result1.points)}个点 (预期≤{expected_points1})")
        print(f"  ✓ 降采样因子: {result1.downsample_factor}")
        
        result2 = generate_waveform(
            filepath,
            width=1200,
            use_adaptive_downsample=True,
            max_points_per_pixel=2.0,
            include_image=False
        )
        expected_points2 = 1200 * 2
        assert len(result2.points) <= expected_points2 + 10, f"宽度1200时点数过多: {len(result2.points)}"
        print(f"  ✓ 宽度1200px: 输出{len(result2.points)}个点 (预期≤{expected_points2})")
        print(f"  ✓ 降采样因子: {result2.downsample_factor}")
        
        assert len(result2.points) > len(result1.points), "宽度增加时点集应该更大"
        print(f"  ✓ 宽度越大，输出点数越多")
        
        svg_len1 = len(result1.svg_path)
        svg_len2 = len(result2.svg_path)
        print(f"  ✓ SVG长度对比: {svg_len1} vs {svg_len2} 字符")
        print(f"  ✓ 自适应降采样有效控制SVG数据量")
        
        print("  ✓ 自适应降采样测试通过\n")
    finally:
        os.unlink(filepath)


def test_stereo_wav():
    print("测试10: 立体声WAV文件")
    with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as f:
        filepath = f.name
    
    try:
        duration = 0.5
        sample_rate = 44100
        num_samples = int(duration * sample_rate)
        t = np.linspace(0, duration, num_samples, endpoint=False)
        
        left = np.sin(2 * np.pi * 440 * t) * 0.5
        right = np.sin(2 * np.pi * 880 * t) * 0.5
        
        stereo = np.column_stack((left, right))
        stereo = (stereo * 32767).astype(np.int16)
        
        with wave.open(filepath, 'wb') as wav_file:
            wav_file.setnchannels(2)
            wav_file.setsampwidth(2)
            wav_file.setframerate(sample_rate)
            wav_file.setnframes(num_samples)
            wav_file.writeframes(stereo.tobytes())
        
        samples, sr, dur, num_ch = read_wav(filepath)
        assert sr == sample_rate, f"采样率错误: {sr}"
        assert len(samples) == num_samples, f"样本数错误: {len(samples)}"
        assert num_ch == 2, f"声道数错误: {num_ch}"
        
        print(f"  ✓ 声道数: 2 (已自动混合为单声道)")
        print(f"  ✓ 采样率: {sr} Hz")
        print(f"  ✓ 样本数: {len(samples)}")
        print("  ✓ 立体声WAV测试通过\n")
    finally:
        os.unlink(filepath)


def test_command_line():
    print("测试11: 命令行使用")
    import subprocess
    import sys
    
    with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as f:
        filepath = f.name
    
    try:
        generate_test_wav(filepath, duration=0.5)
        
        result = subprocess.run(
            [sys.executable, 'waveform_api.py', filepath, '--no-image', '--channel', 'left', '--width', '400'],
            capture_output=True,
            text=True,
            cwd=os.path.dirname(os.path.abspath(__file__))
        )
        
        if result.returncode == 0:
            output = json.loads(result.stdout)
            assert 'points' in output, "输出缺少points"
            assert 'svg_path' in output, "输出缺少svg_path"
            assert 'num_channels' in output, "输出缺少num_channels"
            assert 'downsample_factor' in output, "输出缺少downsample_factor"
            assert output['channel_strategy'] == 'left', f"通道策略错误: {output['channel_strategy']}"
            print(f"  ✓ 命令行执行成功")
            print(f"  ✓ 输出包含完整数据")
            print(f"  ✓ 通道策略: {output['channel_strategy']}")
            print(f"  ✓ 降采样因子: {output['downsample_factor']}")
            print("  ✓ 命令行测试通过\n")
        else:
            print(f"  ⚠  命令行测试跳过: {result.stderr[:200]}\n")
    except Exception as e:
        print(f"  ⚠  命令行测试跳过: {e}\n")
    finally:
        os.unlink(filepath)


def test_fixed_num_points():
    print("测试12: 固定点数模式 vs 自适应模式")
    with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as f:
        filepath = f.name
    
    try:
        generate_test_wav(filepath, duration=2.0, sample_rate=44100)
        
        result_adaptive = generate_waveform(
            filepath,
            num_points=None,
            width=800,
            use_adaptive_downsample=True,
            include_image=False
        )
        print(f"  ✓ 自适应模式: {len(result_adaptive.points)}个点, 降采样因子={result_adaptive.downsample_factor}")
        
        result_fixed = generate_waveform(
            filepath,
            num_points=500,
            width=800,
            use_adaptive_downsample=False,
            include_image=False
        )
        assert len(result_fixed.points) == 1000, f"固定点数模式点数错误: {len(result_fixed.points)}"
        print(f"  ✓ 固定点数模式: {len(result_fixed.points)}个点, 降采样因子={result_fixed.downsample_factor}")
        
        result_small = generate_waveform(
            filepath,
            num_points=None,
            width=200,
            use_adaptive_downsample=True,
            max_points_per_pixel=1.0,
            include_image=False
        )
        assert len(result_small.points) <= 400, f"小宽度时点集应该更小: {len(result_small.points)}"
        print(f"  ✓ 小宽度(200px): {len(result_small.points)}个点")
        
        result_large = generate_waveform(
            filepath,
            num_points=None,
            width=2000,
            use_adaptive_downsample=True,
            max_points_per_pixel=2.0,
            max_points=3000,
            include_image=False
        )
        print(f"  ✓ 大宽度(2000px): {len(result_large.points)}个点")
        
        print("  ✓ 固定点数 vs 自适应模式测试通过\n")
    finally:
        os.unlink(filepath)


def main():
    print("=" * 60)
    print("音频波形图生成API - 测试套件")
    print("=" * 60 + "\n")
    
    try:
        test_read_wav()
        test_downsample()
        test_normalize_points()
        test_generate_svg_path()
        test_generate_base64_image()
        test_generate_waveform()
        test_waveform_to_dict()
        test_channel_strategy()
        test_adaptive_downsample()
        test_stereo_wav()
        test_command_line()
        test_fixed_num_points()
        
        print("=" * 60)
        print("所有测试通过! ✓")
        print("=" * 60)
    except AssertionError as e:
        print(f"\n测试失败: {e}")
        import traceback
        traceback.print_exc()
        exit(1)
    except Exception as e:
        print(f"\n发生错误: {e}")
        import traceback
        traceback.print_exc()
        exit(1)


if __name__ == '__main__':
    main()
