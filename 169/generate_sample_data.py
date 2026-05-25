import numpy as np
from scipy.ndimage import gaussian_filter


def generate_storm_cell(shape: tuple, center: tuple, sigma: float, amplitude: float) -> np.ndarray:
    h, w = shape
    y, x = np.ogrid[:h, :w]
    cy, cx = center
    gaussian = np.exp(-((x - cx)**2 + (y - cy)**2) / (2 * sigma**2))
    return gaussian * amplitude


def generate_radar_sequence(num_frames: int = 10, shape: tuple = (200, 200),
                            velocity: tuple = (3, 2), num_storms: int = 3) -> list:
    sequence = []
    h, w = shape

    storms = []
    for _ in range(num_storms):
        storm = {
            'x0': np.random.uniform(30, w - 30),
            'y0': np.random.uniform(30, h - 30),
            'sigma': np.random.uniform(10, 25),
            'amplitude': np.random.uniform(20, 50),
            'vx': np.random.uniform(-2, 2) + velocity[0],
            'vy': np.random.uniform(-2, 2) + velocity[1],
            'growth_rate': np.random.uniform(0.98, 1.02)
        }
        storms.append(storm)

    for t in range(num_frames):
        frame = np.zeros(shape, dtype=np.float32)
        for storm in storms:
            cx = storm['x0'] + storm['vx'] * t
            cy = storm['y0'] + storm['vy'] * t
            sigma = storm['sigma'] * (storm['growth_rate'] ** t)
            amplitude = storm['amplitude'] * (storm['growth_rate'] ** t)

            if 0 <= cx < w and 0 <= cy < h:
                frame += generate_storm_cell(shape, (cy, cx), sigma, amplitude)

        frame = np.clip(frame, 0, 70)
        sequence.append(frame)

    return sequence


def add_noise(sequence: list, noise_level: float = 2) -> list:
    noisy_sequence = []
    for frame in sequence:
        noise = np.random.normal(0, noise_level, frame.shape)
        noisy_frame = np.clip(frame + noise, 0, None)
        noisy_sequence.append(noisy_frame)
    return noisy_sequence


def generate_rotation_sequence(num_frames: int = 8, shape: tuple = (200, 200),
                               center: tuple = (100, 100), angular_speed: float = 0.1) -> list:
    sequence = []
    h, w = shape
    cy, cx = center
    y, x = np.ogrid[:h, :w]

    for t in range(num_frames):
        frame = np.zeros(shape, dtype=np.float32)

        r = 40
        for i in range(3):
            angle = angular_speed * t + i * 2 * np.pi / 3
            sx = cx + r * np.cos(angle)
            sy = cy + r * np.sin(angle)
            frame += generate_storm_cell(shape, (sy, sx), 12, 40)

        sequence.append(frame)

    return sequence


def generate_convective_line(num_frames: int = 10, shape: tuple = (200, 200),
                             velocity: float = 2.5) -> list:
    sequence = []
    h, w = shape

    for t in range(num_frames):
        frame = np.zeros(shape, dtype=np.float32)

        line_y = h // 2 + velocity * t
        line_width = 8

        y, x = np.ogrid[:h, :w]
        line_mask = np.abs(y - line_y) < line_width
        frame[line_mask] = 35

        for i in range(5):
            cx = 30 + i * 35
            cy = line_y
            frame += generate_storm_cell(shape, (cy, cx), 10, 20)

        frame = gaussian_filter(frame, sigma=2)
        sequence.append(frame)

    return sequence


if __name__ == '__main__':
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap

    radar_cmap = ListedColormap([
        '#FFFFFF', '#80FF80', '#00FF00', '#00C000',
        '#008000', '#FFFF00', '#FFC000', '#FF8000',
        '#FF0000', '#C00000', '#800000', '#FF00FF'
    ])

    seq = generate_radar_sequence(num_frames=6, num_storms=4)
    seq = add_noise(seq, noise_level=1)

    fig, axes = plt.subplots(2, 3, figsize=(12, 8))
    for i, (ax, frame) in enumerate(zip(axes.flat, seq)):
        im = ax.imshow(frame, cmap=radar_cmap, vmin=0, vmax=60)
        ax.set_title(f'Frame {i}')
        ax.axis('off')

    plt.tight_layout()
    plt.savefig('sample_radar_sequence.png', dpi=100)
    plt.close()

    np.save('radar_sequence.npy', np.array(seq))
    print('Sample data generated: radar_sequence.npy and sample_radar_sequence.png')
