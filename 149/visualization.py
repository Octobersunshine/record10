import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


def plot_slice(image, slice_idx=None, axis=2, title=None, cmap='gray', ax=None):
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 8))
    
    if slice_idx is None:
        slice_idx = image.shape[axis] // 2
    
    if axis == 0:
        slc = image[slice_idx, :, :]
    elif axis == 1:
        slc = image[:, slice_idx, :]
    else:
        slc = image[:, :, slice_idx]
    
    im = ax.imshow(slc, cmap=cmap)
    ax.set_title(title or f'Slice {slice_idx}, axis={axis}')
    plt.colorbar(im, ax=ax)
    
    return ax


def plot_comparison_slices(images, titles, slice_idx=None, axis=2, figsize=(15, 5)):
    n = len(images)
    fig, axes = plt.subplots(1, n, figsize=figsize)
    
    if n == 1:
        axes = [axes]
    
    for i, (img, title) in enumerate(zip(images, titles)):
        plot_slice(img, slice_idx, axis, title, ax=axes[i])
    
    plt.tight_layout()
    return fig


def visualize_vessel_surface(binary_image, spacing=(1.0, 1.0, 1.0), figsize=(10, 8)):
    from skimage import measure
    
    verts, faces, _, _ = measure.marching_cubes(binary_image, level=0.5, spacing=spacing)
    
    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_trisurf(verts[:, 0], verts[:, 1], faces, verts[:, 2],
                    alpha=0.3, color='cyan', edgecolor='none')
    
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title('Vessel Surface')
    
    return fig


def visualize_centerline_3d(centerline_points, binary_image=None, spacing=(1.0, 1.0, 1.0), figsize=(12, 10), ax=None):
    if ax is None:
        fig = plt.figure(figsize=figsize)
        ax = fig.add_subplot(111, projection='3d')
    
    if binary_image is not None:
        from skimage import measure
        verts, faces, _, _ = measure.marching_cubes(binary_image, level=0.5, spacing=spacing)
        ax.plot_trisurf(verts[:, 0], verts[:, 1], faces, verts[:, 2],
                        alpha=0.1, color='cyan', edgecolor='none')
    
    points = np.array(centerline_points) * spacing
    ax.scatter(points[:, 0], points[:, 1], points[:, 2], c='red', s=10, label='Centerline')
    
    if len(points) > 1:
        ax.plot(points[:, 0], points[:, 1], points[:, 2], 'r-', linewidth=2, alpha=0.7)
    
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title('3D Centerline Visualization')
    ax.legend()
    
    return fig


def visualize_distance_transform(dt, binary_image, slice_idx=None, axis=2, figsize=(12, 5)):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
    
    if slice_idx is None:
        slice_idx = dt.shape[axis] // 2
    
    if axis == 0:
        dt_slice = dt[slice_idx, :, :]
        bin_slice = binary_image[slice_idx, :, :]
    elif axis == 1:
        dt_slice = dt[:, slice_idx, :]
        bin_slice = binary_image[:, slice_idx, :]
    else:
        dt_slice = dt[:, :, slice_idx]
        bin_slice = binary_image[:, :, slice_idx]
    
    im1 = ax1.imshow(bin_slice, cmap='gray')
    ax1.set_title(f'Binary Mask (Slice {slice_idx})')
    plt.colorbar(im1, ax=ax1)
    
    dt_masked = np.ma.masked_where(~bin_slice, dt_slice)
    im2 = ax2.imshow(dt_masked, cmap='jet')
    ax2.set_title('Distance Transform')
    plt.colorbar(im2, ax=ax2)
    
    plt.tight_layout()
    return fig


def create_summary_figure(original_image, binary_mask, distance_transform, centerline_points, 
                          slice_idx=None, save_path=None):
    fig = plt.figure(figsize=(16, 12))
    
    gs = fig.add_gridspec(3, 4, hspace=0.3, wspace=0.3)
    
    if slice_idx is None:
        slice_idx = original_image.shape[2] // 2
    
    ax1 = fig.add_subplot(gs[0, 0])
    plot_slice(original_image, slice_idx, title='Original Image', cmap='gray', ax=ax1)
    
    ax2 = fig.add_subplot(gs[0, 1])
    plot_slice(binary_mask, slice_idx, title='Binary Mask', cmap='gray', ax=ax2)
    
    ax3 = fig.add_subplot(gs[0, 2])
    dt_masked = np.ma.masked_where(~binary_mask[:, :, slice_idx], distance_transform[:, :, slice_idx])
    im3 = ax3.imshow(dt_masked, cmap='jet')
    ax3.set_title('Distance Transform')
    plt.colorbar(im3, ax=ax3)
    
    ax4 = fig.add_subplot(gs[0, 3])
    cl_slice = np.zeros_like(binary_mask[:, :, slice_idx])
    for point in centerline_points:
        if abs(point[2] - slice_idx) <= 1:
            x, y = int(point[0]), int(point[1])
            if 0 <= x < cl_slice.shape[0] and 0 <= y < cl_slice.shape[1]:
                cl_slice[x, y] = 1
    im4 = ax4.imshow(binary_mask[:, :, slice_idx], cmap='gray', alpha=0.5)
    ax4.imshow(np.ma.masked_where(cl_slice == 0, cl_slice), cmap='hot', vmin=0, vmax=1)
    ax4.set_title('Centerline Overlay')
    
    ax3d = fig.add_subplot(gs[1:, :], projection='3d')
    visualize_centerline_3d(centerline_points, binary_image=binary_mask, ax=ax3d)
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Summary figure saved to {save_path}")
    
    plt.close()
    return fig
