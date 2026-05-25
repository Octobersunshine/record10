import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import pandas as pd


def create_suitability_colormap():
    colors = [
        (0.95, 0.95, 0.95),
        (0.7, 0.9, 0.7),
        (0.4, 0.8, 0.4),
        (0.2, 0.6, 0.2),
        (0.1, 0.4, 0.1)
    ]
    return LinearSegmentedColormap.from_list('suitability', colors, N=100)


def plot_environmental_layers(temperature: np.ndarray, precipitation: np.ndarray,
                              save_path: str = None):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    im1 = axes[0].imshow(temperature, cmap='RdYlBu_r', origin='lower')
    axes[0].set_title('Temperature Layer', fontsize=12, fontweight='bold')
    axes[0].set_xlabel('Longitude')
    axes[0].set_ylabel('Latitude')
    cbar1 = plt.colorbar(im1, ax=axes[0])
    cbar1.set_label('Temperature (°C)')
    
    im2 = axes[1].imshow(precipitation, cmap='Blues', origin='lower')
    axes[1].set_title('Precipitation Layer', fontsize=12, fontweight='bold')
    axes[1].set_xlabel('Longitude')
    axes[1].set_ylabel('Latitude')
    cbar2 = plt.colorbar(im2, ax=axes[1])
    cbar2.set_label('Precipitation (mm/year)')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()


def plot_species_presence(presence_points: np.ndarray, temperature: np.ndarray,
                          save_path: str = None):
    plt.figure(figsize=(8, 6))
    plt.imshow(temperature, cmap='RdYlBu_r', origin='lower', alpha=0.6)
    plt.scatter(presence_points[:, 1], presence_points[:, 0], 
                c='red', s=30, edgecolor='black', linewidth=0.5, 
                label='Species Presence', zorder=5)
    plt.title('Species Presence Points', fontsize=12, fontweight='bold')
    plt.xlabel('Longitude')
    plt.ylabel('Latitude')
    plt.legend()
    plt.colorbar(label='Temperature (°C)')
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()


def plot_suitability_distribution(suitability: np.ndarray, presence_points: np.ndarray = None,
                                  save_path: str = None, title: str = 'Predicted Habitat Suitability'):
    fig, ax = plt.subplots(figsize=(10, 8))
    
    cmap = create_suitability_colormap()
    im = ax.imshow(suitability, cmap=cmap, origin='lower', vmin=0, vmax=1)
    
    if presence_points is not None:
        ax.scatter(presence_points[:, 1], presence_points[:, 0], 
                   c='red', s=25, edgecolor='white', linewidth=0.8, 
                   label='Species Presence', zorder=5, alpha=0.8)
    
    ax.set_title(title, fontsize=14, fontweight='bold', pad=15)
    ax.set_xlabel('Longitude', fontsize=11)
    ax.set_ylabel('Latitude', fontsize=11)
    
    cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label('Habitat Suitability', fontsize=11)
    cbar.set_ticks([0, 0.25, 0.5, 0.75, 1.0])
    cbar.set_ticklabels(['Low', 'Moderate', 'Medium', 'High', 'Very High'])
    
    if presence_points is not None:
        ax.legend(loc='upper right', framealpha=0.9)
    
    ax.grid(alpha=0.2, linestyle='--')
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()


def plot_response_curves(model, temp_range: tuple = (5, 40), 
                         precip_range: tuple = (0, 300),
                         save_path: str = None):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    temp_values = np.linspace(temp_range[0], temp_range[1], 100)
    avg_precip = np.mean(precip_range)
    temp_env = np.column_stack([temp_values, np.full_like(temp_values, avg_precip)])
    temp_suitability = model.predict(temp_env)
    
    axes[0].plot(temp_values, temp_suitability, 'b-', linewidth=2)
    axes[0].fill_between(temp_values, temp_suitability, alpha=0.3, color='blue')
    axes[0].set_title('Temperature Response Curve', fontsize=12, fontweight='bold')
    axes[0].set_xlabel('Temperature (°C)')
    axes[0].set_ylabel('Suitability')
    axes[0].grid(True, alpha=0.3)
    axes[0].set_ylim(0, 1.05)
    
    precip_values = np.linspace(precip_range[0], precip_range[1], 100)
    avg_temp = np.mean(temp_range)
    precip_env = np.column_stack([np.full_like(precip_values, avg_temp), precip_values])
    precip_suitability = model.predict(precip_env)
    
    axes[1].plot(precip_values, precip_suitability, 'g-', linewidth=2)
    axes[1].fill_between(precip_values, precip_suitability, alpha=0.3, color='green')
    axes[1].set_title('Precipitation Response Curve', fontsize=12, fontweight='bold')
    axes[1].set_xlabel('Precipitation (mm/year)')
    axes[1].set_ylabel('Suitability')
    axes[1].grid(True, alpha=0.3)
    axes[1].set_ylim(0, 1.05)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()


def plot_comparison(temperature: np.ndarray, precipitation: np.ndarray,
                    suitability: np.ndarray, presence_points: np.ndarray,
                    save_path: str = None):
    fig = plt.figure(figsize=(16, 10))
    gs = fig.add_gridspec(2, 2, hspace=0.3, wspace=0.25)
    
    ax1 = fig.add_subplot(gs[0, 0])
    im1 = ax1.imshow(temperature, cmap='RdYlBu_r', origin='lower')
    ax1.set_title('Temperature', fontsize=12, fontweight='bold')
    ax1.set_xlabel('Longitude')
    ax1.set_ylabel('Latitude')
    plt.colorbar(im1, ax=ax1, fraction=0.046, pad=0.04).set_label('°C')
    
    ax2 = fig.add_subplot(gs[0, 1])
    im2 = ax2.imshow(precipitation, cmap='Blues', origin='lower')
    ax2.set_title('Precipitation', fontsize=12, fontweight='bold')
    ax2.set_xlabel('Longitude')
    ax2.set_ylabel('Latitude')
    plt.colorbar(im2, ax=ax2, fraction=0.046, pad=0.04).set_label('mm')
    
    ax3 = fig.add_subplot(gs[1, :])
    cmap = create_suitability_colormap()
    im3 = ax3.imshow(suitability, cmap=cmap, origin='lower', vmin=0, vmax=1)
    ax3.scatter(presence_points[:, 1], presence_points[:, 0], 
                c='red', s=20, edgecolor='white', linewidth=0.5, 
                label='Presence Points', zorder=5)
    ax3.set_title('Predicted Habitat Suitability Distribution', fontsize=14, fontweight='bold')
    ax3.set_xlabel('Longitude')
    ax3.set_ylabel('Latitude')
    ax3.legend(loc='upper right')
    cbar = plt.colorbar(im3, ax=ax3, fraction=0.046, pad=0.04)
    cbar.set_label('Suitability')
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()


def plot_model_comparison(best_suitability: np.ndarray, 
                          overfitted_suitability: np.ndarray,
                          presence_points: np.ndarray,
                          save_path: str = None):
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    cmap = create_suitability_colormap()
    
    im1 = axes[0].imshow(overfitted_suitability, cmap=cmap, origin='lower', vmin=0, vmax=1)
    axes[0].scatter(presence_points[:, 1], presence_points[:, 0], 
                    c='red', s=20, edgecolor='white', linewidth=0.5, zorder=5)
    axes[0].set_title('Overfitted Model\n(Product Features, No Regularization)', 
                      fontsize=12, fontweight='bold')
    axes[0].set_xlabel('Longitude')
    axes[0].set_ylabel('Latitude')
    axes[0].grid(alpha=0.2, linestyle='--')
    
    im2 = axes[1].imshow(best_suitability, cmap=cmap, origin='lower', vmin=0, vmax=1)
    axes[1].scatter(presence_points[:, 1], presence_points[:, 0], 
                    c='red', s=20, edgecolor='white', linewidth=0.5, zorder=5)
    axes[1].set_title('Regularized Model\n(Selected by AICc)', 
                      fontsize=12, fontweight='bold')
    axes[1].set_xlabel('Longitude')
    axes[1].set_ylabel('Latitude')
    axes[1].grid(alpha=0.2, linestyle='--')
    
    cbar = plt.colorbar(im2, ax=axes, fraction=0.02, pad=0.03)
    cbar.set_label('Habitat Suitability', fontsize=11)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()


def plot_aicc_results(results_df: pd.DataFrame, save_path: str = None):
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    feature_colors = {'linear': 'blue', 'quadratic': 'green', 'product': 'red'}
    
    for feature_type in results_df['feature_type'].unique():
        subset = results_df[results_df['feature_type'] == feature_type]
        axes[0].scatter(subset['l2_reg'], subset['aicc'], 
                        c=feature_colors.get(feature_type, 'gray'),
                        label=feature_type.capitalize(), alpha=0.7, s=80)
    
    axes[0].set_xlabel('L2 Regularization', fontsize=11)
    axes[0].set_ylabel('AICc Value', fontsize=11)
    axes[0].set_title('AICc vs L2 Regularization', fontsize=12, fontweight='bold')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    for feature_type in results_df['feature_type'].unique():
        subset = results_df[results_df['feature_type'] == feature_type]
        axes[1].scatter(subset['l1_reg'], subset['aicc'], 
                        c=feature_colors.get(feature_type, 'gray'),
                        label=feature_type.capitalize(), alpha=0.7, s=80)
    
    axes[1].set_xlabel('L1 Regularization', fontsize=11)
    axes[1].set_ylabel('AICc Value', fontsize=11)
    axes[1].set_title('AICc vs L1 Regularization', fontsize=12, fontweight='bold')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    feature_summary = results_df.groupby('feature_type').agg({
        'aicc': ['min', 'mean'],
        'n_params': 'mean'
    }).reset_index()
    
    x = np.arange(len(feature_summary))
    width = 0.35
    
    axes[2].bar(x - width/2, feature_summary[('aicc', 'min')], width, 
                label='Min AICc', color='steelblue', alpha=0.8)
    axes[2].bar(x + width/2, feature_summary[('aicc', 'mean')], width, 
                label='Mean AICc', color='lightcoral', alpha=0.8)
    
    axes[2].set_xlabel('Feature Type', fontsize=11)
    axes[2].set_ylabel('AICc Value', fontsize=11)
    axes[2].set_title('AICc by Feature Type', fontsize=12, fontweight='bold')
    axes[2].set_xticks(x)
    axes[2].set_xticklabels([ft.capitalize() for ft in feature_summary['feature_type']])
    axes[2].legend()
    axes[2].grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()


def plot_ensemble_weights(weights_df: pd.DataFrame, save_path: str = None):
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    colors = plt.cm.Set3(np.linspace(0, 1, len(weights_df)))
    
    axes[0].barh(weights_df['model'], weights_df['weight'], 
                 color=colors, edgecolor='black', alpha=0.8)
    axes[0].set_xlabel('Weight', fontsize=11)
    axes[0].set_title('Ensemble Model Weights', fontsize=12, fontweight='bold')
    axes[0].grid(True, alpha=0.3, axis='x')
    
    for i, (idx, row) in enumerate(weights_df.iterrows()):
        axes[0].text(row['weight'] + 0.01, i, f"{row['weight']:.3f}", 
                     va='center', fontweight='bold')
    
    x_pos = np.arange(len(weights_df))
    width = 0.35
    
    axes[1].bar(x_pos - width/2, weights_df['mean_auc'], width, 
                label='Mean AUC', color='steelblue', edgecolor='black', alpha=0.8)
    axes[1].bar(x_pos + width/2, weights_df['std_auc'], width, 
                label='Std AUC', color='lightcoral', edgecolor='black', alpha=0.8)
    
    axes[1].set_xlabel('Model', fontsize=11)
    axes[1].set_ylabel('AUC', fontsize=11)
    axes[1].set_title('Cross-Validation AUC Performance', fontsize=12, fontweight='bold')
    axes[1].set_xticks(x_pos)
    axes[1].set_xticklabels(weights_df['model'], rotation=45, ha='right')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()


def plot_ensemble_comparison(individual_grids: dict, ensemble_grid: np.ndarray,
                             presence_points: np.ndarray, weights_df: pd.DataFrame,
                             save_path: str = None):
    n_models = len(individual_grids) + 1
    n_cols = 3
    n_rows = (n_models + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 5, n_rows * 4.5))
    axes = axes.flatten()
    
    cmap = create_suitability_colormap()
    
    model_order = weights_df['model'].tolist()
    
    for idx, model_name in enumerate(model_order):
        if idx < len(axes) - 1:
            ax = axes[idx]
            im = ax.imshow(individual_grids[model_name], cmap=cmap, 
                           origin='lower', vmin=0, vmax=1)
            ax.scatter(presence_points[:, 1], presence_points[:, 0], 
                       c='red', s=15, edgecolor='white', linewidth=0.5, zorder=5)
            
            weight = weights_df[weights_df['model'] == model_name]['weight'].values[0]
            mean_auc = weights_df[weights_df['model'] == model_name]['mean_auc'].values[0]
            
            ax.set_title(f'{model_name}\nWeight={weight:.3f}, AUC={mean_auc:.3f}', 
                         fontsize=10, fontweight='bold')
            ax.set_xlabel('Longitude', fontsize=8)
            ax.set_ylabel('Latitude', fontsize=8)
            ax.tick_params(axis='both', which='major', labelsize=7)
    
    ensemble_ax = axes[len(model_order)]
    im = ensemble_ax.imshow(ensemble_grid, cmap=cmap, origin='lower', vmin=0, vmax=1)
    ensemble_ax.scatter(presence_points[:, 1], presence_points[:, 0], 
                        c='red', s=15, edgecolor='white', linewidth=0.5, zorder=5)
    ensemble_ax.set_title('ENSEMBLE (Weighted Average)', 
                          fontsize=11, fontweight='bold', color='darkblue')
    ensemble_ax.set_xlabel('Longitude', fontsize=8)
    ensemble_ax.set_ylabel('Latitude', fontsize=8)
    ensemble_ax.tick_params(axis='both', which='major', labelsize=7)
    
    for idx in range(len(model_order) + 1, len(axes)):
        axes[idx].axis('off')
    
    cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])
    cbar = fig.colorbar(im, cax=cbar_ax)
    cbar.set_label('Habitat Suitability', fontsize=10)
    
    plt.tight_layout(rect=[0, 0, 0.9, 1])
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()
