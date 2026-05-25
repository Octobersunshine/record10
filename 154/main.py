import numpy as np
import pandas as pd
from maxent_model import MaxEntModel, MaxEntModelSelector
from ensemble_model import SpeciesDistributionEnsemble
from data_generator import SpeciesDistributionData
from visualization import (
    plot_environmental_layers,
    plot_species_presence,
    plot_suitability_distribution,
    plot_response_curves,
    plot_comparison,
    plot_model_comparison,
    plot_aicc_results,
    plot_ensemble_weights,
    plot_ensemble_comparison
)


def main():
    print("=" * 75)
    print("Species Distribution Modeling - Ensemble Approach with Cross-Validation")
    print("=" * 75)
    
    print("\n[1/9] Generating environmental data...")
    grid_size = 100
    data_generator = SpeciesDistributionData(grid_size=grid_size, random_seed=42)
    
    presence_env, background_env, temperature, precipitation, presence_points = \
        data_generator.prepare_training_data()
    
    print(f"   - Presence points: {presence_env.shape[0]}")
    print(f"   - Background points: {background_env.shape[0]}")
    print(f"   - Grid size: {grid_size}x{grid_size}")
    
    print("\n[2/9] Visualizing environmental layers...")
    plot_environmental_layers(temperature, precipitation, save_path='environmental_layers.png')
    
    print("\n[3/9] Visualizing species presence points...")
    plot_species_presence(presence_points, temperature, save_path='presence_points.png')
    
    print("\n[4/9] Performing AICc model selection for MaxEnt...")
    selector = MaxEntModelSelector()
    
    best_params = selector.grid_search(
        presence_data=presence_env,
        background_data=background_env,
        feature_types=['linear', 'quadratic'],
        l1_reg_values=[0.0, 0.01, 0.1],
        l2_reg_values=[0.1, 0.5, 1.0]
    )
    
    print(f"\n   Best MaxEnt model (AICc): {best_params['feature_type']} features, "
          f"L1={best_params['l1_reg']}, L2={best_params['l2_reg']}")
    print(f"   AICc={best_params['aicc']:.2f}, params={best_params['n_params']}")
    
    print("\n[5/9] Generating AICc comparison plot...")
    results_df = pd.DataFrame(selector.get_results_summary())
    plot_aicc_results(results_df, save_path='aicc_comparison.png')
    
    print("\n[6/9] Training Ensemble Model (5-fold CV for weight assignment)...")
    ensemble = SpeciesDistributionEnsemble(n_splits=5, random_state=42)
    ensemble.fit(presence_env, background_env)
    ensemble.print_weights()
    
    print("\n[7/9] Predicting habitat suitability...")
    grid_env = np.column_stack([temperature.ravel(), precipitation.ravel()])
    
    ensemble_prediction, individual_predictions = ensemble.predict(grid_env)
    ensemble_grid = ensemble_prediction.reshape(temperature.shape)
    
    individual_grids = {}
    for model_name, pred in individual_predictions.items():
        individual_grids[model_name] = pred.reshape(temperature.shape)
    
    best_maxent_model = selector.best_model
    best_maxent_suitability = best_maxent_model.predict(grid_env)
    best_maxent_grid = best_maxent_suitability.reshape(temperature.shape)
    
    print("\n[8/9] Generating ensemble visualization...")
    
    print("   - Ensemble weights plot")
    weights_df = pd.DataFrame(ensemble.get_weights_summary())
    plot_ensemble_weights(weights_df, save_path='ensemble_weights.png')
    
    print("   - Ensemble suitability distribution")
    plot_suitability_distribution(
        ensemble_grid, 
        presence_points,
        save_path='ensemble_suitability.png',
        title='Ensemble Model Prediction (Weighted Average)'
    )
    
    print("   - Ensemble model comparison")
    plot_ensemble_comparison(
        individual_grids,
        ensemble_grid,
        presence_points,
        weights_df,
        save_path='ensemble_comparison.png'
    )
    
    print("   - Best MaxEnt vs Ensemble comparison")
    plot_model_comparison(
        ensemble_grid,
        best_maxent_grid,
        presence_points,
        save_path='maxent_vs_ensemble.png'
    )
    
    print("   - Individual model response curves")
    plot_response_curves(
        best_maxent_model, 
        temp_range=(5, 40), 
        precip_range=(0, 300),
        save_path='best_maxent_response_curves.png'
    )
    
    print("\n[9/9] Generating comprehensive summary...")
    plot_comparison(
        temperature, 
        precipitation, 
        ensemble_grid, 
        presence_points,
        save_path='comprehensive_ensemble_plot.png'
    )
    
    print("\n" + "=" * 75)
    print("Analysis Completed Successfully!")
    print("=" * 75)
    print("\nGenerated files:")
    print("  - environmental_layers.png")
    print("  - presence_points.png")
    print("  - aicc_comparison.png")
    print("  - ensemble_weights.png")
    print("  - ensemble_suitability.png")
    print("  - ensemble_comparison.png")
    print("  - maxent_vs_ensemble.png")
    print("  - best_maxent_response_curves.png")
    print("  - comprehensive_ensemble_plot.png")
    print("\nEnsemble Model Summary:")
    print(weights_df.to_string(index=False))
    print("\n" + "=" * 75)
    
    return ensemble, ensemble_grid, weights_df


if __name__ == "__main__":
    main()
