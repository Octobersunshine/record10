import numpy as np
from typing import Tuple


class SpeciesDistributionData:
    def __init__(self, grid_size: int = 100, random_seed: int = 42):
        self.grid_size = grid_size
        self.random_seed = random_seed
        np.random.seed(random_seed)
        
    def generate_environmental_layers(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        x = np.linspace(0, 1, self.grid_size)
        y = np.linspace(0, 1, self.grid_size)
        xx, yy = np.meshgrid(x, y)
        
        temperature = 20 + 15 * np.sin(3 * xx) + 5 * np.cos(4 * yy)
        temperature += np.random.normal(0, 1, temperature.shape)
        
        precipitation = 100 + 80 * np.cos(2 * xx + yy) + 40 * np.sin(xx - 2 * yy)
        precipitation += np.random.normal(0, 5, precipitation.shape)
        precipitation = np.clip(precipitation, 0, None)
        
        coords = np.column_stack([xx.ravel(), yy.ravel()])
        
        return temperature, precipitation, coords
    
    def generate_species_presence(self, temperature: np.ndarray, 
                                  precipitation: np.ndarray,
                                  n_presence: int = 50,
                                  optimal_temp: float = 22,
                                  optimal_precip: float = 120,
                                  temp_tolerance: float = 8,
                                  precip_tolerance: float = 60) -> np.ndarray:
        suitability = np.exp(
            -((temperature - optimal_temp) ** 2) / (2 * temp_tolerance ** 2)
            - ((precipitation - optimal_precip) ** 2) / (2 * precip_tolerance ** 2)
        )
        
        suitability = suitability / np.max(suitability)
        
        flat_suitability = suitability.ravel()
        indices = np.arange(len(flat_suitability))
        
        selected_indices = np.random.choice(
            indices, 
            size=n_presence * 3,
            p=flat_suitability / np.sum(flat_suitability),
            replace=False
        )
        
        final_indices = np.random.choice(selected_indices, size=n_presence, replace=False)
        
        presence_coords = np.column_stack([
            final_indices // self.grid_size,
            final_indices % self.grid_size
        ])
        
        return presence_coords
    
    def generate_background_points(self, n_background: int = 1000) -> np.ndarray:
        x_indices = np.random.randint(0, self.grid_size, n_background)
        y_indices = np.random.randint(0, self.grid_size, n_background)
        return np.column_stack([x_indices, y_indices])
    
    def extract_env_values(self, points: np.ndarray, 
                            temperature: np.ndarray, 
                            precipitation: np.ndarray) -> np.ndarray:
        temp_values = temperature[points[:, 0], points[:, 1]]
        precip_values = precipitation[points[:, 0], points[:, 1]]
        return np.column_stack([temp_values, precip_values])
    
    def prepare_training_data(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        temp, precip, coords = self.generate_environmental_layers()
        
        presence_points = self.generate_species_presence(temp, precip)
        background_points = self.generate_background_points()
        
        presence_env = self.extract_env_values(presence_points, temp, precip)
        background_env = self.extract_env_values(background_points, temp, precip)
        
        return presence_env, background_env, temp, precip, presence_points
