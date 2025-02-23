import torch
import torch.nn.functional as F
import numpy as np
from typing import Optional

class SimilarityCalibrator:
    def __init__(self, 
                 initial_temperature: float = 1.0,
                 target_mean: float = 0.55,
                 target_std: float = 0.3):
        self.temperature = initial_temperature
        self.target_mean = target_mean
        self.target_std = target_std
        self.reference_mean = None
        self.reference_std = None
        
    def load_reference_similarities(self, reference_path: str) -> None:
        try:
            saved_state = torch.load(reference_path)
            
            if 'metrics' in saved_state and 'raw_similarities' in saved_state['metrics']:
                all_sims = []
                for fold_sims in saved_state['metrics']['raw_similarities']:
                    if isinstance(fold_sims, (list, np.ndarray)):
                        all_sims.extend(fold_sims)
                
                if all_sims:
                    similarities = np.array(all_sims)
                    self.reference_mean = float(np.percentile(similarities, 50))
                    p95 = np.percentile(similarities, 95)
                    p05 = np.percentile(similarities, 5)
                    self.reference_std = float((p95 - p05) / 4)
                    print(f"Loaded reference stats - Median: {self.reference_mean:.4f}")
            else:
                self.reference_mean = 0.85
                self.reference_std = 0.05
                
        except Exception as e:
            print(f"Warning: Could not load reference similarities: {str(e)}")
            self.reference_mean = 0.85
            self.reference_std = 0.05
    
    def calibrate(self, similarities: torch.Tensor) -> torch.Tensor:
        try:
            if not isinstance(similarities, torch.Tensor):
                similarities = torch.tensor(similarities, dtype=torch.float32)
            
            similarities_np = similarities.numpy()
            calibrated = np.zeros_like(similarities_np)
            
            # Refined thresholds for better high-end differentiation
            super_high = 0.99   # New super high threshold
            extreme_high = 0.98
            very_high = 0.97
            high = 0.95
            med = 0.93
            
            # Masks for different ranges
            super_mask = similarities_np >= super_high
            extreme_mask = (similarities_np >= extreme_high) & (similarities_np < super_high)
            very_high_mask = (similarities_np >= very_high) & (similarities_np < extreme_high)
            high_mask = (similarities_np >= high) & (similarities_np < very_high)
            med_mask = (similarities_np >= med) & (similarities_np < high)
            low_mask = similarities_np < med
            
            # More granular scaling for top end
            calibrated[super_mask] = 0.80 + (similarities_np[super_mask] - super_high) * 15.0
            calibrated[extreme_mask] = 0.70 + (similarities_np[extreme_mask] - extreme_high) * 20.0
            calibrated[very_high_mask] = 0.55 + (similarities_np[very_high_mask] - very_high) * 15.0
            calibrated[high_mask] = 0.40 + (similarities_np[high_mask] - high) * 7.5
            calibrated[med_mask] = 0.30 + (similarities_np[med_mask] - med) * 5.0
            calibrated[low_mask] = 0.15 + (similarities_np[low_mask] - 0.9) * 3.0
            
            # Convert back to tensor
            calibrated = torch.tensor(calibrated, dtype=torch.float32)
            
            # Slightly lower clamp range
            calibrated = torch.clamp(calibrated, 0.15, 0.82)
            
            return calibrated
            
        except Exception as e:
            print(f"Error in calibration: {str(e)}")
            return similarities