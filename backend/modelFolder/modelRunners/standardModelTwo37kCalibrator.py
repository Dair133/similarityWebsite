import torch
import torch.nn.functional as F
import numpy as np
from typing import Optional

class SimilarityCalibrator:
    def __init__(self, 
                 initial_temperature: float = 2.0,
                 target_mean: float = 0.5,
                 target_std: float = 0.25):
        self.temperature = initial_temperature
        self.target_mean = target_mean
        self.target_std = target_std
        self.running_mean = None
        self.running_std = None
        self.momentum = 0.1
        
    def fit(self, val_similarities: torch.Tensor) -> None:
        """Compute calibration parameters using validation similarities"""
        with torch.no_grad():
            # Ensure input is a tensor and on CPU
            if not isinstance(val_similarities, torch.Tensor):
                val_similarities = torch.tensor(val_similarities)
            val_similarities = val_similarities.float().cpu()
            
            # Remove any NaN values
            val_similarities = val_similarities[~torch.isnan(val_similarities)]
            
            if len(val_similarities) == 0:
                return
                
            current_mean = val_similarities.mean().item()
            current_std = val_similarities.std().item()
            
            # Initialize running statistics if needed
            if self.running_mean is None:
                self.running_mean = current_mean
                self.running_std = max(current_std, 1e-6)  # Prevent zero std
            else:
                # Update running statistics with momentum
                self.running_mean = (1 - self.momentum) * self.running_mean + self.momentum * current_mean
                self.running_std = max((1 - self.momentum) * self.running_std + self.momentum * current_std, 1e-6)
            
            # Adjust temperature based on distribution
            if self.running_std > 0:
                self.temperature = max(1.0, self.running_std / self.target_std)
    
    def calibrate(self, similarities: torch.Tensor) -> torch.Tensor:
        """Apply calibration to raw similarity scores"""
        with torch.no_grad():
            # Ensure input is a tensor
            if not isinstance(similarities, torch.Tensor):
                similarities = torch.tensor(similarities)
            similarities = similarities.float()
            
            # If running statistics aren't initialized, return raw similarities
            if self.running_mean is None or self.running_std is None:
                return torch.clamp(similarities, 0, 1)
            
            # Temperature scaling
            scaled = similarities / self.temperature
            
            # Z-score normalization
            normalized = (scaled - self.running_mean) / (self.running_std + 1e-8)
            
            # Transform to target distribution
            calibrated = normalized * self.target_std + self.target_mean
            
            # Ensure outputs are between 0 and 1
            return torch.clamp(calibrated, 0, 1)

def evaluate_fold(model, val_loader, calibrator=None):
    device = next(model.parameters()).device
    model.eval()
    raw_similarities = []
    calibrated_similarities = []
    
    with torch.no_grad():
        for paper1, paper2, shared_features in val_loader:
            paper1 = paper1.to(device)
            paper2 = paper2.to(device)
            shared_features = shared_features.to(device)
            
            embedding1, embedding2 = model(paper1, paper2, shared_features)
            similarity = F.cosine_similarity(embedding1, embedding2, dim=1).cpu()
            raw_similarities.extend(similarity.numpy())
            
            # Apply calibration if calibrator is provided
            if calibrator is not None:
                calibrated = calibrator.calibrate(similarity)
                calibrated_similarities.extend(calibrated.numpy())
    
    raw_similarities = np.array(raw_similarities)
    
    if calibrator is not None:
        # Fit calibrator on new validation data
        calibrator.fit(torch.tensor(raw_similarities))
        return raw_similarities, np.array(calibrated_similarities)
        
    return raw_similarities