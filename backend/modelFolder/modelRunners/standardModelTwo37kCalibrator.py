# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# import numpy as np
# from typing import Tuple, Optional

# class SimilarityCalibrator:
#     def __init__(self, 
#                  initial_temperature: float = 2.0,
#                  target_mean: float = 0.5,
#                  target_std: float = 0.25):
#         self.temperature = initial_temperature
#         self.target_mean = target_mean
#         self.target_std = target_std
#         self.running_mean = None
#         self.running_std = None
        
#     def fit(self, val_similarities: torch.Tensor) -> None:
#         """Initialize or update calibration parameters"""
#         with torch.no_grad():
#             # Always update statistics when fit is called
#             current_mean = val_similarities.mean().item()
#             current_std = val_similarities.std().item()
            
#             if self.running_mean is None:
#                 self.running_mean = current_mean
#                 self.running_std = current_std
#             else:
#                 # Using exponential moving average for stability
#                 self.running_mean = 0.9 * self.running_mean + 0.1 * current_mean
#                 self.running_std = 0.9 * self.running_std + 0.1 * current_std
    
#     def calibrate(self, similarities: torch.Tensor) -> torch.Tensor:
#         """Enhanced calibration with guaranteed spread"""
#         with torch.no_grad():
#             # Check if we need to initialize statistics
#             if self.running_mean is None:
#                 self.fit(similarities)
            
#             # Now perform the calibration
#             centered = similarities - self.running_mean
            
#             # Apply aggressive spread transformation
#             spread = torch.sigmoid(centered * 4) * 2 - 1
#             spread = torch.sign(spread) * torch.abs(spread).pow(0.3)
            
#             # Normalize to target distribution
#             min_val, max_val = spread.min(), spread.max()
#             normalized = (spread - min_val) / (max_val - min_val + 1e-8)
            
#             # Scale to target range while preserving order
#             calibrated = normalized * 0.8 + 0.1
            
#             return calibrated

# def evaluate_fold(model, val_loader, calibrator=None):
#     device = next(model.parameters()).device
#     model.eval()
#     raw_similarities = []
#     calibrated_similarities = []
    
#     with torch.no_grad():
#         for paper1, paper2, shared_features in val_loader:
#             paper1 = paper1.to(device)
#             paper2 = paper2.to(device)
#             shared_features = shared_features.to(device)
            
#             embedding1, embedding2 = model(paper1, paper2, shared_features)
#             similarity = F.cosine_similarity(embedding1, embedding2, dim=1).cpu()
#             raw_similarities.extend(similarity.numpy())
            
#             # Apply calibration if calibrator is provided
#             if calibrator is not None:
#                 calibrated = calibrator.calibrate(similarity)
#                 calibrated_similarities.extend(calibrated.numpy())
    
#     raw_similarities = np.array(raw_similarities)
    
#     if calibrator is not None:
#         # Fit calibrator on new validation data
#         calibrator.fit(torch.tensor(raw_similarities))
#         return raw_similarities, np.array(calibrated_similarities)
        
#     return raw_similarities