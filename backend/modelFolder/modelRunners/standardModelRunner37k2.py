# # OLD MODEL RUNNER FILE
# import torch
# from torch import nn
# import torch.nn.functional as F
# import numpy as np
# from typing import Optional, Dict, Any
# from backend.modelFolder.modelRunners.modelDependencies.standardModelTwo37kCalibrator import SimilarityCalibrator
# from modelFolder.standardModelTwo37k import SiameseNetwork

# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# import numpy as np



# class ModelInference:
#     def __init__(self, model_path: str, calibrator_path: Optional[str] = None):
#         self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
#         # Load the model and move to device
#         self.model = self.load_model(model_path)
#         self.model.eval()
        
#         # Initialize calibrator and load reference data
#         self.calibrator = SimilarityCalibrator(
#     initial_temperature=2.5,    # Increased from 1.5
#     target_mean=0.5,           # Changed from 0.6
#     target_std=0.35           # Increased from 0.15
# )
        
#         if calibrator_path:
#             self.calibrator.load_reference_similarities(calibrator_path)
#     def load_model(self, model_path: str) -> nn.Module:
#         """Load the trained model from path"""
#         try:
#             model = SiameseNetwork(scibert_size=768)
#             state_dict = torch.load(model_path, map_location=self.device)
#             model.load_state_dict(state_dict)
#             model = model.to(self.device)
#             return model
#         except Exception as e:
#             raise Exception(f"Error loading model: {str(e)}")
    
#     def load_calibrator_state(self, calibrator_path: str):
#         """Load pre-computed calibrator statistics"""
#         try:
#             saved_state = torch.load(calibrator_path, map_location='cpu')
#             if 'metrics' in saved_state:
#                 # Get all raw similarities from validation
#                 all_raw_similarities = np.concatenate(saved_state['metrics']['raw_similarities'])
#                 # Fit calibrator on these similarities
#                 self.calibrator.fit(torch.tensor(all_raw_similarities))
#             elif isinstance(saved_state, dict):
#                 # Directly load calibrator state
#                 self.calibrator.__dict__.update(saved_state)
#         except Exception as e:
#             print(f"Warning: Could not load calibrator state: {str(e)}")

#     def normalize_metadata(self, shared_data: Dict[str, Any]) -> torch.Tensor:
#         """Normalize metadata features using the same strategy as training"""
#         metadata = [
#             float(shared_data['reference_count']),
#             float(shared_data['reference_cosine']),
#             float(shared_data['citation_count']),
#             float(shared_data['citation_cosine']),
#             float(shared_data['author_count']),
#             float(shared_data['abstract_cosine'])
#         ]
#         return torch.tensor(metadata, dtype=torch.float32)

#     def predict_similarity(self,
#                          paper1_SciBert: list,
#                          paper2_SciBert: list,
#                          shared_data: Dict[str, Any]) -> float:
#         try:
#             with torch.no_grad():
#                 # Convert inputs to tensors
#                 paper1_scibert = torch.tensor(paper1_SciBert, dtype=torch.float32).unsqueeze(0)
#                 paper2_scibert = torch.tensor(paper2_SciBert, dtype=torch.float32).unsqueeze(0)
#                 shared_features = self.normalize_metadata(shared_data).unsqueeze(0)
                
#                 # Move to device
#                 paper1_scibert = paper1_scibert.to(self.device)
#                 paper2_scibert = paper2_scibert.to(self.device)
#                 shared_features = shared_features.to(self.device)
                
#                 # Get embeddings and compute similarity
#                 embedding1, embedding2 = self.model(paper1_scibert, paper2_scibert, shared_features)
#                 raw_similarity = F.cosine_similarity(embedding1, embedding2).cpu()
                
#                 # Debug prints
#                 print(f"Raw similarity: {raw_similarity.item():.4f}")
                
#                 # Apply calibration
#                 calibrated_similarity = self.calibrator.calibrate(raw_similarity)
#                 print(f"Calibrated similarity: {calibrated_similarity.item():.4f}")
                
#                 return float(calibrated_similarity.item())
                
#         except Exception as e:
#             print(f"Error during prediction: {str(e)}")
#             return float(raw_similarity.item())  # Fallback to raw similarity if calibra