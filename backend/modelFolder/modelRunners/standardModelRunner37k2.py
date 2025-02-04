import torch
from torch import nn
import torch.nn.functional as F
import numpy as np
from typing import Optional, Dict, Any
from calibrator import SimilarityCalibrator
from modelFolder.standardModelRunner37k2 import SiameseNetwork

class ModelInference:
    def __init__(self, model_path: str, calibrator_path: Optional[str] = None):
        """
        Initialize the inference setup for the trained model.
        
        Args:
            model_path: Path to your .pth model file
            calibrator_path: Optional path to saved calibrator state
        """
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Load the model and move to device
        self.model = self.load_model(model_path)
        self.model.eval()
        
        # Initialize calibrator
        self.calibrator = None
        if calibrator_path:
            self.calibrator = self.load_calibrator(calibrator_path)
    
    def load_model(self, model_path: str) -> nn.Module:
        """Load the trained model from path"""
        try:
            model = SiameseNetwork(scibert_size=768)
            state_dict = torch.load(model_path, map_location=self.device)
            model.load_state_dict(state_dict)
            model = model.to(self.device)
            return model
            
        except Exception as e:
            raise Exception(f"Error loading model: {str(e)}")
    
    def load_calibrator(self, calibrator_path: str) -> Optional[SimilarityCalibrator]:
        """Load the calibrator if available"""
        try:
            calibrator_state = torch.load(calibrator_path, map_location=self.device)
            calibrator = SimilarityCalibrator(
                initial_temperature=2.0,
                target_mean=0.5,
                target_std=0.25
            )
            calibrator.__dict__.update(calibrator_state)
            return calibrator
        except Exception as e:
            print(f"Warning: Could not load calibrator: {str(e)}")
            return None

    def normalize_metadata(self, shared_data: Dict[str, Any]) -> torch.Tensor:
        """
        Normalize metadata features using the same strategy as training
        """
        metadata = [
            shared_data['reference_count'],
            shared_data['reference_cosine'],
            shared_data['citation_count'],
            shared_data['citation_cosine'],
            shared_data['author_count'],
            shared_data['abstract_cosine']
        ]
        return torch.tensor(metadata, dtype=torch.float32)

    def predict_similarity(self,
                         paper1_SciBert: list,
                         paper2_SciBert: list,
                         shared_data: Dict[str, Any]) -> float:
        """
        Predict similarity between two papers using the trained model.
        
        Args:
            paper1_SciBert: SciBERT embedding for paper 1
            paper2_SciBert: SciBERT embedding for paper 2
            shared_data: Dictionary containing shared features:
                - reference_count: int
                - reference_cosine: float
                - citation_count: int
                - citation_cosine: float
                - author_count: int
                - abstract_cosine: float
        
        Returns:
            float: Similarity score between 0 and 1
        """
        try:
            with torch.no_grad():
                # Convert SciBERT embeddings to tensors
                paper1_scibert = torch.tensor(paper1_SciBert, dtype=torch.float32).unsqueeze(0)
                paper2_scibert = torch.tensor(paper2_SciBert, dtype=torch.float32).unsqueeze(0)
                
                # Create and normalize shared features tensor
                shared_features = self.normalize_metadata(shared_data).unsqueeze(0)
                
                # Move tensors to device
                paper1_scibert = paper1_scibert.to(self.device)
                paper2_scibert = paper2_scibert.to(self.device)
                shared_features = shared_features.to(self.device)
                
                # Get model predictions
                embedding1, embedding2 = self.model(paper1_scibert, paper2_scibert, shared_features)
                
                # Calculate raw similarity score
                raw_similarity = F.cosine_similarity(embedding1, embedding2).item()
                
                # Apply calibration if available
                if self.calibrator is not None:
                    similarity = self.calibrator.calibrate(
                        torch.tensor([raw_similarity])
                    ).item()
                    return similarity
                
                return raw_similarity
                
        except Exception as e:
            raise Exception(f"Error during prediction: {str(e)}")

def run_similarity_prediction(
    model_path: str,
    paper1_data: Dict[str, Any],
    paper2_data: Dict[str, Any],
    shared_data: Dict[str, Any],
    calibrator_path: Optional[str] = None
) -> float:
    """
    Wrapper function to run similarity prediction
    
    Args:
        model_path: Path to trained model
        paper1_data: Dictionary containing paper1's SciBERT embedding
        paper2_data: Dictionary containing paper2's SciBERT embedding
        shared_data: Dictionary containing shared metrics
        calibrator_path: Optional path to saved calibrator state
    
    Returns:
        float: Similarity score
    """
    try:
        # Initialize model with optional calibrator
        inference = ModelInference(model_path, calibrator_path)
        
        # Run prediction
        similarity = inference.predict_similarity(
            paper1_SciBert=paper1_data['scibert'],
            paper2_SciBert=paper2_data['scibert'],
            shared_data=shared_data
        )
        
        return similarity
        
    except Exception as e:
        print(f"Error running similarity prediction: {str(e)}")
        raise

# Example usage:
if __name__ == "__main__":
    # Example data structure
    paper1 = {
        'scibert': [0.1] * 768  # SciBERT embedding
    }
    
    paper2 = {
        'scibert': [0.2] * 768  # SciBERT embedding
    }
    
    shared = {
        'reference_count': 5,
        'reference_cosine': 0.3,
        'citation_count': 10,
        'citation_cosine': 0.4,
        'author_count': 2,
        'abstract_cosine': 0.6
    }
    
    similarity = run_similarity_prediction(
        'path/to/model.pth',
        paper1,
        paper2,
        shared,
        calibrator_path='path/to/calibrator.pth'  # Optional
    )
    print(f"Predicted similarity: {similarity:.4f}")