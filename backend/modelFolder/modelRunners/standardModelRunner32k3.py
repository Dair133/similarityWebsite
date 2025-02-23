import torch
from torch import nn
import torch.nn.functional as F
import numpy as np
from typing import Optional, Dict, Any
# from calibrator import SimilarityCalibrator  # Commented out for now
from modelFolder.standardModelThree32k import SiameseNetwork  # Import the new model class
class ModelInference:
    def __init__(self, model_path: str, calibrator_path: Optional[str] = None):
        """Initialize the model inference class."""
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Load the model and move to device
        self.model = self.load_model(model_path)
        self.model.eval()
        
        # # Initialize calibrator with default values optimized for the new model
        # self.calibrator = SimilarityCalibrator(
        #     initial_temperature=2.0,    
        #     target_mean=0.5,           
        #     target_std=0.3            
        # )
        
        # if calibrator_path:
        #     self.load_calibrator_state(calibrator_path)

    def load_model(self, model_path: str) -> nn.Module:
        """Load the trained model from path."""
        try:

            model = SiameseNetwork(
                scibert_size=768,
                projection_size=64,
                metadata_size=3,
                paper_hidden_size=128,
                metadata_hidden_size=96,
                paper_dropout=0.3,
                metadata_dropout=0.3
            )
            state_dict = torch.load(model_path, map_location=self.device)
            
            # Handle different state dict formats
            if 'model_state_dict' in state_dict:
                model.load_state_dict(state_dict['model_state_dict'])
            else:
                model.load_state_dict(state_dict)
                
            model = model.to(self.device)
            return model
        except Exception as e:
            raise Exception(f"Error loading model: {str(e)}")
    
    # def load_calibrator_state(self, calibrator_path: str):
    #     """Load pre-computed calibrator statistics."""
    #     try:
    #         saved_state = torch.load(calibrator_path, map_location='cpu')
    #         if 'metrics' in saved_state:
    #             # Handle validation similarities format
    #             all_raw_similarities = np.concatenate(saved_state['metrics']['raw_similarities'])
    #             self.calibrator.fit(torch.tensor(all_raw_similarities))
    #         elif isinstance(saved_state, dict):
    #             # Direct calibrator state loading
    #             self.calibrator.__dict__.update(saved_state)
    #     except Exception as e:
    #         print(f"Warning: Could not load calibrator state: {str(e)}")

    def normalize_metadata(self, shared_data: Dict[str, Any]) -> torch.Tensor:
        """Normalize metadata features using standardization."""
        metadata = [
            float(shared_data.get('shared_references', 0)),
            float(shared_data.get('shared_citations', 0)),
            float(shared_data.get('shared_authors', 0))
        ]
        return torch.tensor(metadata, dtype=torch.float32)

    def predict_similarity(self,
                         paper1_SciBert: list,
                         paper2_SciBert: list,
                         shared_data: Dict[str, Any]) -> float:
        """
        Predict similarity between two papers.
        
        Args:
            paper1_SciBert: SciBERT embedding for first paper
            paper2_SciBert: SciBERT embedding for second paper
            shared_data: Dictionary containing metadata features
            
        Returns:
            float: Raw similarity score
        """
        try:
            with torch.no_grad():
                # Convert inputs to tensors and add batch dimension
                paper1_scibert = torch.tensor(paper1_SciBert, dtype=torch.float32).unsqueeze(0)
                paper2_scibert = torch.tensor(paper2_SciBert, dtype=torch.float32).unsqueeze(0)
                shared_features = self.normalize_metadata(shared_data).unsqueeze(0)
                
                # Move to device
                paper1_scibert = paper1_scibert.to(self.device)
                paper2_scibert = paper2_scibert.to(self.device)
                shared_features = shared_features.to(self.device)
                
                # Get embeddings and compute similarity
                embedding1, embedding2 = self.model(paper1_scibert, paper2_scibert, shared_features)
                
                # Compute raw cosine similarity
                raw_similarity = F.cosine_similarity(embedding1, embedding2).cpu()
                
                # Debug info
                print(f"Raw similarity: {raw_similarity.item():.4f}")
                
                return float(raw_similarity.item())
                
        except Exception as e:
            print(f"Error during prediction: {str(e)}")
            return 0.0  # Fallback if computation fails

    def get_embeddings(self,
                      paper1_SciBert: list,
                      paper2_SciBert: list,
                      shared_data: Dict[str, Any]) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Get the embeddings for two papers without computing similarity.
        Useful for downstream tasks or analysis.
        """
        with torch.no_grad():
            # Prepare inputs
            paper1_scibert = torch.tensor(paper1_SciBert, dtype=torch.float32).unsqueeze(0).to(self.device)
            paper2_scibert = torch.tensor(paper2_SciBert, dtype=torch.float32).unsqueeze(0).to(self.device)
            shared_features = self.normalize_metadata(shared_data).unsqueeze(0).to(self.device)
            
            # Get embeddings
            embedding1, embedding2 = self.model(paper1_scibert, paper2_scibert, shared_features)
            
            return embedding1.cpu(), embedding2.cpu()

def load_and_test_model(model_path: str, calibrator_path: Optional[str] = None) -> ModelInference:
    """
    Utility function to load and perform a quick test of the model.
    
    Args:
        model_path: Path to the saved model
        calibrator_path: Optional path to calibrator state (not used currently)
        
    Returns:
        ModelInference: Initialized and tested model inference object
    """
    try:
        # Initialize model
        model_runner = ModelInference(model_path, calibrator_path)
        
        # Create dummy inputs for testing
        dummy_scibert = [0.0] * 768  # SciBERT embedding size
        dummy_shared = {
            'shared_references': 1,
            'shared_citations': 1,
            'shared_authors': 1
        }
        
        # Test prediction
        similarity = model_runner.predict_similarity(
            dummy_scibert,
            dummy_scibert,
            dummy_shared
        )
        
        print(f"Model test successful. Raw similarity: {similarity:.4f}")
        return model_runner
        
    except Exception as e:
        raise Exception(f"Model loading/testing failed: {str(e)}")