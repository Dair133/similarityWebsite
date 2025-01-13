# Model Runner Scripts
# THI IS THE OLD ARCHITECTURE MODEL WITH SCIBETAT 90%
import torch
from torch import nn
import torch.nn.functional as F
import numpy as np
from modelFolder.model import SiameseNetwork

class ModelInference:
    def __init__(self, model_path: str):
        """
        Initialize the inference setup for the trained model.
        
        Args:
            model_path: Path to your .pth model file
        """
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Load the model
        self.model = self.load_model(model_path)
        self.model.eval()  # Set to evaluation mode
        
    def load_model(self, model_path: str):
        """Load the trained model from path"""
        try:
            model = SiameseNetwork(scibert_size=768)  
            
            # Load the trained weights
            model.load_state_dict(torch.load(model_path, map_location=self.device))
            model = model.to(self.device)
            return model
            
        except Exception as e:
            raise Exception(f"Error loading model: {str(e)}")

    def predict_similarity(self,
                         paper1_Citation_Count: int,
                         paper1_Reference_Count: int,
                         paper1_SciBert: list,
                         paper2_Citation_Count: int,
                         paper2_Reference_Count: int,
                         paper2_SciBert: list,
                         shared_author_count: int,
                         shared_reference_count: int,
                         shared_citation_count: int,
                         reference_cosine: float,
                         citation_cosine: float,
                         abstract_cosine: float) -> float:
        """
        Predict similarity between two papers using the trained model.
        
        Returns:
            float: Similarity score between 0 and 1
        """
        try:
            # print('attempting to predict score')
            with torch.no_grad():  
                # Convert SciBERT embeddings to tensors
                paper1_scibert = torch.tensor(paper1_SciBert, dtype=torch.float32).unsqueeze(0)
                paper2_scibert = torch.tensor(paper2_SciBert, dtype=torch.float32).unsqueeze(0)
                
                # Create shared features tensor
                shared_features = torch.tensor([
                    shared_reference_count,
                    reference_cosine,
                    shared_citation_count,
                    citation_cosine,
                    shared_author_count,
                    abstract_cosine
                ], dtype=torch.float32).unsqueeze(0)
                
                # Move to device
                paper1_scibert = paper1_scibert.to(self.device)
                paper2_scibert = paper2_scibert.to(self.device)
                shared_features = shared_features.to(self.device)
                
                # Get model predictions
                embedding1, embedding2 = self.model(paper1_scibert, paper2_scibert, shared_features)
                
                # Calculate similarity score
                similarity = F.cosine_similarity(embedding1, embedding2).item()
                
                return similarity
                
        except Exception as e:
            raise Exception(f"Error during prediction: {str(e)}")

# Example usage:
def run_similarity_prediction(model_path: str, 
                            paper1_data: dict,
                            paper2_data: dict,
                            shared_data: dict) -> float:
    """
    Wrapper function to run similarity prediction
    
    Args:
        model_path: Path to trained model
        paper1_data: Dictionary containing paper1's data
        paper2_data: Dictionary containing paper2's data
        shared_data: Dictionary containing shared metrics
    
    Returns:
        float: Similarity score
    """
    try:
        # Initialize model
        inference = ModelInference(model_path)
        
        # Run prediction
        similarity = inference.predict_similarity(
            paper1_Citation_Count=paper1_data['citation_count'],
            paper1_Reference_Count=paper1_data['reference_count'],
            paper1_SciBert=paper1_data['scibert'],
            paper2_Citation_Count=paper2_data['citation_count'],
            paper2_Reference_Count=paper2_data['reference_count'],
            paper2_SciBert=paper2_data['scibert'],
            shared_author_count=shared_data['author_count'],
            shared_reference_count=shared_data['reference_count'],
            shared_citation_count=shared_data['citation_count'],
            reference_cosine=shared_data['reference_cosine'],
            citation_cosine=shared_data['citation_cosine'],
            abstract_cosine=shared_data['abstract_cosine']
        )
        
        return similarity
        
    except Exception as e:
        print(f"Error running similarity prediction: {str(e)}")
        raise