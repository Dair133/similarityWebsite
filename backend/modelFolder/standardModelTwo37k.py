from typing import Optional
import torch
from torch import nn
import torch.nn.functional as F
import pandas as pd
import numpy as np
import os
from torch.utils.data import Dataset, DataLoader, random_split
from sklearn.model_selection import KFold
from tqdm import tqdm
import logging
from datetime import datetime
from torch.optim.lr_scheduler import ReduceLROnPlateau
import time
import math
from modelRunners.standardModelRunner37k2 import SimilarityCalibrator
# # Set up logging
# logging.basicConfig(
#     level=logging.INFO,
#     format='%(asctime)s - %(levelname)s - %(message)s',
#     # handlers=[
#     #     logging.FileHandler(f'training_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'),
#     #     logging.StreamHandler()
#     # ]
# )

# Temp of 0.3 seems to give highest val Loss
# Increasing droout seems to have small effect on incresing train loss
class NTXentLoss(nn.Module):
    def __init__(self, temperature=0.23):
        super().__init__()
        self.temperature = temperature
        
    def forward(self, z_i, z_j):
        batch_size = z_i.size(0)
        
        # Normalize embeddings
        z_i = F.normalize(z_i, dim=1)
        z_j = F.normalize(z_j, dim=1)
        
        # Get full similarity matrix
        representations = torch.cat([z_i, z_j], dim=0)
        similarity_matrix = torch.matmul(representations, representations.T)
        
        # Create mask for positives and negatives
        mask = torch.eye(2 * batch_size, dtype=bool, device=z_i.device)
 # Get positive pairs
        positives_mask = torch.zeros_like(similarity_matrix, dtype=bool)
        positives_mask[torch.arange(batch_size), torch.arange(batch_size, 2*batch_size)] = True
        positives_mask[torch.arange(batch_size, 2*batch_size), torch.arange(batch_size)] = True
        positives = similarity_matrix[positives_mask].view(2 * batch_size, 1)
        
        # Get positive pairs
        positives = similarity_matrix[positives_mask].view(2 * batch_size, 1)
        # positives = torch.clamp(positives, max=0.85)
        
        # Get all negative pairs
        negatives_mask = ~(mask | positives_mask)
        negatives = similarity_matrix[negatives_mask].view(2 * batch_size, -1)
        
        k = min(14, negatives.size(1))
        hard_negatives, _ = negatives.topk(k, dim=1)
        
        # Modified negative handling
        margin = 0.21
        hard_negatives = torch.clamp(hard_negatives - margin, min=-0.7)  # Apply margin but keep positive
        
        # Modified distance penalty - always positive
        # Dynamic penalty based on batch statistics
        too_close = torch.gt(hard_negatives, 0.1).float()
        mean_sim = similarity_matrix[~mask].mean()
        spread = similarity_matrix[~mask].std()
        penalty_weight = 0.15
        distance_penalty = (hard_negatives * too_close).mean() * penalty_weight
        
        logits = torch.cat([
            positives / self.temperature,
            hard_negatives / self.temperature
        ], dim=1)
        
        labels = torch.zeros(2 * batch_size, dtype=torch.long, device=z_i.device)
        
      # Modified loss combination
        alignment_loss = F.cross_entropy(logits, labels)
        uniformity_loss = torch.pdist(representations).pow(2).mul(-2).exp().mean().log()
        uniformity_term = -torch.log(spread + 1e-6)  # Penalize low spread

        
        return alignment_loss + 0.185 * uniformity_loss + distance_penalty + 0.11 * uniformity_term

# ABOVE LINE WITHT TOTAL LOSS SEEMS TO HAVE A MASSIFE EFFECT ON TRAIN LOSS

class PaperPairDataset(Dataset):
    def __init__(self, excel_file):
        self.data = pd.read_excel(excel_file)
        self.clean_data()
    # Add this
        self.normalize_features()
        
    def normalize_features(self):
        # Improved normalization strategy
        numeric_columns = ['shared_reference_count', 'shared_citation_count', 'shared_author_count']
        for col in numeric_columns:
            q1 = np.percentile(self.data[col], 25)
            q3 = np.percentile(self.data[col], 75)
            iqr = q3 - q1
            self.data[col] = (self.data[col] - q1) / (iqr + 1e-8)

    def clean_data(self):
        valid_rows = []
        for idx, row in self.data.iterrows():
            try:
                paper1_scibert = self.convert_scibert_string(str(row['paper1_SciBert']))
                paper2_scibert = self.convert_scibert_string(str(row['paper2_SciBert']))
                
                if len(paper1_scibert) == 768 and len(paper2_scibert) == 768:
                    valid_rows.append(idx)
            except:
                continue
                
        self.data = self.data.loc[valid_rows].reset_index(drop=True)
        logging.info(f"Cleaned dataset: kept {len(valid_rows)} rows")
    
    def convert_scibert_string(self, scibert_str):
        values = scibert_str.strip('[]').split(',')
        return [float(x.strip()) for x in values if x.strip()]
    
    def __len__(self):
        return len(self.data)
    
    
    def __getitem__(self, idx):
        row = self.data.iloc[idx]
    
        # Convert SciBERT embeddings
        paper1_scibert = self.convert_scibert_string(str(row['paper1_SciBert']))
        paper2_scibert = self.convert_scibert_string(str(row['paper2_SciBert']))
    
        # Create feature tensors with numeric features only
        shared_features = torch.tensor([
            float(row['shared_reference_count']),
            float(row['reference_cosine']),
            float(row['shared_citation_count']),
            float(row['citation_cosine']),
            float(row['shared_author_count']),
            float(row['abstract_cosine'])
        ], dtype=torch.float32)
    
        return (
            torch.tensor(paper1_scibert, dtype=torch.float32),
            torch.tensor(paper2_scibert, dtype=torch.float32),
            shared_features
        )

class SiameseNetwork(nn.Module):
    def __init__(self, scibert_size=768):
        super().__init__()
        
        # Paper encoder remains unchanged as it works well
        self.paper_encoder = nn.Sequential(
            nn.Linear(scibert_size, 512),
            nn.LayerNorm(512),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(512, 384),
            nn.LayerNorm(384),
            nn.GELU(),
            nn.Dropout(0.2)
        )
        
        # Metadata encoder remains unchanged
        self.metadata_encoder = nn.Sequential(
            nn.Linear(6, 48),
            nn.LayerNorm(48),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(48, 64),
            nn.LayerNorm(64)
        )
        
        # Modified comparison network to use LayerNorm instead of BatchNorm
        self.comparison_network = nn.Sequential(
            nn.Linear(384 * 2 + 64, 384),
            nn.LayerNorm(384),  # Changed from BatchNorm
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(384, 256),
            nn.LayerNorm(256),  # Changed from BatchNorm
            nn.GELU()
        )

        # Projection heads remain mostly unchanged but with slightly higher dropout
        self.projection1 = nn.Sequential(
            nn.Linear(256, 128),
            nn.LayerNorm(128),
            nn.GELU(),
            nn.Dropout(0.25),  # Slightly increased dropout
            nn.Linear(128, 64),
            nn.LayerNorm(64)
        )
        
        self.projection2 = nn.Sequential(
            nn.Linear(256, 128),
            nn.LayerNorm(128),
            nn.GELU(),
            nn.Dropout(0.25),  # Slightly increased dropout
            nn.Linear(128, 64),
            nn.LayerNorm(64)
        )
    
    def forward(self, paper1, paper2, metadata):
        p1_encoded = self.paper_encoder(paper1)
        p2_encoded = self.paper_encoder(paper2)
        meta_encoded = self.metadata_encoder(metadata) * 0.15
        
        combined = torch.cat([p1_encoded, p2_encoded, meta_encoded], dim=1)
        compared = self.comparison_network(combined)
        
        # Separate projections instead of chunking
        output1 = self.projection1(compared)
        output2 = self.projection2(compared)
    

        
        return F.normalize(output1, dim=1, eps=1e-8), F.normalize(output2, dim=1, eps=1e-8)




def setup_folders():
    """Create organized folder structure for experiment"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    base_dir = f"experiments/run_{timestamp}"
    
    # Create folders
    folders = {
        'models': os.path.join(base_dir, 'models'),
        'logs': os.path.join(base_dir, 'logs'),
    }
    
    for folder in folders.values():
        os.makedirs(folder, exist_ok=True)


def setup_folders():
    """Create organized folder structure for experiment"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    base_dir = f"experiments/run_{timestamp}"
    
    # Create folders
    folders = {
        'models': os.path.join(base_dir, 'models'),
        'logs': os.path.join(base_dir, 'logs'),
    }
    
    for folder in folders.values():
        os.makedirs(folder, exist_ok=True)
        
    return folders, timestamp

def train_fold(model, train_loader, val_loader, fold, folders, num_epochs=30, learning_rate=0.0001):
    start_time = time.time()  # Add this at the beginning of the function
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    criterion = NTXentLoss(temperature=0.5).to(device)
    
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=learning_rate * 0.91,
        weight_decay=0.028,
        betas=(0.9, 0.999)
    )
    
    # Correct scheduler setup
    warmup_epochs = 5
    warmup_scheduler = torch.optim.lr_scheduler.LinearLR(
        optimizer, 
        start_factor=0.035,
        total_iters=warmup_epochs * len(train_loader)
    )
    
    main_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=num_epochs - warmup_epochs,
        eta_min=1e-6
    )

    best_val_loss = float('inf')
    patience = 15
    patience_counter = 0
    best_model_path = os.path.join(folders['models'], f'best_model_fold_{fold}.pth')
    
    train_losses = []
    val_losses = []
    
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        num_batches = 0
        
        progress_bar = tqdm(train_loader, desc=f'Fold {fold}, Epoch {epoch+1}/{num_epochs}')
        for paper1, paper2, shared_features in progress_bar:
            paper1, paper2, shared_features = paper1.to(device), paper2.to(device), shared_features.to(device)
            
            embedding1, embedding2 = model(paper1, paper2, shared_features)
            loss = criterion(embedding1, embedding2)
            
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            running_loss += loss.item()
            num_batches += 1
            progress_bar.set_postfix({'train_loss': loss.item()})
        
        # Update appropriate scheduler
        if epoch < warmup_epochs:
            warmup_scheduler.step()
        else:
            main_scheduler.step()
            
        avg_train_loss = running_loss / num_batches
        train_losses.append(avg_train_loss)
        
        # Validation phase
        model.eval()
        total_val_loss = 0
        with torch.no_grad():
            for paper1, paper2, shared_features in val_loader:
                paper1 = paper1.to(device)
                paper2 = paper2.to(device)
                shared_features = shared_features.to(device)
                
                embedding1, embedding2 = model(paper1, paper2, shared_features)
                val_loss = criterion(embedding1, embedding2)
                total_val_loss += val_loss.item()
        
        avg_val_loss = total_val_loss / len(val_loader)
        val_losses.append(avg_val_loss)
        
        logging.info(f'Fold {fold}, Epoch {epoch+1}: Train Loss = {avg_train_loss:.4f}, Val Loss = {avg_val_loss:.4f}')
        
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), best_model_path)
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                logging.info(f'Early stopping triggered after {epoch + 1} epochs')
                break
    
    fold_time = time.time() - start_time
    logging.info(f'Fold {fold} Training Time: {fold_time/60:.2f} minutes')
    
    return train_losses, val_losses, best_val_loss, fold_time




def get_paper_similarity(
    model: nn.Module,
    calibrator: SimilarityCalibrator,
    paper1_features: torch.Tensor,
    paper2_features: torch.Tensor,
    shared_features: Optional[torch.Tensor] = None
) -> float:
    """
    Get calibrated similarity score between two papers.
    """
    model.eval()
    with torch.no_grad():
        if shared_features is None:
            shared_features = torch.zeros(6)
            
        emb1, emb2 = model(paper1_features, paper2_features, shared_features)
        raw_similarity = F.cosine_similarity(emb1, emb2, dim=1)
        calibrated_similarity = calibrator.calibrate(raw_similarity)
        
        return calibrated_similarity.item()


def evaluate_fold(model, val_loader, calibrator=None):
    device = next(model.parameters()).device
    model.eval()
    raw_similarities = []
    calibrated_similarities = []
    
    with torch.no_grad():
        # First pass: collect all raw similarities
        for paper1, paper2, shared_features in val_loader:
            paper1 = paper1.to(device)
            paper2 = paper2.to(device)
            shared_features = shared_features.to(device)
            
            embedding1, embedding2 = model(paper1, paper2, shared_features)
            similarity = F.cosine_similarity(embedding1, embedding2, dim=1).cpu()
            raw_similarities.extend(similarity.numpy())
        
        # Convert to tensor for calibration
        raw_similarities = np.array(raw_similarities)
        raw_tensor = torch.tensor(raw_similarities)
        
        # Fit calibrator on all raw similarities first
        if calibrator is not None:
            calibrator.fit(raw_tensor)
            calibrated = calibrator.calibrate(raw_tensor)
            calibrated_similarities = calibrated.numpy()
            
            return raw_similarities, calibrated_similarities
        
    return raw_similarities

def main():
    # Configuration dictionary remains unchanged
    config = {
        'input_file': "paper_tidiedUp_balanced_four_FINAL.xlsx",
        'n_splits': 5,
        'batch_size': 64,
        'learning_rate': 0.0002,
        'num_epochs': 20,
        'weight_decay': 0.01,
        'scheduler_patience': 6,
        'scheduler_factor': 0.5,
        'contrastive_temperature': 0.2,
        'similarity_threshold': 0.5,
        'network_params': {
            'dropout_rates': [0.2, 0.1],
            'hidden_dims': [512, 256, 128],
            'initial_importance': {
                'semantic': 0.7,
                'citation': 0.3
            }
        }
    }
    
    # Setup folders and logging
    folders, timestamp = setup_folders()
    os.makedirs(folders['logs'], exist_ok=True)
    log_file = os.path.join(folders['logs'], f'training_{timestamp}.log')
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file, mode='w'),
            logging.StreamHandler()
        ]
    )
    
    try:
        start_time = time.time()
        
        # Log all configuration parameters
        logging.info("=== Training Configuration ===")
        logging.info(f"Timestamp: {timestamp}")
        logging.info("\nData Parameters:")
        logging.info(f"- Input file: {config['input_file']}")
        logging.info(f"- Number of folds: {config['n_splits']}")
        logging.info(f"- Batch size: {config['batch_size']}")
        
        logging.info("\nTraining Parameters:")
        logging.info(f"- Learning rate: {config['learning_rate']}")
        logging.info(f"- Number of epochs: {config['num_epochs']}")
        logging.info(f"- Weight decay: {config['weight_decay']}")
        
        logging.info("\nScheduler Parameters:")
        logging.info(f"- Patience: {config['scheduler_patience']}")
        logging.info(f"- Reduction factor: {config['scheduler_factor']}")
        
        logging.info("\nContrastive Loss Parameters:")
        logging.info(f"- Temperature: {config['contrastive_temperature']}")
        logging.info(f"- Similarity threshold: {config['similarity_threshold']}")
        
        logging.info("\nNetwork Architecture:")
        logging.info(f"- Hidden dimensions: {config['network_params']['hidden_dims']}")
        logging.info(f"- Dropout rates: {config['network_params']['dropout_rates']}")
        logging.info(f"- Initial importance weights:")
        logging.info(f"  - Semantic: {config['network_params']['initial_importance']['semantic']}")
        logging.info(f"  - Citation: {config['network_params']['initial_importance']['citation']}")
        
        logging.info("\n=== Starting Training ===")
        
        dataset = PaperPairDataset(config['input_file'])
        logging.info(f"Successfully loaded {len(dataset)} paper pairs")
        
        # Initialize K-Fold
        kfold = KFold(n_splits=config['n_splits'], shuffle=True, random_state=42)
        
        # Get scibert size
        paper1_features, _, _ = dataset[0]
        scibert_size = paper1_features.shape[0]
        logging.info(f"SciBERT embedding size: {scibert_size}")
        
        # Initialize calibrator
        calibrator = SimilarityCalibrator(
    initial_temperature=2.0,    # Controls initial scaling
    target_mean=0.5,           # Center point stays at 0.5
    target_std=0.25,           # Target spread
) 
        # Store metrics
        fold_metrics = {
            'train_losses': [],
            'val_losses': [],
            'best_val_losses': [],
            'raw_similarities': [],
            'calibrated_similarities': [],
            'fold_times': [],
            'final_importance_weights': []
        }
        
        # Perform k-fold cross validation
        for fold, (train_idx, val_idx) in enumerate(kfold.split(dataset), 1):
            logging.info(f"\nStarting Fold {fold}/{config['n_splits']}")
            logging.info(f"Train size: {len(train_idx)}, Validation size: {len(val_idx)}")
            
            # Create data loaders
            train_sampler = torch.utils.data.SubsetRandomSampler(train_idx)
            val_sampler = torch.utils.data.SubsetRandomSampler(val_idx)
            
            train_loader = DataLoader(dataset, batch_size=config['batch_size'], sampler=train_sampler)
            val_loader = DataLoader(dataset, batch_size=config['batch_size'], sampler=val_sampler)
            
            # Initialize model
            model = SiameseNetwork(scibert_size)
            
            # Train fold
            train_losses, val_losses, best_val_loss, fold_time = train_fold(
               model, train_loader, val_loader, fold, folders,
               num_epochs=config['num_epochs'],
               learning_rate=config['learning_rate']
            )
            
            # Log final importance weights if they exist
            try:
                with torch.no_grad():
                    if hasattr(model, 'semantic_importance') and hasattr(model, 'citation_importance'):
                        weights = F.softmax(torch.stack([
                            model.semantic_importance,
                            model.citation_importance
                        ]), dim=0)
                        fold_metrics['final_importance_weights'].append({
                            'semantic': weights[0].item(),
                            'citation': weights[1].item()
                        })
                        logging.info("\nFinal importance weights:")
                        logging.info(f"- Semantic: {weights[0].item():.4f}")
                        logging.info(f"- Citation: {weights[1].item():.4f}")
                    else:
                        logging.info("\nModel does not have importance weights")
                        fold_metrics['final_importance_weights'].append({
                            'semantic': None,
                            'citation': None
                        })
            except Exception as e:
                logging.warning(f"Could not log importance weights: {str(e)}")
                fold_metrics['final_importance_weights'].append({
                    'semantic': None,
                    'citation': None
                })
            
            # Load best model and evaluate with calibration
            model.load_state_dict(torch.load(os.path.join(folders['models'], f'best_model_fold_{fold}.pth')))
            raw_similarities, calibrated_similarities = evaluate_fold(model, val_loader, calibrator)
            
            # Store and log metrics
            fold_metrics['train_losses'].append(train_losses)
            fold_metrics['val_losses'].append(val_losses)
            fold_metrics['best_val_losses'].append(best_val_loss)
            fold_metrics['raw_similarities'].append(raw_similarities)
            fold_metrics['calibrated_similarities'].append(calibrated_similarities)
            fold_metrics['fold_times'].append(fold_time)
            
            logging.info(f"\nFold {fold} Results:")
            logging.info(f"Best Validation Loss: {best_val_loss:.4f}")
            logging.info(f"\nSimilarity Statistics:")
            logging.info(f"Raw - Mean: {np.mean(raw_similarities):.4f}, Std: {np.std(raw_similarities):.4f}")
            logging.info(f"Calibrated - Mean: {np.mean(calibrated_similarities):.4f}, Std: {np.std(calibrated_similarities):.4f}")
            logging.info(f"Training Time: {fold_time/60:.2f} minutes")
        
        # Calculate and log final results
        total_time = time.time() - start_time
        
        logging.info("\n=== Final Results ===")
        logging.info(f"Mean Best Validation Loss: {np.mean(fold_metrics['best_val_losses']):.4f}")
        
        # Log overall similarity statistics
        all_raw_similarities = np.concatenate(fold_metrics['raw_similarities'])
        all_calibrated_similarities = np.concatenate(fold_metrics['calibrated_similarities'])
        
        logging.info("\nOverall Similarity Statistics:")
        logging.info(f"Raw - Mean: {np.mean(all_raw_similarities):.4f}, Std: {np.std(all_raw_similarities):.4f}")
        logging.info(f"Calibrated - Mean: {np.mean(all_calibrated_similarities):.4f}, Std: {np.std(all_calibrated_similarities):.4f}")
        
        logging.info("\nTiming Information:")
        logging.info(f"Total Training Time: {total_time/60:.2f} minutes")
        logging.info("Time per fold:")
        for fold, fold_time in enumerate(fold_metrics['fold_times'], 1):
            logging.info(f"Fold {fold}: {fold_time/60:.2f} minutes")
        
        # Save metrics including both raw and calibrated similarities
        metrics_path = os.path.join(folders['models'], 'fold_metrics.pth')
        torch.save({
            'config': config,
            'metrics': fold_metrics,
            'total_time': total_time,
            'calibrator': calibrator  # Save calibrator state for later use
        }, metrics_path)
        
    except Exception as e:
        logging.error(f"Error during execution: {e}", exc_info=True)
        raise

if __name__ == "__main__":
    main()