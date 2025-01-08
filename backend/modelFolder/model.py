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
# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    # handlers=[
    #     logging.FileHandler(f'training_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'),
    #     logging.StreamHandler()
    # ]
)


class NTXentLoss(nn.Module):
    def __init__(self, temperature=0.05):  # Even lower temperature
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
        
        # Create labels
        labels = torch.arange(batch_size, device=z_i.device)
        labels = torch.cat([labels + batch_size, labels])
        
        # Create mask
        mask = torch.eye(2 * batch_size, dtype=bool, device=z_i.device)
        
        # Get positive pairs
        positives = torch.diag(similarity_matrix, batch_size)
        positives = torch.cat([positives, torch.diag(similarity_matrix, -batch_size)])
        
        # Get hard negatives (closest negative pairs)
        similarity_matrix.fill_diagonal_(float('-inf'))  # Exclude self-pairs
        hard_negatives = similarity_matrix.max(dim=1)[0]
        
        # Compute loss with stronger penalties for hard negatives
        logits = torch.cat([
            positives.view(-1, 1) / self.temperature,
            hard_negatives.view(-1, 1) / (self.temperature * 0.5)  # Lower temp for negatives
        ], dim=1)
        
        # Add uniformity regularization
        uniformity = torch.pdist(representations).pow(2).mul(-2).exp().mean().log()
        
        return F.cross_entropy(logits, torch.zeros(2 * batch_size, device=z_i.device, dtype=torch.long)) + 0.2 * uniformity



class PaperPairDataset(Dataset):
    def __init__(self, excel_file):
        self.data = pd.read_excel(excel_file)
        self.clean_data()
    # Add this
        self.normalize_features()
        
    def normalize_features(self):
        numeric_columns = ['shared_reference_count', 'shared_citation_count', 'shared_author_count']
        for col in numeric_columns:
            mean = self.data[col].mean()
            std = self.data[col].std()
            self.data[col] = (self.data[col] - mean) / (std + 1e-8)

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
        
        # Individual paper encoder - maintains more of SciBERT's dimensionality
        self.paper_encoder = nn.Sequential(
            nn.Linear(scibert_size, 768),  # Maintain initial dimensionality
            nn.LayerNorm(768),
            nn.GELU(),
            nn.Dropout(0.15),
            nn.Linear(768, 512),  # Gradual reduction
            nn.LayerNorm(512),
            nn.GELU(),
            nn.Dropout(0.15)
        )
        
        # Metadata features require less complexity
        self.metadata_encoder = nn.Sequential(
            nn.Linear(6, 32),
            nn.LayerNorm(32),
            nn.GELU(),
            nn.Linear(32, 64),
            nn.LayerNorm(64)
        )
        
        # Comparison module - processes concatenated features
        self.comparison_network = nn.Sequential(
            nn.Linear(512 * 2 + 64, 512),  # Concatenated papers + metadata
            nn.LayerNorm(512),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(512, 256),
            nn.LayerNorm(256),
            nn.GELU(),
            nn.Linear(256, 128)
        )
        
    def forward(self, paper1, paper2, metadata):
        # Process each paper independently first
        p1_encoded = self.paper_encoder(paper1)
        p2_encoded = self.paper_encoder(paper2)
        
        # Process metadata
        meta_encoded = self.metadata_encoder(metadata)
        
        # Combine all features for comparison
        combined = torch.cat([p1_encoded, p2_encoded, meta_encoded], dim=1)
        compared = self.comparison_network(combined)
        
        # Split into two vectors for contrastive learning
        output1, output2 = torch.chunk(compared, 2, dim=1)
        
        return F.normalize(output1, dim=1), F.normalize(output2, dim=1)


    def encode_paper(self, paper_embedding, citation_features):
        # Apply noise during training
        if self.training:
            paper_embedding = paper_embedding + torch.randn_like(paper_embedding) * 0.1
            
        # Process paper embedding
        paper_features = self.paper_encoder(paper_embedding)
        
        # Process citation features
        citation_encoded = self.citation_encoder(citation_features)
        
        # Combine features
        combined = torch.cat([paper_features, citation_encoded], dim=1)
        
        # Project to final space
        return self.projector(combined)


def contrastive_loss(embedding1, embedding2, temperature=0.4):  # Note temperature change
    batch_size = embedding1.size(0)
    
    # Compute all pairwise distances
    sim_matrix = torch.matmul(embedding1, embedding2.T) / temperature
    
    # Create labels matrix
    labels = torch.eye(batch_size).to(embedding1.device)
    
    # Compute positive and negative samples
    pos_samples = torch.diag(sim_matrix)
    neg_samples = sim_matrix.view(-1)[~torch.eye(batch_size).bool().view(-1)].view(batch_size, -1)
    
    # Compute InfoNCE loss with stronger regularization
    logits = torch.cat([pos_samples.unsqueeze(1), neg_samples], dim=1)
    labels = torch.zeros(batch_size).long().to(embedding1.device)
    
    # Add regularization terms
    diversity_loss = torch.mean(torch.abs(neg_samples))
    uniformity_loss = torch.pdist(embedding1).pow(2).mul(-2).exp().mean().log()
    
    # Combine losses with stronger weights
    loss = F.cross_entropy(logits, labels) + 0.1 * diversity_loss + 0.2 * uniformity_loss
    
    return loss



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
        lr=learning_rate,
        weight_decay=0.02,
        betas=(0.9, 0.999)
    )
    
    # Correct scheduler setup
    warmup_epochs = 3  # Or use config['warmup_epochs']
    warmup_scheduler = torch.optim.lr_scheduler.LinearLR(
        optimizer, 
        start_factor=0.1,
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

def evaluate_fold(model, val_loader):
    model.eval()
    similarities = []
    
    with torch.no_grad():
        for paper1, paper2, shared_features in val_loader:
            embedding1, embedding2 = model(paper1, paper2, shared_features)
            # Just raw cosine similarity
            similarity = F.cosine_similarity(embedding1, embedding2, dim=1)
            similarities.extend(similarity.numpy())
    
    return np.array(similarities)

def main():
          # ONLY USED FOR PRINTING NOT ACTUAL
    config = {
    'input_file': "paper_tidiedUp_four_FINAL.xlsx",
    'n_splits': 5,
    'batch_size': 64,
    'learning_rate': 0.0002,  # Changed from 0.0008
    'num_epochs': 20,
    'weight_decay': 0.01,    # Changed from 0.02
    'scheduler_patience': 6,  # Changed from 4
    'scheduler_factor': 0.5,  # Changed from 0.6
    'contrastive_temperature': 0.2,  # Changed from 1.8
    'similarity_threshold': 0.5,
    'network_params': {
        'dropout_rates': [0.2, 0.1],  # Changed from [0.6, 0.5, 0.4]
        'hidden_dims': [512, 256, 128],
        'initial_importance': {
            'semantic': 0.7,
            'citation': 0.3
        }
    }
}
    
    # Setup folders and logging
    folders, timestamp = setup_folders()
    
    # Ensure logs folder exists
    os.makedirs(folders['logs'], exist_ok=True)
    
    # Create log filename with timestamp
    # log_file = os.path.join(folders['logs'], f'training_{timestamp}.log')
    
    # # Setup logging with absolute path
    # logging.basicConfig(
    #     level=logging.INFO,
    #     format='%(asctime)s - %(levelname)s - %(message)s',
    #     handlers=[
    #         logging.FileHandler(log_file, mode='w'),
    #         logging.StreamHandler()
    #     ]
    # )
    
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
        
        # Store metrics
        fold_metrics = {
            'train_losses': [],
            'val_losses': [],
            'best_val_losses': [],
            'similarities': [],
            'fold_times': [],
            'final_importance_weights': []  # Track how importance weights evolved
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
            
       # Log final importance weights
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
            
            # Load best model and evaluate
            model.load_state_dict(torch.load(os.path.join(folders['models'], f'best_model_fold_{fold}.pth')))
            similarities = evaluate_fold(model, val_loader)

            
            # Store and log metrics
            fold_metrics['train_losses'].append(train_losses)
            fold_metrics['val_losses'].append(val_losses)
            fold_metrics['best_val_losses'].append(best_val_loss)
            fold_metrics['similarities'].append(similarities)
            fold_metrics['fold_times'].append(fold_time)
            
            logging.info(f"\nFold {fold} Results:")
            logging.info(f"Best Validation Loss: {best_val_loss:.4f}")
            logging.info(f"Mean Similarity: {np.mean(similarities):.4f}")
            logging.info(f"Std Similarity: {np.std(similarities):.4f}")
            logging.info(f"Training Time: {fold_time/60:.2f} minutes")
        
        # Calculate and log final results
        total_time = time.time() - start_time
        
        logging.info("\n=== Final Results ===")
        logging.info(f"Mean Best Validation Loss: {np.mean(fold_metrics['best_val_losses']):.4f}")
        
        all_similarities = np.concatenate(fold_metrics['similarities'])
        logging.info(f"Overall Mean Similarity: {np.mean(all_similarities):.4f}")
        logging.info(f"Overall Std Similarity: {np.std(all_similarities):.4f}")
        
        # Log average final importance weights
        # avg_semantic = np.mean([w['semantic'] for w in fold_metrics['final_importance_weights']])
        # avg_citation = np.mean([w['citation'] for w in fold_metrics['final_importance_weights']])
        # logging.info("\nAverage Final Importance Weights:")
        # logging.info(f"- Semantic: {avg_semantic:.4f}")
        # logging.info(f"- Citation: {avg_citation:.4f}")
        
        logging.info("\nTiming Information:")
        logging.info(f"Total Training Time: {total_time/60:.2f} minutes")
        logging.info("Time per fold:")
        for fold, fold_time in enumerate(fold_metrics['fold_times'], 1):
            logging.info(f"Fold {fold}: {fold_time/60:.2f} minutes")
        
        # Save metrics
        metrics_path = os.path.join(folders['models'], 'fold_metrics.pth')
        torch.save({
            'config': config,
            'metrics': fold_metrics,
            'total_time': total_time
        }, metrics_path)
        
    except Exception as e:
        logging.error(f"Error during execution: {e}", exc_info=True)
        raise

if __name__ == "__main__":
    main()