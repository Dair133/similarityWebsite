import json
import random
import time
import math
import os
import logging
from datetime import datetime
from typing import Optional, Dict, Any

# Data processing imports
import numpy as np
import pandas as pd

# PyTorch imports
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split, SubsetRandomSampler
from torch.optim.lr_scheduler import ReduceLROnPlateau, OneCycleLR

# SKLearn imports
from sklearn.model_selection import GroupShuffleSplit, KFold

# Progress tracking
from tqdm import tqdm

# Optuna for hyperparameter optimization
import optuna
from optuna.pruners import MedianPruner

# Argument parsing
import argparse


from modelFolder.modelRunners.modelDependencies.memoryBankNew import MemoryBank

def train_val_group_split(df, group_column='paper1_Main_Topic', train_size=0.8):  # Changed default!
    splitter = GroupShuffleSplit(train_size=train_size, random_state=42, n_splits=1)
    train_inds, val_inds = next(splitter.split(df, groups=df[group_column]))
    train_df = df.iloc[train_inds]
    val_df = df.iloc[val_inds]
    return train_df, val_df

def objective(trial, train_dataset, val_dataset):
    """Objective function ensuring current config values are within search space"""
    
    config = {
        'paper_hidden_size': trial.suggest_categorical("paper_hidden_size", [64, 128, 256, 512]),
        'metadata_hidden_size': trial.suggest_categorical("metadata_hidden_size", [16, 32, 64, 96]),
        'paper_dropout': trial.suggest_float("paper_dropout", 0.1, 0.5),
        'metadata_dropout': trial.suggest_float("metadata_dropout", 0.1, 0.5),
'weight_decay': trial.suggest_float("weight_decay", 5e-6, 1e-4, log=True),
        'batch_size': trial.suggest_categorical("batch_size", [32, 64, 128, 256]),
        'learning_rate': trial.suggest_float("learning_rate", 1e-5, 5e-3, log=True),
        'temperature': trial.suggest_float("temperature", 0.08, 0.3),
        'num_negatives': trial.suggest_categorical("num_negatives", [64, 128, 256, 512]),
        'hard_fraction': trial.suggest_float("hard_fraction", 0.3, 0.7),
        'augment_prob': trial.suggest_float("augment_prob", 0.1, 0.35),
        'augment_noise_scale': trial.suggest_float("augment_noise_scale", 0.01, 0.08, log=True),
        'mask_prob': trial.suggest_float("mask_prob", 0.1, 0.35),
        'shuffle_percentage': trial.suggest_float("shuffle_percentage", 0.05, 0.2),
        'memory_bank_size': trial.suggest_categorical("memory_bank_size", [1024, 2048, 4096, 8192])
    }

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Apply augmentation to training dataset
    train_dataset.augment = True
    train_dataset.augment_prob = config['augment_prob']
    train_dataset.augment_noise_scale = config['augment_noise_scale']
    train_dataset.mask_prob = config['mask_prob']
    train_dataset.shuffle_percentage = config['shuffle_percentage']


    train_loader = DataLoader(
        train_dataset, 
        batch_size=config['batch_size'],
        shuffle=True,
        num_workers=4,
        pin_memory=True,
        prefetch_factor=2,
        persistent_workers=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config['batch_size'],
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )

    model = SiameseNetwork(
        paper_dropout=config['paper_dropout'],
        metadata_dropout=config['metadata_dropout'],
        paper_hidden_size=config['paper_hidden_size'],
        metadata_hidden_size=config['metadata_hidden_size']
    ).to(device)

    try:
        best_val_loss = train_fold(
            model, 
            train_loader, 
            val_loader, 
            fold=0,
            num_epochs=15, 
            learning_rate=config['learning_rate'],
            weight_decay=config['weight_decay'],  
            folders=None, 
            device=device, 
            trial=trial,
            temperature=config['temperature'],
            num_negatives=config['num_negatives'],
            hard_fraction=config['hard_fraction'],
            memory_bank_size=config['memory_bank_size'] 
        )

        if hasattr(trial, 'study'):
            state_dict = {k: v.cpu().numpy().tolist() for k, v in model.state_dict().items()}
            trial.set_user_attr('best_model_state', state_dict)
            trial.set_user_attr('config', config)
    except optuna.TrialPruned:
        raise

    return best_val_loss



class NTXentLoss(nn.Module):
    def __init__(self, temperature=0.1, num_negatives=512, hard_fraction=0.3, memory_bank_size=4096, device='cpu'):
        super().__init__()
        self.temperature = temperature
        self.num_negatives = num_negatives
        self.hard_fraction = hard_fraction
        self.memory_bank = MemoryBank(size=memory_bank_size, feature_dim=64, device=device)
        self.device = device

    def forward(self, z_i, z_j):
        batch_size = z_i.size(0)
        z_i = F.normalize(z_i, dim=1)
        z_j = F.normalize(z_j, dim=1)

        memory = self.memory_bank.get_memory().to(self.device)
        memory = F.normalize(memory, dim=1)

        # Compute similarities
        sim_i = torch.matmul(z_i, memory.T) / self.temperature
        sim_j = torch.matmul(z_j, memory.T) / self.temperature
        pos_sim = torch.sum(z_i * z_j, dim=1, keepdim=True) / self.temperature

        # Calculate number of negatives
        k_total = min(self.num_negatives, memory.size(0))  # Cannot exceed memory size
        k_hard = min(int(k_total * self.hard_fraction), memory.size(0))
        k_rand = k_total - k_hard

        # Get hard negatives
        hard_negatives_i, _ = torch.topk(sim_i, k=k_hard, dim=1) if k_hard > 0 else (torch.tensor([]).to(self.device), None)
        hard_negatives_j, _ = torch.topk(sim_j, k=k_hard, dim=1) if k_hard > 0 else (torch.tensor([]).to(self.device), None)

        # Get random negatives
        if k_rand > 0 and memory.size(0) > k_rand:
            rand_indices = torch.randperm(memory.size(0), device=self.device)[:k_rand]
            random_negatives_i = sim_i[:, rand_indices]
            random_negatives_j = sim_j[:, rand_indices]
        else:
            random_negatives_i = torch.tensor([]).to(self.device)
            random_negatives_j = torch.tensor([]).to(self.device)

        # Concatenate all logits
        logits_i = torch.cat([hard_negatives_i, random_negatives_i, pos_sim], dim=1)
        logits_j = torch.cat([hard_negatives_j, random_negatives_j, pos_sim], dim=1)

        # The target is now guaranteed to be the last index and within bounds
        labels = torch.full((batch_size,), logits_i.size(1) - 1, device=self.device, dtype=torch.long)

        # Compute loss
        loss_i = F.cross_entropy(logits_i, labels)
        loss_j = F.cross_entropy(logits_j, labels)

        # Update memory bank
        with torch.no_grad():
            update_size = min(batch_size, self.memory_bank.size // 12)  # Ensure we don't exceed memory size
            indices = torch.randperm(batch_size, device=self.device)[:update_size]
            combined = torch.cat([z_i[indices], z_j[indices]], dim=0).detach()
            self.memory_bank.update(combined)

        return (loss_i + loss_j) / 2
class PaperPairDataset(Dataset):
    def __init__(self, dataframe: pd.DataFrame, augment: bool = False, augment_prob: float = 0.1,
                 augment_noise_scale: float = 0.01, mask_prob: float = 0.1, shuffle_percentage: float = 0.05):
        self.data = dataframe
        self.data = self.data.dropna()
        self.clean_data()
        self.normalize_features()
        self.augment = augment
        self.augment_prob = augment_prob
        self.augment_noise_scale = augment_noise_scale
        self.mask_prob = mask_prob  
        self.shuffle_percentage = shuffle_percentage  
        if self.augment:
            print(f"Data augmentation enabled with probability {self.augment_prob}, noise scale {self.augment_noise_scale}, mask prob {self.mask_prob}, shuffle percentage {self.shuffle_percentage}")

    def clean_data(self):
        """Cleans and validates SciBERT embeddings, dropping invalid rows."""
        valid_rows = []
        for idx, row in self.data.iterrows():
            try:
                paper1 = self._parse_scibert_string(row['paper1_SciBert'])
                paper2 = self._parse_scibert_string(row['paper2_SciBert'])
                if len(paper1) == 768 and len(paper2) == 768:
                    valid_rows.append(idx)
            except (ValueError, TypeError) as e:
                print(f"Skipping row {idx} due to parsing error: {e}")  
                continue
        print(f"Original data length before the prune: {len(self.data)}")
        self.data = self.data.loc[valid_rows].reset_index(drop=True)
        print(f"Cleaned dataset: kept {len(self.data)} rows")

    def normalize_features(self):
        """Normalizes specified numerical features."""

        numeric_columns = ['shared_references', 'shared_citations', 'shared_authors']
        for col in numeric_columns:
            if col in self.data.columns:
                mean = self.data[col].mean()
                std = self.data[col].std()
                if std > 1e-8:
                    self.data[col] = (self.data[col] - mean) / std
                else:
                    print(f"Warning: Standard Deviation for {col} is zero (or very close).  Skipping normalization.")
                    self.data[col] = 0.0

                print(f"Feature: {col}, Mean: {self.data[col].mean():.4f}, Std: {self.data[col].std():.4f}")
            else:
                print(f"Warning: Column {col} not found in data. Skipping.")


    def _parse_scibert_string(self, s: str) -> list[float]:
        """Parses a SciBERT string representation into a list of floats."""
        try:
            # Remove brackets, split by comma, and convert to float
            return [float(x.strip()) for x in s.strip('[]').split(',') if x.strip()]
        except ValueError as e:
            raise ValueError(f"Could not parse SciBERT string: {s}. Error: {e}")

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        row = self.data.iloc[idx]
        try:
            paper1_embedding = torch.tensor(self._parse_scibert_string(str(row['paper1_SciBert'])), dtype=torch.float32)
            paper2_embedding = torch.tensor(self._parse_scibert_string(str(row['paper2_SciBert'])), dtype=torch.float32)

            shared_features = torch.tensor([
                float(row['shared_references']),
                float(row['shared_citations']),
                float(row['shared_authors'])
            ], dtype=torch.float32)

            if self.augment:
                if random.random() < self.augment_prob:
                    paper1_embedding = self.mask_embeddings(paper1_embedding)
                    paper1_embedding = self.shuffle_embeddings(paper1_embedding)
                if random.random() < self.augment_prob:
                    paper2_embedding = self.mask_embeddings(paper2_embedding)
                    paper2_embedding = self.shuffle_embeddings(paper2_embedding)

        except (ValueError, KeyError) as e:
            print(f"Error processing row {idx}: {e}")
            return torch.zeros(768, dtype=torch.float32), torch.zeros(768, dtype=torch.float32), torch.zeros(3,dtype=torch.float32)

        return paper1_embedding, paper2_embedding, shared_features


    def mask_embeddings(self, embedding):
        mask = torch.rand(embedding.size()) < self.mask_prob
        masked_embedding = embedding.clone()  # Important to clone!
        masked_embedding[mask] = 0.0
        return masked_embedding

    def shuffle_embeddings(self, embedding):
        num_to_shuffle = int(len(embedding) * self.shuffle_percentage)
        indices_to_shuffle = random.sample(range(len(embedding)), num_to_shuffle)
        shuffled_embedding = embedding.clone()
        shuffled_values = shuffled_embedding[indices_to_shuffle].clone()
        random.shuffle(shuffled_values)  # In-place shuffle
        shuffled_embedding[indices_to_shuffle] = shuffled_values
        return shuffled_embedding



import torch
from torch import nn
import torch.nn.functional as F

class SiameseNetwork(nn.Module):
    def __init__(self, scibert_size: int = 768, projection_size: int = 64, metadata_size: int = 3,
                 paper_hidden_size: int = 128, metadata_hidden_size: int = 32,
                 paper_dropout: float = 0.3, metadata_dropout: float = 0.3):
        super().__init__()

        self.paper_encoder = nn.Sequential(
            nn.Linear(scibert_size, 512), 
            nn.LayerNorm(512),
            nn.GELU(),
            nn.Dropout(paper_dropout),
            nn.Linear(512, 256), 
            nn.LayerNorm(256)
        )

     
        self.metadata_encoder = nn.Sequential(
            nn.Linear(metadata_size, metadata_hidden_size),
            nn.LayerNorm(metadata_hidden_size),
            nn.GELU(),
            nn.Dropout(metadata_dropout),
            nn.Linear(metadata_hidden_size, 64),
            nn.LayerNorm(64)
        )

   
        self.fusion_layer = nn.Sequential(
            nn.Linear(256 + 64, projection_size),
            nn.GELU(),
            nn.Dropout(0.2)
        )

    def forward(self, paper1: torch.Tensor, paper2: torch.Tensor, metadata: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        p1 = self.paper_encoder(paper1)
        p2 = self.paper_encoder(paper2)
        meta = self.metadata_encoder(metadata)

        combined1 = torch.cat([p1, meta], dim=1)
        combined2 = torch.cat([p2, meta], dim=1)

        fused1 = F.normalize(self.fusion_layer(combined1), dim=1)
        fused2 = F.normalize(self.fusion_layer(combined2), dim=1)

        return fused1, fused2



 



def setup_folders(experiment_name: str = "run") -> tuple[dict[str, str], str]:
    """Create organized folder structure for the experiment."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    base_dir = f"experiments/{experiment_name}_{timestamp}"  # Put everything under /experiments

    folders = {
        'models': os.path.join(base_dir, 'models'),
        'logs': os.path.join(base_dir, 'logs'),  # Log files go here
        'optuna': base_dir,  # Optuna DB in the base directory
    }

    os.makedirs(folders['models'], exist_ok=True)
    os.makedirs(folders['logs'], exist_ok=True)

    for i in range(5):  # Assuming 5 folds for KFold
        fold_dir = os.path.join(folders['models'], f'fold_{i + 1}')
        os.makedirs(fold_dir, exist_ok=True)

    return folders, timestamp

def update_momentum_encoder(query_encoder: nn.Module, key_encoder: nn.Module, m: float = 0.999):
    # Update key encoder parameters as: key = m * key + (1 - m) * query
    for param_q, param_k in zip(query_encoder.parameters(), key_encoder.parameters()):
        param_k.data = param_k.data * m + param_q.data * (1.0 - m)

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




def setup_folders(experiment_name: str = "run") -> tuple[Dict[str, str], str]:
    """Create organized folder structure for experiment with separate fold directories."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    base_dir = os.path.join("experiments", f"{experiment_name}_{timestamp}")

    folders = {
        'models': os.path.join(base_dir, 'models'),
        'logs': os.path.join(base_dir, 'logs'),
    }

    os.makedirs(base_dir, exist_ok=True)
    for folder in folders.values():
        os.makedirs(folder, exist_ok=True)
    for i in range(5):  # Assuming 5 folds
        fold_dir = os.path.join(folders['models'], f'fold_{i+1}')
        os.makedirs(fold_dir, exist_ok=True)

    return folders, timestamp





def train_fold(model, train_loader, val_loader, fold, num_epochs, learning_rate, 
               weight_decay, folders, device, trial=None, temperature=0.2, 
               num_negatives=128, hard_fraction=0.3, memory_bank_size=2048):  # Add 
    criterion = NTXentLoss(
        temperature=temperature,
        num_negatives=num_negatives,
        hard_fraction=hard_fraction,
        memory_bank_size=memory_bank_size,  # Use the passed parameter
        device=device
    )
    
    # Use the passed weight_decay
    optimizer = torch.optim.AdamW(
        model.parameters(), 
        lr=learning_rate, 
        weight_decay=weight_decay  # Use the passed weight_decay
    )
    
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3, verbose=True)

    train_losses = []
    val_losses = []
    best_val_loss = float('inf')
    best_epoch = -1

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        progress_bar = tqdm(train_loader, desc=f"Fold {fold+1}, Epoch {epoch+1}/{num_epochs}")

        for paper1, paper2, shared in progress_bar:
            paper1, paper2, shared = paper1.to(device), paper2.to(device), shared.to(device)
            embedding1, embedding2 = model(paper1, paper2, shared)
            loss = criterion(embedding1, embedding2)
            optimizer.zero_grad()
            loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            running_loss += loss.item()
            progress_bar.set_postfix({'train_loss': loss.item()})

        avg_train_loss = running_loss / len(train_loader)
        train_losses.append(avg_train_loss)
        model.eval()
        total_val_loss = 0.0

        with torch.no_grad():
            for paper1, paper2, shared in val_loader:
                paper1, paper2, shared = paper1.to(device), paper2.to(device), shared.to(device)
                embedding1, embedding2 = model(paper1, paper2, shared)
                loss_val = criterion(embedding1, embedding2)
                total_val_loss += loss_val.item()

        avg_val_loss = total_val_loss / len(val_loader)
        val_losses.append(avg_val_loss)
        log_message = f"Fold {fold+1}, Epoch {epoch+1}: Train Loss = {avg_train_loss:.4f}, Val Loss = {avg_val_loss:.4f}"
        print(log_message)
        logging.info(log_message)


        if trial:  # Only report to Optuna during optimization
          trial.report(avg_val_loss, step=epoch)  # Report val_loss for pruning
          trial.set_user_attr(f'epoch_{epoch}_metrics', {
        'train_loss': float(avg_train_loss),
        'val_loss': float(avg_val_loss)
    })
          if trial.should_prune():
            print(f"Trial pruned at epoch {epoch+1} for fold {fold+1}")
            logging.info(f"Trial pruned at epoch {epoch+1} for fold {fold+1}")
            return float('inf')  # Return inf, not best_val_loss
        scheduler.step(avg_val_loss) # Add scheduler step for Reduce
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_epoch = epoch
            if folders:
                model_path = os.path.join(folders['models'], f'epoch_{epoch+1}.pth')
                os.makedirs(os.path.dirname(model_path), exist_ok=True) #Ensure the paths exist
                try:
                    torch.save({'epoch': epoch, 'model_state_dict': model.state_dict(),
                                'optimizer_state_dict': optimizer.state_dict(),
                                'scheduler_state_dict': scheduler.state_dict(),  # Save scheduler state
                                'train_loss': avg_train_loss, 'val_loss': avg_val_loss,
                                'best_val_loss': best_val_loss}, model_path)
                    logging.info(f"Saved best model for fold {fold+1} at epoch {best_epoch+1}")
                except OSError as e:  # Catch *specific* OSError (file/path errors)
                    logging.error(f"Error saving model: {e}")
                    # Optionally, re-raise if you *want* to stop on a save failure.
                    # raise

        if folders:  #Save even if not best val loss
            metrics_path = os.path.join(folders['logs'], f'fold_{fold+1}_metrics.csv')
            try:
                with open(metrics_path, 'a') as f:
                    if epoch == 0:
                        f.write('epoch,train_loss,val_loss\n')
                    f.write(f'{epoch+1},{avg_train_loss:.6f},{avg_val_loss:.6f}\n')
            except OSError as e:
                logging.error(f"Error saving metrics: {e}")

    print(f"Fold {fold+1}: Best Val Loss = {best_val_loss:.4f} at epoch {best_epoch + 1}")
    logging.info(f"Fold {fold+1}: Best Val Loss = {best_val_loss:.4f} at epoch {best_epoch + 1}")
    return  best_val_loss # Return only the best val loss



def load_best_params_and_train(db_path: str, train_dataset: Dataset, val_dataset: Dataset, config: dict, folders: dict): # Pass datasets
    """Loads the best parameters and trains a final model."""

    storage_path = f"sqlite:///{db_path}"
    study = optuna.load_study(study_name="siamese_hyperparameter_study", storage=storage_path)
    best_trial = study.best_trial

    print(f"Loading best parameters from trial: {best_trial.number}")
    print(f"Best trial value (validation loss): {best_trial.value}")
    print("Best trial parameters:")
    for key, value in best_trial.params.items():
        print(f"  {key}: {value}")

    # Update the config with the best hyperparameters
    config.update(best_trial.params)

    # NO K-Fold here.  Train on the *entire* training set.
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"Training final model...")

    model = SiameseNetwork(
        paper_dropout=config['paper_dropout'],
        metadata_dropout=config['metadata_dropout'],
        paper_hidden_size=config['paper_hidden_size'],
        metadata_hidden_size=config['metadata_hidden_size']
    )
    model.to(device)

    # Use the *entire* train_dataset
    train_loader = DataLoader(
                            train_dataset, 
                            batch_size=config['batch_size'], 
                            shuffle=True,
                            num_workers=5,
                            pin_memory=True,
                            prefetch_factor=2,
                            persistent_workers=True
                              )
    # Use the *entire* val_dataset
    val_loader = DataLoader(
        val_dataset,
        batch_size=config['batch_size'],
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )

    # Train the model using the updated config
    train_fold(model, train_loader, val_loader, fold=0, num_epochs=config['num_epochs'], learning_rate=config['learning_rate'], folders=folders, device=device)  # No trial object, fold=0 (since it's not K-Fold)

    print("Final training complete.  Models saved in:", folders['models'])



def evaluate_fold(model, val_loader, calibrator=None, save_similarities=True):
  """Evaluates the model on a given fold, optionally calibrating and saving similarities."""
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

  if save_similarities:
    # Convert to tensor and save
    torch.save(torch.tensor(raw_similarities), 'reference_similarities.pt')
    
  raw_similarities = np.array(raw_similarities)

  if calibrator is not None:
      calibrator.fit(torch.tensor(raw_similarities))
      calibrated = calibrator.calibrate(torch.tensor(raw_similarities))
      return raw_similarities, calibrated.numpy()

  return raw_similarities

def load_and_run_specific_trial(trial_number: int, train_dataset: Dataset, val_dataset: Dataset, config: dict, folders: dict):
    """Loads parameters from a specific trial and runs training with those parameters."""
    
    storage_path = "sqlite:///optuna_study.db"
    study = optuna.load_study(study_name="siamese_hyperparameter_study", storage=storage_path)
    
    # Find the specified trial
    target_trial = None
    for trial in study.trials:
        if trial.number == trial_number:
            target_trial = trial
            break
            
    if target_trial is None:
        raise ValueError(f"Trial {trial_number} not found in study.")
        
    print(f"Loading parameters from trial: {trial_number}")
    print(f"Trial value (validation loss): {target_trial.value}")
    print("Trial parameters:")
    for key, value in target_trial.params.items():
        print(f"  {key}: {value}")
        
    # Update config with trial parameters
    config.update(target_trial.params)
    
    # Set augmentation parameters on train dataset
    train_dataset.augment = True
    train_dataset.augment_prob = config['augment_prob']
    train_dataset.augment_noise_scale = config['augment_noise_scale']
    train_dataset.mask_prob = config['mask_prob']
    train_dataset.shuffle_percentage = config['shuffle_percentage']
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    model = SiameseNetwork(
        paper_dropout=config['paper_dropout'],
        metadata_dropout=config['metadata_dropout'],
        paper_hidden_size=config['paper_hidden_size'],
        metadata_hidden_size=config['metadata_hidden_size']
    ).to(device)
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=config['batch_size'],
        shuffle=True,
        num_workers=4,
        pin_memory=True,
        prefetch_factor=2,
        persistent_workers=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config['batch_size'],
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )
    
    # Create the new directory structure
    base_dir = os.path.join('experiments', 'NewModels', f'Trial{trial_number}')
    os.makedirs(base_dir, exist_ok=True)
    
    # Update folders dictionary with new paths
    trial_folders = {
        'models': base_dir,
        'logs': os.path.join(base_dir, 'logs')
    }
    os.makedirs(trial_folders['logs'], exist_ok=True)
    
    # Save trial configuration
    config_path = os.path.join(base_dir, 'trial_config.json')
    with open(config_path, 'w') as f:
        json.dump({
            'trial_number': trial_number,
            'parameters': target_trial.params,
            'original_val_loss': target_trial.value,
            'timestamp': datetime.now().strftime("%Y%m%d_%H%M%S")
        }, f, indent=4)
    
    # Train using parameters from the specified trial
    best_val_loss = train_fold(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        fold=0,
        num_epochs=15,  # Set to 30 epochs as specified
        learning_rate=config['learning_rate'],
        weight_decay=config['weight_decay'],
        folders=trial_folders,  # Use the new folder structure
        device=device,
        temperature=config['temperature'],
        num_negatives=config['num_negatives'],
        hard_fraction=config['hard_fraction'],
        memory_bank_size=config['memory_bank_size']
    )
    
    print(f"Training complete for trial {trial_number}. Final validation loss: {best_val_loss}")
    return best_val_loss




def main():
    parser = argparse.ArgumentParser(description="Siamese Network Training with Optuna")
    parser.add_argument("mode", type=int, choices=[1, 2, 3, 4],
                       help="1 for TEST RUN, 2 for Optuna Optimization, 3 for Final Train, 4 for specific trial")
    parser.add_argument("--trial_number", type=int, help="Trial number to run when using mode 4")
    args = parser.parse_args()

    # --- Configuration ---
    config = {
        'input_file': "THREE_paper_tidiedUp_balanced_four_FINAL.xlsx",
        'batch_size': 64,
        'num_epochs': 100,
        'learning_rate': 1e-4,
        'memory_bank_size': 2048,
        'paper_hidden_size': 128,
        'metadata_hidden_size': 32,
        'paper_dropout': 0.3,
        'metadata_dropout': 0.3,
        'weight_decay': 1e-4,
        'temperature': 0.1,
        'num_negatives': 128,
        'hard_fraction': 0.85,
        'augment_prob': 0.1,
        'augment_noise_scale': 0.01
    }

    # --- Setup Folders and Logging ---
    folders, timestamp = setup_folders()
    log_file = os.path.join(folders['logs'], 'training.log')
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s',
                       handlers=[logging.FileHandler(log_file), logging.StreamHandler()])

    logging.info(f"Starting run at {timestamp}")
    logging.info(f"Config: {config}")

    # --- Load Data ---
    df_paired = pd.read_excel(config['input_file'])
    logging.info(f"Loaded {len(df_paired)} paper pairs.")

    # --- Create Train/Val Splits with CONTROLLED Leakage ---
    leakage_fraction = 0.4
    leaked_df = df_paired.sample(frac=leakage_fraction, random_state=42)
    remaining_df = df_paired.drop(leaked_df.index)
    train_df_clean, val_df_clean = train_val_group_split(remaining_df, group_column='paper1_Main_Topic')
    train_df = pd.concat([train_df_clean, leaked_df])

    # Create Datasets
    train_dataset = PaperPairDataset(train_df)
    val_dataset = PaperPairDataset(val_df_clean)

    train_loader = DataLoader(
        train_dataset,
        batch_size=config['batch_size'],
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )
    val_loader = DataLoader(val_dataset, batch_size=config['batch_size'])

    # --- Model Setup ---
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = SiameseNetwork(
        paper_dropout=config['paper_dropout'],
        metadata_dropout=config['metadata_dropout'],
        paper_hidden_size=config['paper_hidden_size'],
        metadata_hidden_size=config['metadata_hidden_size']
    ).to(device)

    if args.mode == 1:  # Test Run Mode
        config['paper_hidden_size'] = 128
        config['metadata_hidden_size'] = 32
        config['paper_dropout'] = 0.5
        config['metadata_dropout'] = 0.5
        config['weight_decay'] = 5e-4
        config['learning_rate'] = 5e-4
        config['batch_size'] = 128
        config['num_epochs'] = 100
        config['augment_prob'] = 0.25
        config['augment_noise_scale'] = 0.03
        config['mask_prob'] = 0.35
        config['shuffle_percentage'] = 0.15

        best_val_loss = train_fold(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            fold=0,
            num_epochs=config['num_epochs'],
            learning_rate=config['learning_rate'],
            weight_decay=config['weight_decay'],
            folders=folders,
            device=device,
            trial=None
        )
        print(f"Test Run Completed. Best Validation Loss: {best_val_loss}")
        logging.info(f"Test Run Completed. Best Validation Loss: {best_val_loss}")

    elif args.mode == 2:  # Optuna Optimization
        if os.path.exists("optuna_study.db"):
            os.remove("optuna_study.db")
            print("Removed existing study database")

        study_name = "siamese_hyperparameter_study"
        storage_path = f"sqlite:///optuna_study.db"
        
        pruner = MedianPruner(
            n_startup_trials=3,
            n_warmup_steps=5,
            interval_steps=1
        )
        
        study = optuna.create_study(
            direction="minimize",
            study_name=study_name,
            storage=storage_path,
            pruner=pruner,
            sampler=optuna.samplers.TPESampler()
        )
        print(f"Created new Optuna study '{study_name}'")

        try:
            study.optimize(
                lambda trial: objective(trial, train_dataset, val_dataset),
                n_trials=1000,
                timeout=60*60*24,
                gc_after_trial=True
            )
            
            # Save comprehensive results including all trials' data
            results = {
                'best_trial': {
                    'number': study.best_trial.number,
                    'params': study.best_trial.params,
                    'value': study.best_trial.value
                },
                'all_trials': [
                    {
                        'number': trial.number,
                        'params': trial.params,
                        'value': trial.value,
                        'state': str(trial.state),
                        'intermediate_values': {
                            str(step): {
                                'train_loss': values['train_loss'],
                                'val_loss': values['val_loss']
                            } for step, values in trial.intermediate_values.items()
                        }
                    }
                    for trial in study.trials if trial.state.is_finished()
                ],
                'timestamp': datetime.now().strftime("%Y%m%d_%H%M%S")
            }
            
            results_file = os.path.join(folders['logs'], 'optuna_results.json')
            with open(results_file, 'w') as f:
                json.dump(results, f, indent=4)
                
        except KeyboardInterrupt:
            print("\nOptimization interrupted.")
            print("Best trial so far:")
            print(f"Value: {study.best_trial.value}")
            for key, value in study.best_trial.params.items():
                print(f"  {key}: {value}")

    elif args.mode == 3:
        load_best_params_and_train("optuna_study.db", train_dataset, val_dataset, config, folders)

    elif args.mode == 4:
        if args.trial_number is None:
            parser.error("Trial number is required for mode 4")
        try:
            val_loss = load_and_run_specific_trial(
                args.trial_number, 
                train_dataset, 
                val_dataset, 
                config, 
                folders
            )
            print(f"Completed running trial {args.trial_number}")
            logging.info(f"Completed running trial {args.trial_number}")
        except Exception as e:
            print(f"Error running trial {args.trial_number}: {str(e)}")
            logging.error(f"Error running trial {args.trial_number}: {str(e)}")

    else:
        print("Invalid mode. Choose 1 for test run, 2 for Optuna optimization, or 3 for final training.")
        logging.error("Invalid mode selected.")

if __name__ == "__main__":
    main()