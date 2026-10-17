import os
import re
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
from sklearn.metrics import f1_score

# ----------------------------------------------------
# 1. Dataset & Collate Function Definition
# ----------------------------------------------------
class DAICWOZDataset(Dataset):
    """
    Custom Dataset for loading pre-extracted multi-modal features 
    using the DAIC-WOZ / AVEC 2017 splits.
    """
    def __init__(self, csv_path, features_dir, audio_dim=128, text_dim=768):
        self.df = pd.read_csv(csv_path)
        self.features_dir = features_dir
        self.audio_dim = audio_dim
        self.text_dim = text_dim
        
        # Standardize target labels (AVEC 2017 uses 'PHQ8_Binary' or 'PHQ_Binary')
        self.label_col = None
        for col in ['PHQ8_Binary', 'PHQ_Binary', 'Depression', 'label']:
            if col in self.df.columns:
                self.label_col = col
                break
        if not self.label_col:
            raise ValueError(f"Target label column not found in CSV keys: {list(self.df.columns)}")
            
        # Standardize identifier keys (e.g. 'Participant_ID', 'ID')
        self.id_col = None
        for col in ['Participant_ID', 'ID', 'subject_id']:
            if col in self.df.columns:
                self.id_col = col
                break
        if not self.id_col:
            raise ValueError(f"Identifier key column not found in CSV keys: {list(self.df.columns)}")

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        participant_id = int(row[self.id_col])
        label = int(row[self.label_col])
        
        # Paths for cached pre-extracted features
        audio_path = os.path.join(self.features_dir, 'audio', f"{participant_id}.npy")
        text_path = os.path.join(self.features_dir, 'text', f"{participant_id}.npy")
        
        # Load audio features (fallback to zero-tensor if not found)
        if os.path.exists(audio_path):
            audio_feat = np.load(audio_path)
            audio_tensor = torch.tensor(audio_feat, dtype=torch.float32)
        else:
            # Mock placeholder sequence: shape [seq_len=100, audio_dim]
            audio_tensor = torch.randn(100, self.audio_dim)
            
        # Load linguistic features (fallback to zero-tensor if not found)
        if os.path.exists(text_path):
            text_feat = np.load(text_path)
            text_tensor = torch.tensor(text_feat, dtype=torch.float32)
        else:
            # Mock placeholder sequence: shape [seq_len=50, text_dim]
            text_tensor = torch.randn(50, self.text_dim)
            
        return {
            'audio': audio_tensor,
            'text': text_tensor,
            'label': label
        }

def collate_fn(batch):
    """
    Collate function to dynamically pad variable-length audio and text sequences.
    """
    audio_seqs = [item['audio'] for item in batch]
    text_seqs = [item['text'] for item in batch]
    labels = [item['label'] for item in batch]
    
    # Pad sequences to max sequence length in this batch (batch_first=True)
    audio_padded = pad_sequence(audio_seqs, batch_first=True, padding_value=0.0)
    text_padded = pad_sequence(text_seqs, batch_first=True, padding_value=0.0)
    labels_tensor = torch.tensor(labels, dtype=torch.float32).unsqueeze(1)
    
    return {
        'audio': audio_padded,
        'text': text_padded,
        'label': labels_tensor
    }

# ----------------------------------------------------
# 2. Model Architecture
# ----------------------------------------------------
class CrossModalAttention(nn.Module):
    """
    Aligns Query sequences to Key/Value sequences via Multi-head Attention.
    """
    def __init__(self, d_model=256, nhead=4):
        super().__init__()
        self.mha = nn.MultiheadAttention(embed_dim=d_model, num_heads=nhead, batch_first=True)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, query, key_val):
        attn_out, _ = self.mha(query, key_val, key_val)
        return self.norm(query + attn_out)

class DualModalityClassifier(nn.Module):
    """
    Combines Acoustic (WavLM/Mel) and Linguistic (Whisper) features
    using dual Cross-Modal Attention and outputs binary classification probabilities.
    """
    def __init__(self, audio_dim=128, text_dim=768, d_model=256, nhead=4):
        super().__init__()
        # Projections to common d_model space
        self.audio_proj = nn.Linear(audio_dim, d_model)
        self.text_proj = nn.Linear(text_dim, d_model)
        
        # Bidirectional cross-modal attention
        self.audio_queries_text = CrossModalAttention(d_model, nhead)
        self.text_queries_audio = CrossModalAttention(d_model, nhead)
        
        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(d_model * 2, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 1),
            nn.Sigmoid()  # Required for BCELoss output
        )

    def forward(self, audio, text):
        # 1. Project to common hidden dimension
        audio_h = self.audio_proj(audio)
        text_h = self.text_proj(text)
        
        # 2. Align token spaces via attention
        fused_audio = self.audio_queries_text(audio_h, text_h)
        fused_text = self.text_queries_audio(text_h, audio_h)
        
        # 3. Global mean pool along sequence dimensions
        pooled_audio = fused_audio.mean(dim=1)
        pooled_text = fused_text.mean(dim=1)
        
        # 4. Concatenate modalities and classify
        fused_representation = torch.cat([pooled_audio, pooled_text], dim=-1)
        return self.classifier(fused_representation)

# ----------------------------------------------------
# 3. Training & Validation Loop
# ----------------------------------------------------
def train_model():
    # Set explicit device constraint to CUDA 0 (RTX 3050 Laptop friendly)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"[Hardware Configuration] Active Device: {device}")
    
    # Configure input directories & hyper-parameters
    features_dir = './features'
    train_csv = 'train_split_Depression_AVEC2017.csv'
    dev_csv = 'dev_split_Depression_AVEC2017.csv'
    
    audio_dim = 128
    text_dim = 768
    batch_size = 4  # Saved constraint to avoid laptop GPU Out-Of-Memory errors
    epochs = 50
    patience = 5
    best_val_loss = float('inf')
    early_stop_counter = 0
    
    # Mock files creation if they do not exist (for demonstration safety)
    os.makedirs(os.path.join(features_dir, 'audio'), exist_ok=True)
    os.makedirs(os.path.join(features_dir, 'text'), exist_ok=True)
    
    for csv_file in [train_csv, dev_csv]:
        if not os.path.exists(csv_file):
            print(f"[Warning] Split file '{csv_file}' missing. Creating mock CSV splits.")
            mock_df = pd.DataFrame({
                'Participant_ID': [303, 304, 305, 306, 307, 308, 309, 310],
                'PHQ8_Binary': [0, 1, 0, 0, 1, 0, 1, 0]
            })
            mock_df.to_csv(csv_file, index=False)

    # Initialize datasets & loaders
    train_dataset = DAICWOZDataset(train_csv, features_dir, audio_dim, text_dim)
    dev_dataset = DAICWOZDataset(dev_csv, features_dir, audio_dim, text_dim)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
    dev_loader = DataLoader(dev_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)
    
    # Initialize Model, Loss Function and Optimizer
    model = DualModalityClassifier(audio_dim=audio_dim, text_dim=text_dim).to(device)
    criterion = nn.BCELoss()
    optimizer = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-2)
    
    print("\n--- Training Loop Initialized ---")
    for epoch in range(1, epochs + 1):
        # 1. Training Phase
        model.train()
        train_loss = 0.0
        for batch in train_loader:
            audio = batch['audio'].to(device)
            text = batch['text'].to(device)
            labels = batch['label'].to(device)
            
            optimizer.zero_grad()
            outputs = model(audio, text)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item() * audio.size(0)
            
        train_loss /= len(train_loader.dataset)
        
        # 2. Validation Phase
        model.eval()
        val_loss = 0.0
        val_preds = []
        val_targets = []
        
        with torch.no_grad():
            for batch in dev_loader:
                audio = batch['audio'].to(device)
                text = batch['text'].to(device)
                labels = batch['label'].to(device)
                
                outputs = model(audio, text)
                loss = criterion(outputs, labels)
                val_loss += loss.item() * audio.size(0)
                
                preds = (outputs >= 0.5).int().cpu().numpy()
                val_preds.extend(preds.flatten())
                val_targets.extend(labels.cpu().numpy().flatten())
                
        val_loss /= len(dev_loader.dataset)
        val_f1 = f1_score(val_targets, val_preds, zero_division=0)
        
        print(f"Epoch {epoch:02d}/{epochs:02d} | "
              f"Train Loss: {train_loss:.4f} | "
              f"Val Loss: {val_loss:.4f} | "
              f"Val F1-Score: {val_f1:.4f}")
        
        # 3. Early Stopping Check
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), 'best_dual_modality_model.pt')
            early_stop_counter = 0
        else:
            early_stop_counter += 1
            if early_stop_counter >= patience:
                print(f"[Early Stopping] Validation loss did not improve for {patience} epochs. Stopping training.")
                break

if __name__ == "__main__":
    train_model()
