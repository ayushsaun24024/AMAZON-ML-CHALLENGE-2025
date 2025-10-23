#!/usr/bin/env python3


import os
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split
from transformers import (
    AutoTokenizer, AutoModel, 
    get_linear_schedule_with_warmup,
    AutoConfig
)
import re
import random
import warnings
from tqdm.auto import tqdm
import json
from datetime import datetime
import gc


# Configuration
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
warnings.filterwarnings('ignore')
torch.backends.cudnn.benchmark = True


MODEL_NAME = "microsoft/deberta-v3-base"
SEED = 42
MAX_LENGTH = 384
TRAIN_FILE = "train.csv"
TEST_FILE = "test.csv"
OUTPUT_FILE = "prediction_23_46.csv"
MODEL_DIR = "./best_model_23_46"


# Training parameters
EPOCHS = 12
BATCH_SIZE = 8
LEARNING_RATE = 1.5e-5
WARMUP_RATIO = 0.1
WEIGHT_DECAY = 0.01
PATIENCE = 4
GRADIENT_ACCUMULATION_STEPS = 4
MAX_GRAD_NORM = 1.0
USE_AMP = True


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


set_seed(SEED)


class RobustPriceProcessor:
    
    def __init__(self):
        self.feature_stats = {}
        self.price_stats = {}
        self.is_fitted = False
        
    def extract_numeric_features(self, text):
        """Extract numeric features from product description"""
        text = str(text)
        features = {}
        
        # Extract value and unit
        value_match = re.search(r'Value:\s*([0-9.]+)', text)
        features['value'] = float(value_match.group(1)) if value_match else 1.0
        
        # Extract pack size
        pack_match = re.search(r'Pack of\s*([0-9]+)', text, re.IGNORECASE)
        features['pack_size'] = int(pack_match.group(1)) if pack_match else 1
        
        # Count bullet points
        features['bullet_count'] = len(re.findall(r'Bullet Point', text))
        
        # Text characteristics
        features['text_length'] = len(text)
        features['word_count'] = len(text.split())
        
        # Detect product type patterns
        features['has_weight'] = 1 if any(unit in text.lower() for unit in ['ounce', 'oz', 'pound', 'lb', 'gram', 'g']) else 0
        features['has_volume'] = 1 if any(unit in text.lower() for unit in ['fl oz', 'ml', 'liter', 'gallon']) else 0
        features['has_count'] = 1 if 'pack of' in text.lower() else 0
        
        return features
    
    def fit(self, train_df):
        """Fit processor on training data only"""
        print("Fitting processor on training data...")
        
        # Clean training data
        train_df = train_df.dropna(subset=['price', 'catalog_content']).copy()
        train_df['catalog_content'] = train_df['catalog_content'].fillna('')
        
        # Extract features for training data
        feature_data = []
        for text in train_df['catalog_content']:
            feature_data.append(self.extract_numeric_features(text))
        
        feature_df = pd.DataFrame(feature_data)
        
        # Store feature statistics from training data
        for col in feature_df.columns:
            self.feature_stats[col] = {
                'mean': feature_df[col].mean(),
                'std': feature_df[col].std(),
                'min': feature_df[col].min(),
                'max': feature_df[col].max()
            }
        
        # Store price statistics
        self.price_stats = {
            'mean': train_df['price'].mean(),
            'std': train_df['price'].std(),
            'min': train_df['price'].min(),
            'max': train_df['price'].max()
        }
        
        # Remove extreme outliers (more conservative)
        q1, q3 = train_df['price'].quantile(0.01), train_df['price'].quantile(0.99)
        iqr = q3 - q1
        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr
        
        train_df = train_df[(train_df['price'] >= lower_bound) & (train_df['price'] <= upper_bound)]
        
        print(f"Training data after outlier removal: {len(train_df)} samples")
        print(f"Price range: ${self.price_stats['min']:.2f} - ${self.price_stats['max']:.2f}")
        self.is_fitted = True
        
        return train_df
    
    def transform(self, df, is_train=False):
        """Transform data using fitted statistics"""
        if not self.is_fitted:
            raise ValueError("Processor must be fitted before transformation")
        
        df = df.copy()
        df['catalog_content'] = df['catalog_content'].fillna('')
        
        # Extract features
        feature_data = []
        for text in df['catalog_content']:
            feature_data.append(self.extract_numeric_features(text))
        
        feature_df = pd.DataFrame(feature_data)
        
        # Normalize features using training statistics
        for col in feature_df.columns:
            if col in self.feature_stats:
                stats = self.feature_stats[col]
                if stats['std'] > 0:
                    feature_df[col] = (feature_df[col] - stats['mean']) / stats['std']
                else:
                    feature_df[col] = 0.0
        
        # Replace any NaN or inf values in features
        feature_df = feature_df.replace([np.inf, -np.inf], 0.0)
        feature_df = feature_df.fillna(0.0)
        
        # Add normalized features to dataframe
        for col in feature_df.columns:
            df[f'feature_{col}'] = feature_df[col]
        
        # Transform prices for training data only
        if is_train and 'price' in df.columns:
            # Clean prices first
            df['price'] = df['price'].replace([np.inf, -np.inf], self.price_stats['mean'])
            df['price'] = df['price'].fillna(self.price_stats['mean'])
            # Clip to reasonable range
            df['price'] = np.clip(df['price'], 0.1, self.price_stats['max'] * 2)
            # Log transform
            df['price'] = np.log1p(df['price'])
        
        return df
    
    def inverse_transform(self, prices):
        """Convert predictions back to original scale with NaN protection"""
        prices = np.array(prices, dtype=np.float64)
        

        prices = np.nan_to_num(
            prices, 
            nan=np.log1p(self.price_stats['mean']),  # Replace NaN with log of mean
            posinf=np.log1p(self.price_stats['max']),  # Replace +inf
            neginf=np.log1p(self.price_stats['min'])   # Replace -inf
        )
        
        # Clip log values to reasonable range before expm1
        max_log_value = np.log1p(self.price_stats['max'] * 3)
        prices = np.clip(prices, -1, max_log_value)
        
        # Apply inverse log transform
        prices = np.expm1(prices)
        
        # Final safety clip
        prices = np.clip(prices, 0.1, self.price_stats['max'] * 2)
        
        # Final NaN check
        prices = np.nan_to_num(prices, nan=self.price_stats['mean'])
        
        return prices


class SmartTextDataset(Dataset):
    """Efficient dataset with text and features"""
    
    def __init__(self, df, tokenizer, max_length=384):
        self.df = df.reset_index(drop=True)
        self.tokenizer = tokenizer
        self.max_length = max_length
        
        self.texts = self.df['catalog_content'].fillna('').astype(str).tolist()
        
        # Prepare feature vectors
        feature_cols = [col for col in df.columns if col.startswith('feature_')]
        self.features = []
        for _, row in self.df[feature_cols].iterrows():
            feature_array = row.values.astype(np.float32)

            feature_array = np.nan_to_num(feature_array, nan=0.0)
            self.features.append(torch.tensor(feature_array, dtype=torch.float))
        
        if 'price' in self.df.columns:
            prices = self.df['price'].astype(np.float32).values
            # Clean prices
            prices = np.nan_to_num(prices, nan=0.0)
            self.prices = prices.tolist()
        else:
            self.prices = None
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = self.texts[idx]
        
        encoding = self.tokenizer(
            text,
            max_length=self.max_length,
            padding=False,
            truncation=True,
            return_tensors=None
        )
        
        item = {
            'input_ids': encoding['input_ids'],
            'attention_mask': encoding['attention_mask'],
            'features': self.features[idx]
        }
        
        if self.prices is not None:
            item['labels'] = torch.tensor(self.prices[idx], dtype=torch.float)
        
        return item


def smart_collate_fn(batch, tokenizer, max_length=384):
    """Efficient collate function"""
    input_ids = [item['input_ids'] for item in batch]
    attention_mask = [item['attention_mask'] for item in batch]
    features = [item['features'] for item in batch]
    
    # Pad sequences
    padded = tokenizer.pad(
        {'input_ids': input_ids, 'attention_mask': attention_mask},
        padding='longest',
        max_length=max_length,
        return_tensors='pt'
    )
    
    result = {
        'input_ids': padded['input_ids'],
        'attention_mask': padded['attention_mask'],
        'features': torch.stack(features)
    }
    
    if 'labels' in batch[0]:
        labels = torch.tensor([item['labels'] for item in batch], dtype=torch.float)
        result['labels'] = labels
    
    return result


class FastRegressor(nn.Module):
    """Optimized regression model with initialization"""
    
    def __init__(self, model_name, feature_dim, dropout=0.1):
        super().__init__()
        
        self.transformer = AutoModel.from_pretrained(model_name)
        hidden_size = self.transformer.config.hidden_size
        
        # Text processing head
        self.text_proj = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden_size, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, 128),
            nn.ReLU(),
        )
        
        # Feature processing head  
        self.feature_proj = nn.Sequential(
            nn.Linear(feature_dim, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 32),
            nn.ReLU(),
        )
        
        # Fusion head
        self.fusion = nn.Sequential(
            nn.Linear(128 + 32, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )
        
        # Initialize weights properly
        self._init_weights()
        
    def _init_weights(self):
        """Initialize weights to prevent NaN"""
        for module in [self.text_proj, self.feature_proj, self.fusion]:
            for m in module:
                if isinstance(m, nn.Linear):
                    nn.init.xavier_normal_(m.weight, gain=0.1)
                    if m.bias is not None:
                        nn.init.constant_(m.bias, 0)
        
    def forward(self, input_ids, attention_mask, features, labels=None):
        # Text embeddings
        outputs = self.transformer(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=False,
            return_dict=True
        )
        
        # Use [CLS] token representation
        text_emb = outputs.last_hidden_state[:, 0, :]
        text_features = self.text_proj(text_emb)
        
        # Process numeric features
        feature_emb = self.feature_proj(features)
        
        # Concatenate and predict
        combined = torch.cat([text_features, feature_emb], dim=1)
        logits = self.fusion(combined).squeeze(-1)
        
        loss = None
        if labels is not None:
            loss = nn.HuberLoss(delta=1.0)(logits, labels)
        
        return (loss, logits) if loss is not None else logits


def calculate_smape(y_true, y_pred):
    """Symmetric Mean Absolute Percentage Error with NaN handling"""
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    
    # Remove any NaN or inf
    mask = np.isfinite(y_true) & np.isfinite(y_pred)
    y_true = y_true[mask]
    y_pred = y_pred[mask]
    
    if len(y_true) == 0:
        return 100.0
    
    return 100 * np.mean(2 * np.abs(y_pred - y_true) / (np.abs(y_true) + np.abs(y_pred) + 1e-8))


class SimpleEarlyStopping:
    """Early stopping with patience"""
    
    def __init__(self, patience=5, min_delta=0.0):
        self.patience = patience
        self.min_delta = min_delta
        self.best_score = None
        self.counter = 0
        
    def __call__(self, score):
        if self.best_score is None:
            self.best_score = score
            return False
        
        if score < self.best_score - self.min_delta:
            self.best_score = score
            self.counter = 0
            return False
        else:
            self.counter += 1
            return self.counter >= self.patience


def clear_gpu_memory():
    """Clear GPU memory safely"""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()


def train_fast_model():
    """Fast and robust training function with NaN protection"""
    print("🚀 Starting fast training...")
    
    # Load data
    train_df = pd.read_csv(TRAIN_FILE)
    test_df = pd.read_csv(TEST_FILE)
    
    processor = RobustPriceProcessor()
    train_df = processor.fit(train_df)  # Fit only on training data
    
    # Transform training data
    train_df = processor.transform(train_df, is_train=True)
    
    train_split, val_split = train_test_split(
        train_df, 
        test_size=0.15, 
        random_state=SEED, 
        shuffle=True
    )
    
    print(f"📊 Data split: Train={len(train_split)}, Val={len(val_split)}")
    
    # Initialize model and tokenizer
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    
    # Get feature dimension
    feature_cols = [col for col in train_split.columns if col.startswith('feature_')]
    feature_dim = len(feature_cols)
    
    model = FastRegressor(MODEL_NAME, feature_dim)
    
    # Setup device
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    
    # Create datasets
    train_dataset = SmartTextDataset(train_split, tokenizer, MAX_LENGTH)
    val_dataset = SmartTextDataset(val_split, tokenizer, MAX_LENGTH)
    
    # Data loaders
    train_loader = DataLoader(
        train_dataset, 
        batch_size=BATCH_SIZE, 
        shuffle=True,
        num_workers=2,
        pin_memory=True,
        collate_fn=lambda x: smart_collate_fn(x, tokenizer, MAX_LENGTH)
    )
    
    val_loader = DataLoader(
        val_dataset, 
        batch_size=BATCH_SIZE * 2,
        shuffle=False,
        num_workers=2,
        pin_memory=True,
        collate_fn=lambda x: smart_collate_fn(x, tokenizer, MAX_LENGTH)
    )
    
    # Optimizer
    no_decay = ['bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {
            'params': [p for n, p in model.named_parameters() 
                      if not any(nd in n for nd in no_decay)],
            'weight_decay': WEIGHT_DECAY,
        },
        {
            'params': [p for n, p in model.named_parameters() 
                      if any(nd in n for nd in no_decay)],
            'weight_decay': 0.0,
        },
    ]
    
    optimizer = torch.optim.AdamW(optimizer_grouped_parameters, lr=LEARNING_RATE)
    
    # Scheduler
    num_training_steps = len(train_loader) * EPOCHS // GRADIENT_ACCUMULATION_STEPS
    num_warmup_steps = int(num_training_steps * WARMUP_RATIO)
    
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=num_warmup_steps,
        num_training_steps=num_training_steps
    )
    
    # Mixed precision
    scaler = torch.cuda.amp.GradScaler(enabled=USE_AMP)
    
    # Early stopping
    early_stopping = SimpleEarlyStopping(patience=PATIENCE)
    
    print(f" Training for {EPOCHS} epochs...")
    
    best_smape = float('inf')
    history = []
    
    for epoch in range(EPOCHS):
        print(f"\n Epoch {epoch + 1}/{EPOCHS}")
        
        # Training phase
        model.train()
        total_loss = 0
        train_bar = tqdm(train_loader, desc='Training')
        
        optimizer.zero_grad()
        
        for step, batch in enumerate(train_bar):
            # Move to device
            input_ids = batch['input_ids'].to(device, non_blocking=True)
            attention_mask = batch['attention_mask'].to(device, non_blocking=True)
            features = batch['features'].to(device, non_blocking=True)
            labels = batch['labels'].to(device, non_blocking=True)
            
            # Forward pass with mixed precision
            with torch.cuda.amp.autocast(enabled=USE_AMP):
                loss, _ = model(input_ids, attention_mask, features, labels)
                loss = loss / GRADIENT_ACCUMULATION_STEPS
            
            # Check for NaN loss
            if torch.isnan(loss) or torch.isinf(loss):
                print(f"\nWARNING: NaN/Inf loss detected at step {step}. Skipping batch.")
                optimizer.zero_grad()
                continue
            
            # Backward pass
            scaler.scale(loss).backward()
            
            if (step + 1) % GRADIENT_ACCUMULATION_STEPS == 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), MAX_GRAD_NORM)
                scaler.step(optimizer)
                scaler.update()
                scheduler.step()
                optimizer.zero_grad()
            
            total_loss += loss.item() * GRADIENT_ACCUMULATION_STEPS
            train_bar.set_postfix({'loss': f'{loss.item():.4f}'})
            
            # Clear memory periodically
            if step % 100 == 0:
                clear_gpu_memory()
        
        avg_train_loss = total_loss / len(train_loader)
        
        # Validation phase
        model.eval()
        val_preds = []
        val_targets = []
        
        with torch.no_grad():
            for batch in tqdm(val_loader, desc='Validating'):
                input_ids = batch['input_ids'].to(device, non_blocking=True)
                attention_mask = batch['attention_mask'].to(device, non_blocking=True)
                features = batch['features'].to(device, non_blocking=True)
                labels = batch['labels'].to(device, non_blocking=True)
                
                with torch.cuda.amp.autocast(enabled=USE_AMP):
                    outputs = model(input_ids, attention_mask, features)
                
                val_preds.extend(outputs.cpu().numpy())
                val_targets.extend(labels.cpu().numpy())
        
        # Convert to numpy and check for NaN
        val_preds = np.array(val_preds)
        val_targets = np.array(val_targets)
        
        print(f"\n Pre-transform check:")
        print(f"   Predictions - NaN: {np.isnan(val_preds).sum()}, Inf: {np.isinf(val_preds).sum()}")
        print(f"   Targets - NaN: {np.isnan(val_targets).sum()}, Inf: {np.isinf(val_targets).sum()}")
        
        # Convert back to original scale for metrics
        val_preds_orig = processor.inverse_transform(val_preds)
        val_targets_orig = processor.inverse_transform(val_targets)
        
        print(f" Post-transform check:")
        print(f"   Predictions - NaN: {np.isnan(val_preds_orig).sum()}, Inf: {np.isinf(val_preds_orig).sum()}")
        print(f"   Targets - NaN: {np.isnan(val_targets_orig).sum()}, Inf: {np.isinf(val_targets_orig).sum()}")
        
        # Final safety check - replace any remaining NaN/inf
        val_preds_orig = np.nan_to_num(val_preds_orig, nan=processor.price_stats['mean'])
        val_targets_orig = np.nan_to_num(val_targets_orig, nan=processor.price_stats['mean'])
        
        # Calculate metrics
        val_smape = calculate_smape(val_targets_orig, val_preds_orig)
        val_mae = mean_absolute_error(val_targets_orig, val_preds_orig)
        
        print(f"Epoch {epoch + 1} Results:")
        print(f"   Train Loss: {avg_train_loss:.4f}")
        print(f"   Val SMAPE:  {val_smape:.4f}%")
        print(f"   Val MAE:    {val_mae:.4f}")
        
        history.append({
            'epoch': epoch + 1,
            'train_loss': avg_train_loss,
            'val_smape': val_smape,
            'val_mae': val_mae
        })
        
        # Save best model
        if val_smape < best_smape:
            best_smape = val_smape
            os.makedirs(MODEL_DIR, exist_ok=True)
            
            # Save model checkpoint
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'smape': val_smape,
                'processor': processor,
            }, os.path.join(MODEL_DIR, 'best_model.pth'))
            
            tokenizer.save_pretrained(MODEL_DIR)
            print(f"New best model saved! SMAPE: {val_smape:.4f}%")
        
        # Early stopping check
        if early_stopping(val_smape):
            print(f" Early stopping at epoch {epoch + 1}")
            break
        
        clear_gpu_memory()
    
    print(f"\n Training completed! Best SMAPE: {best_smape:.4f}%")
    
    # Transform test data for prediction
    test_df_processed = processor.transform(test_df, is_train=False)
    
    return tokenizer, model, processor, test_df_processed, best_smape, history


def predict_fast(model, processor, tokenizer, test_df):
    """Fast and safe prediction with NaN protection"""
    print("\nGenerating predictions...")
    
    model.eval()
    device = next(model.parameters()).device
    
    # Create test dataset
    test_dataset = SmartTextDataset(test_df, tokenizer, MAX_LENGTH)
    test_loader = DataLoader(
        test_dataset,
        batch_size=64,  # Larger batch for inference
        shuffle=False,
        num_workers=2,
        collate_fn=lambda x: smart_collate_fn(x, tokenizer, MAX_LENGTH)
    )
    
    predictions = []
    
    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Predicting"):
            input_ids = batch['input_ids'].to(device, non_blocking=True)
            attention_mask = batch['attention_mask'].to(device, non_blocking=True)
            features = batch['features'].to(device, non_blocking=True)
            
            with torch.cuda.amp.autocast(enabled=USE_AMP):
                batch_preds = model(input_ids, attention_mask, features)
            
            predictions.extend(batch_preds.cpu().numpy())
            
            # Clear memory
            if len(predictions) % 1000 == 0:
                clear_gpu_memory()
    
    # Convert to numpy
    predictions = np.array(predictions)
    
    # Check for NaN before transformation
    print(f" Predictions before transform - NaN: {np.isnan(predictions).sum()}, Inf: {np.isinf(predictions).sum()}")
    
    # Convert to original price scale
    final_predictions = processor.inverse_transform(predictions)
    
    # Final check
    print(f" Predictions after transform - NaN: {np.isnan(final_predictions).sum()}, Inf: {np.isinf(final_predictions).sum()}")
    
    # Ensure no NaN or infinite values
    final_predictions = np.nan_to_num(
        final_predictions, 
        nan=processor.price_stats['mean'], 
        posinf=processor.price_stats['max'], 
        neginf=processor.price_stats['min']
    )
    
    # Create output
    output_df = pd.DataFrame({
        'sample_id': test_df['sample_id'].values,
        'price': final_predictions
    })
    
    output_df.to_csv(OUTPUT_FILE, index=False)
    
    # Statistics
    print(f"\n Prediction Statistics:")
    print(f"   Mean price: ${output_df['price'].mean():.2f}")
    print(f"   Std price:  ${output_df['price'].std():.2f}")
    print(f"   Min price:  ${output_df['price'].min():.2f}")
    print(f"   Max price:  ${output_df['price'].max():.2f}")
    print(f"   Median:     ${output_df['price'].median():.2f}")
    
    print(f" Predictions saved to: {OUTPUT_FILE}")
    
    return output_df


def load_best_model():
    """Load the best saved model safely"""
    print(" Loading best model...")
    
    checkpoint_path = os.path.join(MODEL_DIR, 'best_model.pth')
    
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError("No trained model found! Train first.")
    
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    processor = checkpoint['processor']
    tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
    
    # Get feature dimension
    feature_cols = [col for col in processor.feature_stats.keys()]
    feature_dim = len(feature_cols)
    
    # Load model
    model = FastRegressor(MODEL_NAME, feature_dim)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    model.eval()
    
    print(f" Loaded best model (Epoch {checkpoint['epoch']}, SMAPE: {checkpoint['smape']:.4f}%)")
    
    return model, processor, tokenizer


def main():
    """Main execution pipeline"""
    print("=" * 60)
    print(" ULTRA OPTIMIZED PRICE PREDICTION")
    print("=" * 60)
    print(f"Model: {MODEL_NAME}")
    print(f"Epochs: {EPOCHS}, Batch: {BATCH_SIZE}x{GRADIENT_ACCUMULATION_STEPS}")
    print("=" * 60)
    
    os.makedirs(MODEL_DIR, exist_ok=True)
    
    try:
        # GPU info
        if torch.cuda.is_available():
            print(f"GPU: {torch.cuda.get_device_name(0)}")
            print(f"Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
        
        # Train or load model
        if not os.path.exists(os.path.join(MODEL_DIR, 'best_model.pth')):
            print(" Training new model...")
            tokenizer, model, processor, test_df, best_smape, history = train_fast_model()
        else:
            print(" Loading pre-trained model...")
            model, processor, tokenizer = load_best_model()
            test_df = pd.read_csv(TEST_FILE)
            # Transform test data using fitted processor
            test_df = processor.transform(test_df, is_train=False)
            best_smape = "N/A"
        
        # Generate predictions
        predictions_df = predict_fast(model, processor, tokenizer, test_df)
        
        print("\n" + "=" * 60)
        print("PIPELINE COMPLETED SUCCESSFULLY!")
        if best_smape != "N/A":
            print(f" Best Validation SMAPE: {best_smape:.4f}%")
        print(f" Model: {MODEL_DIR}")
        print(f" Predictions: {OUTPUT_FILE}")
        print("=" * 60)
        
    except Exception as e:
        print(f"\n ERROR: {str(e)}")
        import traceback
        traceback.print_exc()
        raise


if __name__ == "__main__":
    main()
