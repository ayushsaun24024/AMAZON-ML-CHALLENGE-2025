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
    get_cosine_schedule_with_warmup,
    PretrainedConfig
)
import random
import warnings
from tqdm.auto import tqdm
import json
from datetime import datetime
import gc
from torch.cuda.amp import autocast, GradScaler
import torch.nn.functional as F

os.environ['CUDA_VISIBLE_DEVICES'] = '1'
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:512'
os.environ['TOKENIZERS_PARALLELISM'] = 'false'

warnings.filterwarnings('ignore')
torch.backends.cudnn.benchmark = True
torch.backends.cudnn.deterministic = False
torch.set_float32_matmul_precision('high')

MODEL_NAME = "microsoft/deberta-v3-small"
SEED = 42
MAX_LENGTH = 320
TRAIN_FILE = "train.csv"
TEST_FILE = "test.csv"
OUTPUT_FILE = "prediction.csv"
CHECKPOINT_DIR = "./checkpoints"
MODEL_DIR = "./best_model"

EPOCHS = 15
BATCH_SIZE = 24
LEARNING_RATE = 2.5e-5
WARMUP_RATIO = 0.05
WEIGHT_DECAY = 0.01
PATIENCE = 5
GRADIENT_ACCUMULATION_STEPS = 1
MAX_GRAD_NORM = 1.0
USE_AMP = True

USE_FLASH_ATTENTION = True
USE_CHECKPOINT = True
USE_TORCH_COMPILE = False
ENABLE_AUTOCAST_FALLBACK = True

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

set_seed(SEED)

class amazonDataset(Dataset):
    
    def __init__(self, texts, prices=None, tokenizer=None, max_length=320, precompute=False):
        self.texts = texts
        self.prices = prices
        self.max_length = max_length
        
        if precompute and tokenizer is not None:
            self.encodings = [
                tokenizer(
                    text,
                    max_length=max_length,
                    truncation=True,
                    return_tensors='pt'
                )
                for text in tqdm(texts, desc='Tokenizing')
            ]
            self.tokenizer = None
        else:
            self.tokenizer = tokenizer
            self.encodings = None
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        if self.encodings is not None:
            enc = self.encodings[idx]
            item = {
                'input_ids': enc['input_ids'].squeeze(),
                'attention_mask': enc['attention_mask'].squeeze(),
            }
        else:
            text = self.texts[idx]
            enc = self.tokenizer(
                text,
                max_length=self.max_length,
                truncation=True,
                return_tensors=None
            )
            item = {
                'input_ids': enc['input_ids'],
                'attention_mask': enc['attention_mask'],
            }
        
        if self.prices is not None:
            item['labels'] = torch.tensor(self.prices[idx], dtype=torch.float16)
        
        return item

def ultra_collate_fn(batch):
    input_ids_list = [item['input_ids'] for item in batch]
    attention_mask_list = [item['attention_mask'] for item in batch]
    
    max_len = max(len(x) for x in input_ids_list)
    max_len = min(max_len, 320)
    
    input_ids = torch.zeros(len(batch), max_len, dtype=torch.long)
    attention_mask = torch.zeros(len(batch), max_len, dtype=torch.long)
    
    for i, (ids, mask) in enumerate(zip(input_ids_list, attention_mask_list)):
        len_ids = min(len(ids), max_len)
        input_ids[i, :len_ids] = torch.tensor(ids[:len_ids], dtype=torch.long)
        attention_mask[i, :len_ids] = torch.tensor(mask[:len_ids], dtype=torch.long)
    
    result = {
        'input_ids': input_ids,
        'attention_mask': attention_mask,
    }
    
    if 'labels' in batch[0]:
        labels = torch.stack([item['labels'] for item in batch])
        result['labels'] = labels
    
    return result

class RegressorFunction(nn.Module):
    
    def __init__(self, model_name, dropout=0.15):
        super().__init__()
        
        self.transformer = AutoModel.from_pretrained(
            model_name,
            torch_dtype=torch.float32
        )
        
        if USE_CHECKPOINT:
            self.transformer.gradient_checkpointing_enable()
        
        hidden_size = self.transformer.config.hidden_size
        
        self.regressor = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden_size, 256),
            nn.BatchNorm1d(256),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.GELU(),
            nn.Linear(64, 1)
        )
        
        self._init_weights()
    
    def _init_weights(self):
        for m in self.regressor:
            if isinstance(m, nn.Linear):
                nn.init.kaiming_uniform_(m.weight, nonlinearity='relu')
                nn.init.zeros_(m.bias)
    
    def forward(self, input_ids, attention_mask, labels=None):
        with autocast(enabled=USE_AMP, dtype=torch.float16):
            outputs = self.transformer(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_hidden_states=False,
                return_dict=True
            )
        
        hidden = outputs.last_hidden_state
        
        mask_exp = attention_mask.unsqueeze(-1).float()
        sum_hidden = torch.sum(hidden * mask_exp, dim=1)
        sum_mask = torch.clamp(torch.sum(mask_exp, dim=1), min=1e-9)
        pooled = sum_hidden / sum_mask
        
        logits = self.regressor(pooled).squeeze(-1)
        
        loss = None
        if labels is not None:
            loss = F.smooth_l1_loss(logits, labels, beta=1.0)
        
        return (loss, logits) if loss is not None else logits

class Processor:
    
    def __init__(self):
        self.stats = {}
    
    def preprocess(self, train_df, test_df):
        train_df = train_df.dropna(subset=['price', 'catalog_content']).copy()
        train_df['catalog_content'] = train_df['catalog_content'].fillna('').astype(str)
        test_df['catalog_content'] = test_df['catalog_content'].fillna('').astype(str)
        
        train_df['catalog_content'] = train_df['catalog_content'].str[:512]
        test_df['catalog_content'] = test_df['catalog_content'].str[:512]
        
        q1, q99 = train_df['price'].quantile([0.01, 0.99])
        train_df = train_df[(train_df['price'] >= q1) & (train_df['price'] <= q99)]
        
        prices = np.log1p(train_df['price'].values)
        train_df['price'] = prices
        
        self.stats = {
            'price_mean': prices.mean(),
            'price_std': prices.std(),
        }
        
        train_df['price'] = (train_df['price'] - self.stats['price_mean']) / (self.stats['price_std'] + 1e-8)
        
        return train_df, test_df
    
    def inverse_transform(self, prices):
        prices = np.array(prices)
        prices = prices * (self.stats['price_std'] + 1e-8) + self.stats['price_mean']
        prices = np.expm1(prices)
        return np.maximum(prices, 0.1)

def calculate_smape(y_true, y_pred):
    return 100 * np.mean(
        2 * np.abs(y_pred - y_true) / (np.abs(y_true) + np.abs(y_pred) + 1e-8)
    )

class TrackFunction:
    
    def __init__(self, ckpt_dir, patience=5):
        self.ckpt_dir = ckpt_dir
        self.patience = patience
        self.best_smape = float('inf')
        self.best_epoch = 0
        self.no_improve = 0
        self.history = []
        os.makedirs(ckpt_dir, exist_ok=True)
    
    def update(self, epoch, smape, loss, model, opt, sched):
        self.history.append({
            'epoch': epoch,
            'smape': smape,
            'loss': loss,
            'time': datetime.now().isoformat()
        })
        
        improved = False
        if smape < self.best_smape:
            self.best_smape = smape
            self.best_epoch = epoch
            self.no_improve = 0
            improved = True
            
            torch.save({
                'epoch': epoch,
                'model': model.state_dict(),
                'opt': opt.state_dict(),
                'sched': sched.state_dict() if sched else None,
                'smape': smape,
                'loss': loss,
            }, os.path.join(self.ckpt_dir, 'best.pth'))
            
            print(f"BEST Epoch {epoch}: SMAPE={smape:.4f}%")
        else:
            self.no_improve += 1
        
        with open(os.path.join(self.ckpt_dir, 'history.json'), 'w') as f:
            json.dump(self.history, f)
        
        return improved
    
    def should_stop(self):
        return self.no_improve >= self.patience

def clear_gpu_mem():
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()

def train():
    print(f"GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}")
    print(f"Batch: {BATCH_SIZE}, Epochs: {EPOCHS}, LR: {LEARNING_RATE}")

    train_df = pd.read_csv(TRAIN_FILE)
    test_df = pd.read_csv(TEST_FILE)
    
    proc = Processor()
    train_df, test_df = proc.preprocess(train_df, test_df)
    
    train_split, val_split = train_test_split(
        train_df, test_size=0.15, random_state=SEED, shuffle=True
    )
    
    print(f"Train: {len(train_split)}, Val: {len(val_split)}, Test: {len(test_df)}")
    
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = RegressorFunction(MODEL_NAME)
    
    device = torch.device('cuda:0')
    model.to(device)
    
    train_ds = amazonDataset(
        train_split['catalog_content'].values,
        train_split['price'].values,
        tokenizer,
        MAX_LENGTH
    )
    val_ds = amazonDataset(
        val_split['catalog_content'].values,
        val_split['price'].values,
        tokenizer,
        MAX_LENGTH
    )
    
    train_dl = DataLoader(
        train_ds,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
        drop_last=True,
        collate_fn=ultra_collate_fn,
        prefetch_factor=2
    )
    
    val_dl = DataLoader(
        val_ds,
        batch_size=BATCH_SIZE * 2,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
        collate_fn=ultra_collate_fn
    )
    
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
        amsgrad=True
    )
    
    num_steps = len(train_dl) * EPOCHS
    warmup_steps = int(num_steps * WARMUP_RATIO)
    
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=num_steps
    )
    
    scaler = GradScaler(enabled=USE_AMP)
    tracker = TrackFunction(CHECKPOINT_DIR, PATIENCE)
    
    print(f"Total steps: {num_steps}, Warmup: {warmup_steps}\n")
    
    for epoch in range(EPOCHS):
        print(f"EPOCH {epoch + 1}/{EPOCHS}")
        
        model.train()
        total_loss = 0
        
        pbar = tqdm(train_dl, desc='Training', leave=False)
        for batch in pbar:
            ids = batch['input_ids'].to(device, non_blocking=True)
            mask = batch['attention_mask'].to(device, non_blocking=True)
            labels = batch['labels'].to(device, non_blocking=True)
            
            with autocast(enabled=USE_AMP, dtype=torch.float16):
                loss, _ = model(ids, mask, labels)
            
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), MAX_GRAD_NORM)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)  # Faster
            
            total_loss += loss.item()
            pbar.set_postfix({'loss': f'{loss.item():.4f}'})
        
        avg_loss = total_loss / len(train_dl)
        
        model.eval()
        val_preds, val_targets = [], []
        
        with torch.no_grad():
            for batch in tqdm(val_dl, desc='Validation', leave=False):
                ids = batch['input_ids'].to(device, non_blocking=True)
                mask = batch['attention_mask'].to(device, non_blocking=True)
                
                with autocast(enabled=USE_AMP, dtype=torch.float16):
                    preds = model(ids, mask)
                
                val_preds.extend(preds.cpu().numpy())
                val_targets.extend(batch['labels'].numpy())
        
        val_preds_orig = proc.inverse_transform(val_preds)
        val_targets_orig = proc.inverse_transform(val_targets)
        val_smape = calculate_smape(val_targets_orig, val_preds_orig)
        val_mae = mean_absolute_error(val_targets_orig, val_preds_orig)
        
        print(f"Loss: {avg_loss:.4f} | SMAPE: {val_smape:.4f}% | MAE: {val_mae:.2f}")
        
        tracker.update(epoch + 1, val_smape, avg_loss, model, optimizer, scheduler)
        clear_gpu_mem()
        
        if tracker.should_stop():
            print(f"Early stop at epoch {epoch + 1}")
            break
    
    print(f"TRAINING DONE! Best SMAPE: {tracker.best_smape:.4f}%")
    
    os.makedirs(MODEL_DIR, exist_ok=True)
    torch.save(model.state_dict(), os.path.join(MODEL_DIR, 'final.pth'))
    tokenizer.save_pretrained(MODEL_DIR)
    torch.save(proc, os.path.join(MODEL_DIR, 'proc.pth'))
    
    return tokenizer, model, proc, test_df

def predict(model, proc, tokenizer, test_df):
    
    model.eval()
    device = next(model.parameters()).device
    
    test_ds = amazonDataset(
        test_df['catalog_content'].values,
        tokenizer=tokenizer,
        max_length=MAX_LENGTH
    )
    
    test_dl = DataLoader(
        test_ds,
        batch_size=BATCH_SIZE * 2,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
        collate_fn=ultra_collate_fn
    )
    
    preds = []
    
    with torch.no_grad():
        for batch in tqdm(test_dl, desc='Predicting'):
            ids = batch['input_ids'].to(device, non_blocking=True)
            mask = batch['attention_mask'].to(device, non_blocking=True)
            
            with autocast(enabled=USE_AMP, dtype=torch.float16):
                batch_preds = model(ids, mask)
            
            preds.extend(batch_preds.cpu().numpy())
    
    final_preds = proc.inverse_transform(preds)
    
    out_df = pd.DataFrame({
        'sample_id': test_df['sample_id'].values,
        'price': final_preds
    })
    
    out_df.to_csv(OUTPUT_FILE, index=False)
    
    print(f"Mean: ${out_df['price'].mean():.2f} | Std: ${out_df['price'].std():.2f}")
    
    return out_df

def load_best():
    ckpt = torch.load(os.path.join(CHECKPOINT_DIR, 'best.pth'))
    proc = torch.load(os.path.join(MODEL_DIR, 'proc.pth'))
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    
    model = RegressorFunction(MODEL_NAME)
    model.load_state_dict(ckpt['model'])
    model.to(torch.device('cuda:0'))
    model.eval()
    
    print(f"Loaded checkpoint (Epoch {ckpt['epoch']}, SMAPE: {ckpt['smape']:.4f}%)")
    return model, proc, tokenizer

if __name__ == "__main__":
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    os.makedirs(MODEL_DIR, exist_ok=True)
    
    try:
        if torch.cuda.is_available():
            print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f}GB")
        
        if not os.path.exists(os.path.join(CHECKPOINT_DIR, 'best.pth')):
            tokenizer, model, proc, test_df = train()
        else:
            model, proc, tokenizer = load_best()
            test_df = pd.read_csv(TEST_FILE)
            _, test_df = proc.preprocess(pd.DataFrame(), test_df)
        
        predict(model, proc, tokenizer, test_df)
        
        
    except Exception as e:
        import traceback
        traceback.print_exc()
