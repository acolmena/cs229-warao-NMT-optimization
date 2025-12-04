# -*- coding: utf-8 -*-
"""
M2M-100 PyTorch Training Loop adapted from Hugging Face Trainer API 
and David Dale's implementation in: https://cointegrated.medium.com/how-to-fine-tune-a-nllb-200-model-for-translating-a-new-language-a37fc706b865

"""

import os
import sys
import logging
import numpy as np
import pandas as pd
import json
import random
import gc
import torch
from tqdm import trange
from datasets import load_dataset
from evaluate import load
from transformers import (
    AutoConfig,
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    set_seed,
)
from transformers.optimization import Adafactor
from transformers import get_constant_schedule_with_warmup

random.seed(42)

# Configuration
MODEL_NAME = "facebook/m2m100_418M"
MODEL = MODEL_NAME.split('/')[1]
TRAIN_FILE = "./input/parallel_train.csv"
VAL_FILE = "./input/parallel_val.csv"
TEST_FILE = "./input/parallel_test.csv"
OUTPUT_DIR = f"./{MODEL}-finetuned-warao-es"
SOURCE_CODE = "pt"  # M2M-100 language code
TARGET_CODE = "es"
MAX_LEN = 128
EPOCHS = 3
BATCH_SIZE = 16
LEARNING_RATE = 1e-4
NUM_BEAMS = 4
TOKEN_FILE = './input/warao_tokenizer_monolingual.json'
VERSION = 'v2'
TRAINING_STEPS = 60000  # or calculate from epochs: len(train_data) // BATCH_SIZE * EPOCHS
SAVE_STEPS = 1000
LOG_STEPS = 100

device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
print("USING DEVICE:", device)


def cleanup():
    """Try to free GPU memory"""
    gc.collect()
    torch.cuda.empty_cache()


def load_new_tokens(token_file, version='v1'):
    """Load new tokens for vocabulary extension"""
    if version == 'v1':
        warao_vocab = pd.read_csv(token_file)
        warao_vocab_lst = list(warao_vocab['warao_words'])
        num_wrds = 200
        new_toks = random.sample(warao_vocab_lst, k=num_wrds)
    elif version == 'v2':
        with open(token_file, 'r', encoding='utf-8') as f:
            token_data = json.load(f)
        new_toks = list(token_data['model']['vocab'].keys())[95:]
    
    print(f"Preview vocab: {new_toks[:5]}")
    return new_toks


def get_batch_pairs(batch_size, data, tokenizer, source_lang, target_lang):
    """Get a batch of source-target pairs"""
    xx, yy = [], []
    
    # Sample random indices
    indices = [random.randint(0, len(data)-1) for _ in range(batch_size)]
    
    for idx in indices:
        item = data.iloc[idx]
        xx.append(item['warao_sentence'])
        yy.append(item['spanish_sentence'])
    
    return xx, yy, source_lang, target_lang


def train_pytorch_loop(model_name_or_path, train_file, val_file, test_file, output_dir,
                      source_lang, target_lang, max_length=128,
                      batch_size=16, learning_rate=1e-4, num_beams=4,
                      training_steps=60000, save_steps=1000, log_steps=100,
                      token_file=None, version='v2'):
    
    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)
    
    # Load datasets
    print("\n" + "=" * 50)
    logger.info("Loading datasets...")
    print("=" * 50)
    
    df_train = pd.read_csv(train_file)
    df_val = pd.read_csv(val_file)
    df_test = pd.read_csv(test_file)
    
    print(f"Train size: {len(df_train)}")
    print(f"Val size: {len(df_val)}")
    print(f"Test size: {len(df_test)}")
    
    # Load tokenizer
    print("\n" + "=" * 50)
    print(f"Loading {model_name_or_path} model and tokenizer...")
    print("=" * 50)
    
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
    
    # Load and extend vocabulary
    if token_file:
        new_toks = load_new_tokens(token_file=token_file, version=version)
        num_added = tokenizer.add_tokens(new_toks)
        print(f"Added tokens: {num_added}")
    
    # Load model
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name_or_path)
    
    # Resize embeddings if tokens were added
    if token_file:
        model.resize_token_embeddings(len(tokenizer))
        
        # Initialize new embeddings
        tokens_added = len(new_toks)
        params = model.state_dict()
        embeddings = params['model.shared.weight']
        pre_expansion_embeddings = embeddings[:-tokens_added, :]
        mu = torch.mean(pre_expansion_embeddings, dim=0)
        n = pre_expansion_embeddings.size()[0]
        sigma = ((pre_expansion_embeddings - mu).T @ (pre_expansion_embeddings - mu)) / n
        dist = torch.distributions.multivariate_normal.MultivariateNormal(
            mu, covariance_matrix=1e-5*sigma)
        
        new_embeddings = torch.stack(tuple((dist.sample() for _ in range(tokens_added))), dim=0)
        embeddings[-tokens_added:, :] = new_embeddings
        params['model.shared.weight'][-tokens_added:, :] = new_embeddings
        model.load_state_dict(params)
    
    # Move model to device
    model.to(device)
    model.train()
    
    # Setup optimizer and scheduler
    optimizer = Adafactor(
        [p for p in model.parameters() if p.requires_grad],
        scale_parameter=False,
        relative_step=False,
        lr=learning_rate,
        clip_threshold=1.0,
        weight_decay=1e-3,
    )
    
    scheduler = get_constant_schedule_with_warmup(optimizer, num_warmup_steps=1000)
    
    # Training loop
    losses = []
    x, y, loss = None, None, None
    cleanup()
    
    print("\n" + "=" * 50)
    print("Starting training...")
    print("=" * 50)
    
    tq = trange(training_steps)
    for i in tq:
        xx, yy, lang1, lang2 = get_batch_pairs(
            batch_size, df_train, tokenizer, source_lang, target_lang
        )
        
        try:
            # Tokenize source
            tokenizer.src_lang = lang1
            x = tokenizer(
                xx, 
                return_tensors='pt', 
                padding=True, 
                truncation=True, 
                max_length=max_length
            ).to(device)
            
            # Tokenize target
            tokenizer.src_lang = lang2
            y = tokenizer(
                yy, 
                return_tensors='pt', 
                padding=True, 
                truncation=True, 
                max_length=max_length
            ).to(device)
            
            # Replace padding token id with -100 (ignored in loss)
            y.input_ids[y.input_ids == tokenizer.pad_token_id] = -100
            
            # Forward pass
            loss = model(**x, labels=y.input_ids).loss
            
            # Backward pass
            loss.backward()
            losses.append(loss.item())
            
            # Optimizer step
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            scheduler.step()
            
        except RuntimeError as e:
            # Handle OOM errors
            optimizer.zero_grad(set_to_none=True)
            x, y, loss = None, None, None
            cleanup()
            print(f'Error at step {i}, max length: {max(len(s) for s in xx + yy)}, {e}')
            continue
        
        # Logging
        if i % log_steps == 0 and i > 0:
            avg_loss = np.mean(losses[-log_steps:])
            tq.set_description(f'Step {i}, Avg Loss: {avg_loss:.4f}')
            print(f"Step {i}, Average Loss (last {log_steps} steps): {avg_loss:.4f}")
        
        # Saving
        if i % save_steps == 0 and i > 0:
            print(f"\nSaving model at step {i}...")
            os.makedirs(output_dir, exist_ok=True)
            model.save_pretrained(output_dir)
            tokenizer.save_pretrained(output_dir)
            print(f"Model saved to {output_dir}")
            
            # Optional: Evaluate on validation set
            if i % (save_steps * 2) == 0:
                eval_loss = evaluate_model(
                    model, tokenizer, df_val, 
                    source_lang, target_lang, max_length, device
                )
                print(f"Validation Loss: {eval_loss:.4f}")
                model.train()  # Back to training mode
    
    # Final save
    print("\nTraining completed! Saving final model...")
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    print(f"Final model saved to {output_dir}")
    
    return model, tokenizer, losses


def evaluate_model(model, tokenizer, data, source_lang, target_lang, max_length, device):
    """Evaluate model on validation set"""
    model.eval()
    eval_losses = []
    
    with torch.no_grad():
        for i in range(0, len(data), 16):  # Use batch size of 16 for eval
            batch_data = data.iloc[i:i+16]
            xx = batch_data['warao_sentence'].tolist()
            yy = batch_data['spanish_sentence'].tolist()
            
            try:
                tokenizer.src_lang = source_lang
                x = tokenizer(xx, return_tensors='pt', padding=True, 
                            truncation=True, max_length=max_length).to(device)
                
                tokenizer.src_lang = target_lang
                y = tokenizer(yy, return_tensors='pt', padding=True, 
                            truncation=True, max_length=max_length).to(device)
                
                y.input_ids[y.input_ids == tokenizer.pad_token_id] = -100
                
                loss = model(**x, labels=y.input_ids).loss
                eval_losses.append(loss.item())
            except RuntimeError:
                continue
    
    return np.mean(eval_losses) if eval_losses else float('inf')


def translate(text, model, tokenizer, src_lang, tgt_lang, 
              a=32, b=3, max_input_length=128, num_beams=4, **kwargs):
    """Translate text from source to target language"""
    tokenizer.src_lang = src_lang
    tokenizer.tgt_lang = tgt_lang
    
    inputs = tokenizer(
        text, 
        return_tensors='pt', 
        padding=True, 
        truncation=True, 
        max_length=max_input_length
    )
    
    model.eval()
    
    with torch.no_grad():
        result = model.generate(
            **inputs.to(model.device),
            forced_bos_token_id=tokenizer.convert_tokens_to_ids(tgt_lang),
            max_new_tokens=int(a + b * inputs.input_ids.shape[1]),
            num_beams=num_beams, 
            **kwargs
        )
    
    return tokenizer.batch_decode(result, skip_special_tokens=True)


if __name__ == "__main__":
    set_seed(42)
    
    model, tokenizer, losses = train_pytorch_loop(
        model_name_or_path=MODEL_NAME,
        train_file=TRAIN_FILE,
        val_file=VAL_FILE,
        test_file=TEST_FILE,
        output_dir=OUTPUT_DIR,
        source_lang=SOURCE_CODE,
        target_lang=TARGET_CODE,
        max_length=MAX_LEN,
        batch_size=BATCH_SIZE,
        learning_rate=LEARNING_RATE,
        num_beams=NUM_BEAMS,
        training_steps=TRAINING_STEPS,
        save_steps=SAVE_STEPS,
        log_steps=LOG_STEPS,
        token_file=TOKEN_FILE,
        version=VERSION,
    )
    
    # Test translation
    print("\n" + "=" * 50)
    print("Testing translation...")
    print("=" * 50)
    
    test_sentence = "Example Warao sentence"
    translation = translate(
        test_sentence, 
        model, 
        tokenizer, 
        SOURCE_CODE, 
        TARGET_CODE, 
        num_beams=NUM_BEAMS
    )
    print(f"Source: {test_sentence}")
    print(f"Translation: {translation[0]}")