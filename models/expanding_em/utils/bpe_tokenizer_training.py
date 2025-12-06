# from tokenizers import ByteLevelBPETokenizer
from tokenizers.trainers import BpeTrainer
from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.pre_tokenizers import Whitespace
from transformers import AutoTokenizer, PreTrainedTokenizerFast
from datasets import load_dataset
import pandas as pd
import random

random.seed(42)

# FOR MONOLINGUAL DATA
DATA_PATH = "./input/clean_monolingual_warao_train.txt"
OUTPUT_TOK_PATH = './output/warao_tokenizer_monolingual'


# FOR PARALLEL DATA
# DATA_PATH = "./input/final_parallel_train.csv"
# OUTPUT_TOK_PATH = './output/warao_tokenizer_parallel'



""" 
Train a tokenizer on Warao training data. 

To find useful token candidates to expand 
embeddin matrices of pre-trained models. 

Training code adapted from:
https://github.com/huggingface/tokenizers
"""

def generate_toks(example_text, tokenizer):
    for example in example_text:
        tokens = tokenizer.tokenize(example)
        print("-" * 30)
        print(f"\nText: {example}")
        print(f"\nTokenized text: {tokens}")
        print("-" * 30)


def compute_token_stats(tokens):
    series_toks = pd.Series(list(tokens))
    print("Learned token length stats:")
    print(series_toks.apply(len).describe())

def test_old_tokenizer(model_name=None, example_text=None):
    old_tokenizer = AutoTokenizer.from_pretrained(model_name)
    print(f'Special tokens from old tokenizer: {old_tokenizer.special_tokens_map}\n')
    # output old tokens:
    print("Old tokenizer tokens:")
    generate_toks(example_text, old_tokenizer)
    print(old_tokenizer)


def train_tokenizer(model_name=None, training_data_path=None, output_tok_path=None, example_text=None):
    
    # new_tokenizer = old_tokenizer.train_new_from_iterator(warao_text, 10000) <--- WHATS THE DIFFERENCE BTWN THIS

    # !! train BPE tokenizer to get useful tokens!!
    new_tokenizer = Tokenizer(BPE())     # <---- AND THESE LINES (WHATS THE DIFFERENCE BWN TRAINING A TOKENIZER FROM ANOTHER TOKENIZER VS THIS?)
    new_tokenizer.pre_tokenizer = Whitespace()
    # special_tokens={'bos_token': '<s>', 'eos_token': '</s>', 'unk_token': '<unk>', 'sep_token': '</s>', 'pad_token': '<pad>', <-- from other models
    trainer = BpeTrainer(special_tokens=['<s>', '</s>', '<unk>', "</s>", '<pad>'])
    new_tokenizer.train(files=[training_data_path], trainer=trainer)

    print(f"Total learned tokens: {new_tokenizer.get_vocab_size()}")

    learned_tokens = list(new_tokenizer.get_vocab().keys())
    print(random.sample(learned_tokens, 50))
    # breakpoint()
    

    # compute statistics on learned tokens
    compute_token_stats(learned_tokens)

    candidate_tokens = [
        token for token in learned_tokens 
        if not token.startswith("<")
    ]

    # preview candidate tokens
    if len(candidate_tokens) != len(learned_tokens):
        print("Preview of candidate tokens: {candidate_tokens[:50]}")
        # compute statistics on learned tokens
        compute_token_stats(candidate_tokens)
    else:
        print("No tokens removed.")
    
    # output new tokens on same text:
    print("New tokenizer tokens:")

    # save new tokenizer (raw)
    new_tokenizer.save(f'{output_tok_path}.json')

    # save tokenizer in HF style 
    hf_tokenizer = PreTrainedTokenizerFast(
        tokenizer_file=f'{output_tok_path}.json',
        unk_token='<unk>',
        pad_token='<pad>',
        sep_token='</s>',
        mask_token='<mask>'
    )
    hf_tokenizer.save_pretrained(output_tok_path)

    print("Tokenizer saved (raw and HF-style)! YAY!")





if __name__ == "__main__":
    model_name = "openai-community/gpt2"
    example_text = ["Yatu karata teribuya", # "you guys study"
                    "Tatuma kotubukunarai", # "let them play"
                    "Ma ijoro nisakitía",   # "i will remove my molar"
                    "Ya araisamaya Pablo kaiamo naruae Jacobo mikitane. Ikeresia airamo kokotuka tata ja.,"] # "Al día siguiente Pablo fue con nosotros a ver a Jacobo (Santiago, hermano de Jesús), y todos los ancianos estaban presentes."" 
    

    # output_tok_path = './output/warao_tokenizer_monolingual.json'
    # monolingual_data_path = "./input/monolingual_warao_bible.txt"

    test_old_tokenizer(model_name, example_text)
    train_tokenizer(model_name, DATA_PATH, OUTPUT_TOK_PATH, example_text)
