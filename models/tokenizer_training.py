# from tokenizers import ByteLevelBPETokenizer
from tokenizers.trainers import BpeTrainer
# from transformers import RobertaTokenizerFast
from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.pre_tokenizers import Whitespace
from transformers import AutoTokenizer
from datasets import load_dataset
import pandas as pd
import random

random.seed(42)


""" Train a tokenizer on Warao training data. To find useful token candidates
    to expand embeddin matrices of pre-trained models. 
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

def train_tokenizer(model_name=None, training_data_path=None, output_tok_path=None, example_text=None):
    with open(training_data_path, 'r', encoding='utf-8') as f:
        warao_text = f.read()

    old_tokenizer = AutoTokenizer.from_pretrained(model_name)
    # output old tokens:
    print("Old tokenizer tokens:")
    generate_toks(example_text, old_tokenizer)
    print(old_tokenizer)
    
    # new_tokenizer = old_tokenizer.train_new_from_iterator(warao_text, 10000) <--- WHATS THE DIFFERENCE BTWN THIS

    new_tokenizer = Tokenizer(BPE())     # <---- AND THESE LINES (WHATS THE DIFFERENCE BWN TRAINING A TOKENIZER FROM ANOTHER TOKENIZER VS THIS?)
    new_tokenizer.pre_tokenizer = Whitespace()
    trainer = BpeTrainer(special_tokens=["[UNK]", "[CLS]", "[SEP]", "[PAD]", "[MASK]"])
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
    
    breakpoint()
    # output new tokens on same text:
    print("New tokenizer tokens:")

    # save new tokenizer
    new_tokenizer.save_pretrained("warao_{model_name}_tokenizer")
    # tokenizer = Tokenizer(BPE())
    # tokenizer.pre_tokenizer = Whitespace()
    # trainer = BpeTrainer(special_tokens=["[UNK]", "[CLS]", "[SEP]", "[PAD]", "[MASK]"])
    # tokenizer.train(files=[training_data_path], trainer=trainer)
    # tokenizer.save_model(output_tok_path)

    # tokenizer = ByteLevelBPETokenizer()
    # tokenizer.train(files="input.txt", vocab_size=1000) # WHAT IS THIS VOCAB_SIZE PARAMETER




if __name__ == "__main__":
    model_name = "facebook/m2m100_418M"
    example_text = ["Yatu karata teribuya", # "you guys study"
                    "Tatuma kotubukunarai", # "let them play"
                    "Ma ijoro nisakitía",   # "i will remove my molar"
                    "Ya araisamaya Pablo kaiamo naruae Jacobo mikitane. Ikeresia airamo kokotuka tata ja.,"] # "Al día siguiente Pablo fue con nosotros a ver a Jacobo (Santiago, hermano de Jesús), y todos los ancianos estaban presentes."" 
    output_tok_path = 'warao_tokenizer_model'
    training_data_path = "./utils/input/parallel_train.txt"

    train_tokenizer(model_name, training_data_path, output_tok_path, example_text)