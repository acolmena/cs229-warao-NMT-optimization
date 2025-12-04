from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from datasets import load_dataset
import random
from tqdm.auto import tqdm, trange
import logging
import sys
import pandas as pd

"""
Different analyses on tokenizer to see how Warao text is being tokenized

Many of the analyses and functions are borrowed from David Dale
in his article: https://cointegrated.medium.com/how-to-fine-tune-a-nllb-200-model-for-translating-a-new-language-a37fc706b865

"""

random.seed(42)

def generate_tokens(tokenizer, dataset, col_name):
    all_tokens = []
    for example in dataset:
        sentence = example[col_name]
        tokens = tokenizer.tokenize(sentence)
        all_tokens += tokens
    return all_tokens

def unknown_tokens(tokenizer, dataset, col_name):
    texts_with_unk = [
        text for text in tqdm(dataset[col_name], desc="Checking for unknown tokens") 
        if tokenizer.unk_token_id in tokenizer(text).input_ids
    ]

    return texts_with_unk
    

def compute_stats(all_tokens, language):
    all_uniq_tokens = list(set(all_tokens))
    print(f"\nTotal unique tokens in {language} dataset: {len(all_uniq_tokens)}")
    print(f"\nPreview of unique tokens: {all_uniq_tokens[:5]}")
    series_tokens = pd.Series(all_tokens)
    series_tok_lens = series_tokens.apply(len)
    print(series_tokens.describe())
    print(series_tokens.value_counts())
    print(series_tok_lens.describe())



def tok_analysis(dataset=None, tokenizer=None, col_name=None, logger=None):
    """
    Analyze tokenizer of a given pre-trained LM on Warao dataset.
    We will use only our Warao training dataset
    
    Args:
        model_name (str): The name of the pre-trained,

    """
    language = col_name.split('_')[0].capitalize()

    logger.info(f"\nTokenizer analysis for {language} training sentences")
    logger.info(f"\nUsing model: {model_name}")

    all_tokens = generate_tokens(tokenizer, dataset, col_name)
    filename = f'{language.lower()}_tokens.csv'
    df_all_tokens = pd.DataFrame(all_tokens, columns=['token'])
    df_all_tokens.to_csv(filename, index=False)
    logger.info(f"Saved tokens to {filename}")
    

    # breakpoint()
    compute_stats(all_tokens, language)

    
    # determine how many unknown tokens we're getting 
    texts_with_unk = unknown_tokens(tokenizer, dataset, col_name)
    num_unk_toks = len(texts_with_unk)
    logger.info(f"\n{f"Number of {language} sentences with unknown tokens: {num_unk_toks}" if num_unk_toks > 0 else f"No unknown tokens generated for {language} sentences"}")

    return 




def analysis(model_name, dataset_path):
    """
    Run tokenizer analysis on Warao-Spanish parallel training dataset
    1. Token statistics
    2. Unknown tokens
    """

    # set up logger -- load only once
    logging.basicConfig(
      format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
      datefmt="%m/%d/%Y %H:%M:%S",
      handlers=[logging.StreamHandler(sys.stdout)],
    )
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)
    

    # load tokenizer -- load only once
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # load dataset -- load only once
    data_files = {'train': dataset_path}
    dataset = load_dataset('csv', data_files=data_files)['train']
    logger.info(f"\nLoaded dataset from {dataset_path}")
    print(f"\nDataset preview: {dataset[:2]}")


    warao_col = 'warao_sentence'
    spanish_col = 'spanish_sentence'
    # run analysis for Warao sentences
    tok_analysis(dataset, tokenizer, warao_col, logger)

    # run analysis for Spanish sentences
    tok_analysis(dataset, tokenizer, spanish_col, logger)




if __name__ == "__main__":
    model_name = "facebook/mBART-large-50"
    dataset_path = "./input/parallel_train.csv"

    analysis(model_name, dataset_path)