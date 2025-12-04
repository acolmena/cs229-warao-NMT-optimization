from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch
from evaluate import load
from datasets import load_dataset
import pandas as pd
from tqdm import tqdm

FROM_FILE = True
TEST_ON_TEST = True
BATCH_SIZE = 32
MAX_LEN = 128
NUM_BEAMS = 4



model_configs = {
    "facebook/m2m100_418M": ("pt", "es"),
    "facebook/mbart-large-50": ("pt_XX", "es_XX"),
    "facebook/mbart-large-50-one-to-many-mmt": ("pt_XX", "es_XX"),
    "facebook/nllb-200-distilled-600M": ("wro_Latn", "spa_Latn"),
    "google/mt5-base": ("Portuguese", "Spanish"),
    "google/mt5-small": ("Portuguese", "Spanish"),
    "google/byt5-base": ("Portuguese", "Spanish"),
    "google/byt5-small": ("Portuguese", "Spanish"),
}

MODEL_NAME = list(model_configs.keys())[2]
MODEL = MODEL_NAME.split('/')[1]
OUTPUT_DIR = 'best-' + MODEL
if FROM_FILE:
    model_dir = "/Users/anabelle/Desktop/cs229-warao-NMT-optimization/models/input/mbart-large-50-many-to-many-mmt-finetuned-warao-es_sweep_6j1apmrg"
else:
    model_dir = MODEL_NAME

tokenizer = AutoTokenizer.from_pretrained(model_dir, local_files_only=FROM_FILE)
model = AutoModelForSeq2SeqLM.from_pretrained(model_dir, local_files_only=FROM_FILE)
print(model.eval())

device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)
model.eval()


bleu = load("sacrebleu")
chrf = load("chrf")


if TEST_ON_TEST:
    dataset = load_dataset("csv", data_files={'test': "/Users/anabelle/Desktop/cs229-warao-NMT-optimization/models/input/final_parallel_test.csv"})['test']
else:
    dataset = load_dataset("csv", data_files={'validation': "/Users/anabelle/Desktop/cs229-warao-NMT-optimization/models/input/final_parallel_val.csv"})['validation']

source_sentences = [ex['warao_sentence'] for ex in dataset]
references_full = [ex['spanish_sentence'] for ex in dataset]


src_lang, tgt_lang = model_configs[MODEL_NAME] 


def batch_data(data, batch_size):
    for i in range(0, len(data), batch_size):
        yield data[i:i+batch_size]

def prepare_batch(batch_texts, tokenizer, src_lang, tgt_lang, model_name, max_length=128):

    if "m2m100" in model_name:
        tokenizer.src_lang = src_lang
        inputs = tokenizer(batch_texts, return_tensors="pt", padding=True,
                          truncation=True, max_length=max_length).to(device)
        forced_bos_token_id = tokenizer.get_lang_id(tgt_lang)

    elif "mbart" in model_name:
        tokenizer.src_lang = src_lang
        inputs = tokenizer(batch_texts, return_tensors="pt", padding=True,
                          truncation=True, max_length=max_length).to(device)
        forced_bos_token_id = tokenizer.lang_code_to_id[tgt_lang]

    elif "nllb" in model_name:
        tokenizer.src_lang = src_lang
        inputs = tokenizer(batch_texts, return_tensors="pt", padding=True,
                          truncation=True, max_length=max_length).to(device)
        forced_bos_token_id = tokenizer.convert_tokens_to_ids(tgt_lang)

    elif "mt5" in model_name or "byt5" in model_name:
        batch_with_prefix = [f"translate {src_lang} to {tgt_lang}: {text}"
                            for text in batch_texts]
        inputs = tokenizer(batch_with_prefix, return_tensors="pt", padding=True,
                          truncation=True, max_length=max_length).to(device)
        forced_bos_token_id = None

    else:
        raise ValueError(f"Unknown model type: {model_name}")

    return inputs, forced_bos_token_id



predictions = []
references = []

for batch in tqdm(batch_data(dataset, BATCH_SIZE), total=(len(dataset)+BATCH_SIZE-1)//BATCH_SIZE, desc="Translating"):
    # print(batch)
    batch_texts = [ex for ex in batch["warao_sentence"]]
    batch_refs = [ex for ex in batch["spanish_sentence"]]

    inputs, forced_bos_token_id = prepare_batch(batch_texts, tokenizer, src_lang, tgt_lang, MODEL_NAME, MAX_LEN)

    with torch.no_grad():
        if forced_bos_token_id is not None:
            outputs = model.generate(
                **inputs,
                max_length=MAX_LEN,
                num_beams=NUM_BEAMS,
                forced_bos_token_id=forced_bos_token_id
            )
        else:
            outputs = model.generate(
                **inputs,
                max_length=MAX_LEN,
                num_beams=NUM_BEAMS
            )

    batch_preds = tokenizer.batch_decode(outputs, skip_special_tokens=True)

    predictions.extend(batch_preds)
    references.extend(batch_refs)

    if len(predictions) % 40 == 0:
        print('-' * 60)
        print(batch_preds[0])
        print(batch_refs[0])
        print('-' * 60)
        print(f"Processed {len(predictions)} examples...")


 
# Save predictions
df = pd.DataFrame({
    "predicted_spanish": predictions,
    "reference_spanish": references
})
df.to_csv(f"{OUTPUT_DIR}-predictions.csv", index=False)
print("\nSaved:", f"{OUTPUT_DIR}-predictions.csv")



# Compute metrics: BLEU and chrF
bleu_score = bleu.compute(predictions=predictions, references=[[r] for r in references])
print(bleu_score)
print(f"\nBLEU Score: {bleu_score['score']:.4f}")

chrf_score = chrf.compute(predictions=predictions, references=[[r] for r in references])
print(chrf_score)
print(f"chrF Score: {chrf_score['score']:.4f}")

print("\nDetailed BLEU scores:")
print(f"BLEU-1: {bleu_score['precisions'][0]:.4f}")
print(f"BLEU-2: {bleu_score['precisions'][1]:.4f}")
print(f"BLEU-3: {bleu_score['precisions'][2]:.4f}")
print(f"BLEU-4: {bleu_score['precisions'][3]:.4f}")


