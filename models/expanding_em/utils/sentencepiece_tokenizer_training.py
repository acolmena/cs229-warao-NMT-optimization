"""
Train SentencePiece model for Warao monolingual tokenizer
"""

import sentencepiece as spm
from sentencepiece import sentencepiece_model_pb2 as sp_pb2_model


DATA_PATH = "./input/clean_monolingual_warao_train.txt"  
SPM_PREFIX = "warao_sentencepiece"                                                          


with open(DATA_PATH, "r", encoding="utf-8") as f:
    all_texts = [line.strip() for line in f if line.strip()]

required_chars = "áéíóú"  

spm.SentencePieceTrainer.train(
    input=DATA_PATH,
    model_prefix=SPM_PREFIX,
    vocab_size=8000,  # to fix error: Vocabulary size too high (16384). Please set it to a value <= 8980.
    character_coverage = 1,
    num_threads=16,
    train_extremely_large_corpus=False,
    add_dummy_prefix=False,
    max_sentencepiece_length=128,
    max_sentence_length=4192*4,
    pad_id=0,
    eos_id=1,
    unk_id=2,
    bos_id=-1,
    required_chars=required_chars,
)

print(f"SPM saved: {SPM_PREFIX}.model & {SPM_PREFIX}.vocab")


sp_trained = spm.SentencePieceProcessor(model_file=f"{SPM_PREFIX}.model")

added_spm = sp_pb2_model.ModelProto()
added_spm.ParseFromString(sp_trained.serialized_model_proto())

print("First 20 vocab pieces:", [p.piece for p in added_spm.pieces[:20]])
print("Vocab size:", len(added_spm.pieces))
