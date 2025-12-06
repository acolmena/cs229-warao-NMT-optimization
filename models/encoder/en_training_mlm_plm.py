"""
Training a Warao encoder on a self-supervised objective on monolingual Warao.

We essentially want to train this encoder to understand Warao text. 

We will use Warao monolingual text and the Warao text from the parallel training set

Encoder produces embeddings. 
"""

from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForMaskedLM, DataCollatorForLanguageModeling, Trainer, TrainingArguments

TRAIN_FILE = "clean_monolingual_warao_train.txt"

MODEL_NAME = "FacebookAI/xlm-roberta-base"
OUTPUT_DIR = "./warao-encoder-mlm"
BATCH_SIZE = 16
EPOCHS = 5  
LEARNING_RATE = 5e-5
WEIGHT_DECAY = 0.01
SAVE_STEPS = 100
LOGGING_STEPS = 1 if "tiny" in TRAIN_FILE else 20

# load everything necessary
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForMaskedLM.from_pretrained(MODEL_NAME)
dataset = load_dataset("text", data_files={"train": TRAIN_FILE})["train"]


def tokenize(batch):
    return tokenizer(batch["text"], truncation=True, max_length=256)


dataset = dataset.map(tokenize, batched=True, remove_columns=["text"])


data_collator = DataCollatorForLanguageModeling(tokenizer, mlm=True, mlm_probability=0.15)

training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    per_device_train_batch_size=BATCH_SIZE,
    num_train_epochs=EPOCHS,
    learning_rate=LEARNING_RATE,
    weight_decay=WEIGHT_DECAY,
    save_strategy="no",
    # save_steps=,
    logging_steps=LOGGING_STEPS,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset,
    data_collator=data_collator,
)

trainer.train()
trainer.save_model("./warao-encoder-mlm")
