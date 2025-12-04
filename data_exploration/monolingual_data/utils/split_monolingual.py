import random

random.seed(42)

INPUT_FILE = './input/clean_monolingual_warao_final.txt'
TRAIN_PERCENTAGE = 0.9

with open(INPUT_FILE, "r", encoding="utf-8") as f:
    lines = [l.strip() for l in f if l.strip()]

random.shuffle(lines)

split = int(len(lines) * TRAIN_PERCENTAGE)
train_lines = lines[:split]
eval_lines  = lines[split:]

with open("clean_monolingual_warao_train.txt", "w", encoding="utf-8") as f:
    f.write("\n".join(train_lines))

with open("clean_monolingual_warao_eval.txt", "w", encoding="utf-8") as f:
    f.write("\n".join(eval_lines))
