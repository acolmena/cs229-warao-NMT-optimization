import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from collections import Counter
from sentence_dict_builder import dict_builder

# load parallel data
df_train = pd.read_csv("./data/parallel_train.csv")
df_valid = pd.read_csv("./data/parallel_val.csv")
df_test = pd.read_csv("./data/parallel_test.csv")
df_dict = pd.read_csv("./data/warao_dictionary_final.csv")

df_dict.columns = df_train.columns
df_all = pd.concat([df_train, df_valid, df_test, df_dict], axis=0)

print(df_all.columns)
assert len(df_all) == len(df_train) + len(df_valid) + len(df_test) + len(df_dict)


word_dict = dict_builder(df_all)


# 1. Vocabulary insights: what's our Warao vs. Spanish vocabulary?
print('\n' + '=' * 60)
print(f"1) Vocabulary insights: what's our Warao vs. Spanish vocabulary?")
uniq_warao_words = set(word_dict['Warao']['all_words'])
uniq_spanish_words = set(word_dict['Spanish']['all_words'])

print("Number of unique Warao words: ", len(uniq_warao_words))
print("\nPreview of Warao vocab:")
print(list(uniq_warao_words)[:10])
print("Number of unique Spanish words: ", len(uniq_spanish_words))
print("\nPreview of Spanish vocab:")
print(list(uniq_spanish_words)[:10])

# plt.figure(9, figsize=(10, 6))
fig, ax = plt.subplots()
# error = np.random.rand(len(people))
y = ['Warao', 'Spanish']
ax.barh(y, [len(uniq_warao_words), len(uniq_spanish_words)], align='center')
ax.set_yticks(y, labels=y)
ax.invert_yaxis()  # labels read top-to-bottom
ax.set_xlabel('# Unique Words')
ax.set_title('Vocabulary Size per Language')
print('=' * 60)


# ----------------------------------------------------------------------------------------


# 2. Morphological richness: individual word length distributons
print('\n' + '=' * 60)
print(f"2) Morphological richness: individual word length distributons")
warao_word_lengths = [len(w) for w in uniq_warao_words]
spanish_word_lengths = [len(w) for w in uniq_spanish_words]


fig, ax = plt.subplots(sharey=True, tight_layout=True)
ax.boxplot([warao_word_lengths, spanish_word_lengths], sym='rs', orientation='horizontal', tick_labels=['Warao', 'Spanish'])
# ax.boxplot(spanish_word_lengths, sym='rs', orientation='horizontal')
ax.set_xlabel('Word Length (# of letters)')
ax.set_ylabel('Languages')
ax.set_title('Distribution of Individual Word Lengths')

print('=' * 60)

# visualizing the longest words in each language


# ----------------------------------------------------------------------------------------


# 3. Morphological richness: Type Token Ratio (TTR) -- # unique words / # total words
print('\n' + '=' * 60)
print(f"3) Type Token Ratio (TTR)")
print(f"Warao TTR: {len(uniq_warao_words) / len(word_dict['Warao']['all_words'])}")
print(f"Spanish TTR: {len(uniq_spanish_words) / len(word_dict['Spanish']['all_words'])}")

print('=' * 60)


# export Warao vocabulary
df_warao_vocab = pd.DataFrame(uniq_warao_words, columns=['warao_words']).to_csv('./output/warao_vocab.csv', index=False)

plt.show()