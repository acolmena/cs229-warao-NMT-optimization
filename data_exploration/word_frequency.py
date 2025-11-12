import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from collections import Counter
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords



# load parallel data
df_train = pd.read_csv("./data/parallel_train.csv")
df_valid = pd.read_csv("./data/parallel_val.csv")
df_test = pd.read_csv("./data/parallel_test.csv")

df_all = pd.concat([df_train, df_valid, df_test])

print('\n' + '=' * 60)
print(f"Data overview: \n")
print(f"* Number of sentence translations: {len(df_all)}")
print(f" \n * Data preview: {df_all.head(10)}")
print('=' * 60)

# Start data exploration
# 1. General Word Frequency 
word_dict = {
    'Warao': {'all_words': None, 'words': None, 'freqs': None, 'counts': None},
    'Spanish': {'all_words': None, 'words': None, 'freqs': None, 'counts': None},
}

for col in list(df_all.columns): 
    language = col.split("_")[0].capitalize()
    sentences = list(df_all[col])
    all_words = []
    for sentence in sentences: 
        words = sentence.split()
        for wrd in words:
            wrd = wrd.strip().lower()
            wrd = ''.join(ch for ch in wrd if ch.isalpha())  # use isalpha cus it handles any type of letter char even accented ones
            all_words.append(wrd)

    word_dict[language]['all_words'] = all_words
    print('\n' + '=' * 60)
    print(f"Preview of {language} words: \n")
    print(all_words[:30])
    print(f"\nTotal number of words: {len(all_words)}")
    print('=' * 60)
    counts = Counter(all_words)
    word_dict[language]['counts'] = counts
    most_common_words = counts.most_common(30) 
    word_dict[language]['words'], word_dict[language]['freqs'] = zip(*most_common_words)


# Warao plot
plt.figure(1, figsize=(10, 6))
plt.bar(word_dict['Warao']['words'], word_dict['Warao']['freqs'])
plt.xlabel("Words")
plt.ylabel("Frequency")
plt.title("Top 30 Most Frequent Warao Words")
plt.xticks(rotation=45, ha='right')
plt.tight_layout()

# Spanish plot
plt.figure(2, figsize=(10, 6))
plt.bar(word_dict['Spanish']['words'], word_dict['Spanish']['freqs'])
plt.xlabel("Words")
plt.ylabel("Frequency")
plt.title("Top 30 Most Frequent Spanish Words")
plt.xticks(rotation=45, ha='right')
plt.tight_layout()



# 2. Word Frequency Without Stopwords: What are the core words used frequently
warao_stopwords = ["ine", "iji", "tai", "oko", "yatu", "tatuma"]  # English translation "I, you, he/she, us, you (plural, so 'yall'), they"
add_stopwords = ["ustedes", "entonces", "sino", "pues", "si", "así", "después", "ser", "mismo"]
spanish_stopwords = stopwords.words('spanish') + add_stopwords
# print(spanish_stopwords[:200], len(spanish_stopwords), [0 for word in add_stopwords if word not in list(spanish_stopwords)])

no_stop_warao_words = [wrd for wrd in word_dict['Warao']['all_words'] if wrd not in warao_stopwords]
no_stop_spanish_wrds = [wrd for wrd in word_dict['Spanish']['all_words'] if wrd not in spanish_stopwords]

counts_warao = Counter(no_stop_warao_words)
warao_most_common_words = counts_warao.most_common(30) 
ns_warao_wrds, ns_warao_freqs = zip(*warao_most_common_words)

counts_spani = Counter(no_stop_spanish_wrds)
spani_most_common_words = counts_spani.most_common(30) 
ns_spani_wrds, ns_spani_freqs = zip(*spani_most_common_words)

# Warao plot
plt.figure(3, figsize=(10, 6))
plt.bar(ns_warao_wrds, ns_warao_freqs, color='red')
plt.xlabel("Words")
plt.ylabel("Frequency")
plt.title("Top 30 Most Frequent Warao Words")
plt.xticks(rotation=45, ha='right')
plt.tight_layout()

# Spanish plot
plt.figure(4, figsize=(10, 6))
plt.bar(ns_spani_wrds, ns_spani_freqs, color='red')
plt.xlabel("Words")
plt.ylabel("Frequency")
plt.title("Top 30 Most Frequent Spanish Words")
plt.xticks(rotation=45, ha='right')
plt.tight_layout()


# 3. Top 50 Least Frequent Words: What does the other extreme of word frequncy look like?
counts_warao = word_dict['Warao']['counts']
warao_least_common_words = counts_warao.most_common()[::-1][:50]   # last part reverses the result
ns_warao_wrds, ns_warao_freqs = zip(*warao_least_common_words)

counts_spani = word_dict['Spanish']['counts']
spani_least_common_words = counts_spani.most_common()[::-1][:50]  
ns_spani_wrds, ns_spani_freqs = zip(*spani_least_common_words)

# Warao plot
plt.figure(5, figsize=(10, 6))
plt.bar(ns_warao_wrds, ns_warao_freqs, color='green')
plt.xlabel("Words")
plt.ylabel("Frequency")
plt.title("Top 30 Least Frequent Warao Words")
plt.xticks(rotation=45, ha='right')
plt.tight_layout()

# Spanish plot
plt.figure(6, figsize=(10, 6))
plt.bar(ns_spani_wrds, ns_spani_freqs, color='green')
plt.xlabel("Words")
plt.ylabel("Frequency")
plt.title("Top 30 Least Frequent Spanish Words")
plt.xticks(rotation=45, ha='right')
plt.tight_layout()



# 4. Sentence Length Comparisons: Do Warao or Spanish sentences tend to be more verbose?




plt.show()