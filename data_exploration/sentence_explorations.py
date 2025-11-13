import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from collections import Counter
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
from sentence_dict_builder import dict_builder



# load parallel data
df_train = pd.read_csv("./data/parallel_train.csv")
df_valid = pd.read_csv("./data/parallel_val.csv")
df_test = pd.read_csv("./data/parallel_test.csv")

df_all = pd.concat([df_train, df_valid, df_test])

print('\n' + '=' * 60)
print(f"Data overview: \n")
print(f"* Number of sentence translations: {len(df_all)}")
print(f"{df_all.head(10)}")
print('=' * 60)

# ---------------------------------------------------------------------------------

# Start data exploration
# 0. Check for NaNs & duplicates
print('\n' + '=' * 60)
print("0) Checking for NaN values & duplicates. . .")
if df_all.isna().any().any():
    df_all.isna().sum()
else:
    print("Our dataset has no NaN values, awesome!!")

print("\n Duplicates: ")
df_all.duplicated().sum()
print('=' * 60)


# ---------------------------------------------------------------------------------


# 1. General Word Frequency 
print('\n' + '=' * 60)
print("1) General Word Frequency Analysis . . .")

word_dict = dict_builder(df_all)

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

print('=' * 60)


# ---------------------------------------------------------------------------------


# 2. Word Frequency Without Stopwords: What are the core words used frequently
print('\n' + '=' * 60)
print("2) Word Frequency Without Stopwords Analysis . . .")
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

print(f"- Most Frequent Warao words w/out stopwords:")
print(pd.DataFrame(warao_most_common_words, columns=['word', 'frequency']).to_string(index=False))
print(f"- Most Frequent Spanish words w/out stopwords:")
print(pd.DataFrame(spani_most_common_words, columns=['word', 'frequency']).to_string(index=False))

# Warao plot
plt.figure(3, figsize=(10, 6))
plt.bar(ns_warao_wrds, ns_warao_freqs, color='red')
plt.xlabel("Words")
plt.ylabel("Frequency")
plt.title("Top 30 Most Frequent Warao Words (No Stopwords)")
plt.xticks(rotation=45, ha='right')
plt.tight_layout()

# Spanish plot
plt.figure(4, figsize=(10, 6))
plt.bar(ns_spani_wrds, ns_spani_freqs, color='red')
plt.xlabel("Words")
plt.ylabel("Frequency")
plt.title("Top 30 Most Frequent Spanish Words (No Stopwords)")
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
print('=' * 60)


# ---------------------------------------------------------------------------------


# 3. Top 30 Least Frequent Words: What does the other extreme of word frequncy look like?
print('\n' + '=' * 60)
top_n = 30
print(f"3) Top {top_n} Least Frequent Words Analysis . . .")
counts_warao = word_dict['Warao']['counts']
warao_least_common_words = counts_warao.most_common()[::-1][:top_n]   # last part reverses the result
wa_least_common_wrds, wa_least_common_freqs = zip(*warao_least_common_words)

counts_spani = word_dict['Spanish']['counts']
spani_least_common_words = counts_spani.most_common()[::-1][:top_n]  
sp_least_common_wrds, sp_least_common_freqs = zip(*spani_least_common_words)

print(f"- Top {top_n} Least Frequent Warao words:")
print(pd.DataFrame(warao_least_common_words))
print(f"- Top {top_n} Least Frequent Spanish Words:")
print(pd.DataFrame(spani_least_common_words))

# Warao plot -- this and plot below were adapted from matplotlib documentation: https://matplotlib.org/stable/gallery/lines_bars_and_markers/bar_label_demo.html
fig, ax = plt.subplots()
hbars = ax.barh(wa_least_common_wrds, wa_least_common_freqs, align='center')
ax.set_yticks(wa_least_common_wrds, labels=wa_least_common_wrds)
ax.invert_yaxis()  # labels read top-to-bottom
ax.set_xlabel('Frequency')
ax.set_ylabel('Warao Words')
ax.set_title(f'Top {top_n} Least Frequent Warao Words')

# Label with specially formatted floats
ax.bar_label(hbars)
ax.set_xlim(right=1.5)  # adjust xlim to fit labels

# Spanish plot
fig, ax = plt.subplots()
hbars = ax.barh(sp_least_common_wrds, sp_least_common_freqs, align='center')
ax.set_yticks(sp_least_common_wrds, labels=sp_least_common_wrds)
ax.invert_yaxis()  # labels read top-to-bottom
ax.set_xlabel('Frequency')
ax.set_ylabel('Spanish Words')
ax.set_title(f'Top {top_n} Least Frequent Warao Words')

# Label with specially formatted floats
ax.bar_label(hbars)
ax.set_xlim(right=1.5)  # adjust xlim to fit labels



# ---------------------------------------------------------------------------------


# 4. Sentence Length Comparisons: Do Warao or Spanish sentences tend to be more verbose?
print('\n' + '=' * 60)
top_n = 60
print(f"4) Sentence Length Comparisons: Do Warao or Spanish sentences tend to be more verbose?")
sentences_lst = list(df_all.to_records(index=False)) # generates a list of tuples
spanish_more_verbose = 0
warao_more_verbose = 0
for sentence in sentences_lst:
    warao_sent = sentence[0]
    spanish_sent = sentence[1]
    if len(warao_sent.split()) > len(spanish_sent.split()):
        warao_more_verbose += 1
    else: 
        spanish_more_verbose += 1

print(f"Number of times when Warao sentences were more verbose: {warao_more_verbose} \n")
print(f"Number of times when Spanish sentences were more verbose: {spanish_more_verbose} \n")
print(f"{'Spanish' if spanish_more_verbose > warao_more_verbose else 'Warao'} sentences are more verbose.\n")
print('=' * 60)


# ----------------------------------------------------------------------------------------


# 5. Sentence Length Variation: How much do Warao and Spanish sentences vary in length?
print('\n' + '=' * 60)
print(f"5) Sentence Length Variation: How much do Warao and Spanish sentences vary in length?")
variations = []
for sentence in sentences_lst:
    warao_sent = sentence[0]
    spanish_sent = sentence[1]
    diff = len(spanish_sent.split()) - len(warao_sent.split())
    variations.append(diff)

plt.figure(7, figsize=(10, 6))
plt.hist(variations, bins=50, color='skyblue', edgecolor='black')
plt.title('Distribution of Sentence Length Differences (Spanish Sent. Length − Warao Sent. Length)')
plt.xlabel('Length Difference (# of words)')
plt.ylabel('Frequency')
plt.grid(True, linestyle='--', alpha=0.6)

plt.figure(8, figsize=(10, 6))
plt.boxplot(variations, vert=False)
plt.title('Distribution of Sentence Length Differences (Spanish Sent. Length − Warao Sent. Length)')
plt.xlabel('Length Difference (# of words)')
plt.grid(True, axis='y', linestyle='--', alpha=0.6)

print(f"Outliers in length variations: {sorted(variations)[-10:]} \n")
print('=' * 60)




plt.show()