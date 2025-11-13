from collections import Counter

def dict_builder(df_all):
    word_dict = {
    'Warao': {'all_words': None, 'words': None, 'freqs': None, 'counts': None},
    'Spanish': {'all_words': None, 'words': None, 'freqs': None, 'counts': None},
    }

    for col in list(df_all.columns): 
        language = col.split("_")[0].capitalize()
        sentences = list(df_all[col])
        all_words = []
        for sentence in sentences: 
            if isinstance(sentence, str):
                words = sentence.split()
            else:
                continue 
            for wrd in words:
                wrd = wrd.strip().lower()
                wrd = ''.join(ch for ch in wrd if ch.isalpha())  # use isalpha cus it handles any type of letter char even accented ones
                if wrd == '': continue
                all_words.append(wrd)

        word_dict[language]['all_words'] = all_words
        print(f"\nPreview of {language} words: \n")
        print(all_words[:30])
        print(f"\nTotal number of words: {len(all_words)}")
        counts = Counter(all_words)
        word_dict[language]['counts'] = counts
        most_common_words = counts.most_common(30) 
        word_dict[language]['words'], word_dict[language]['freqs'] = zip(*most_common_words)
    return word_dict