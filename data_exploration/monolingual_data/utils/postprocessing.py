"""
Do some extra cleaning of monolingual Warao data
"""
import pandas as pd
import re


def process_txt_file(text):
    """
    1) Remove " --- Page --" from .txt file 
    """
    # 1 
    clean_txt = re.sub(r'-{3,}\s*Page\s*-{2,}', '', text, flags=re.IGNORECASE)
    return clean_txt

def process_dfs(df1, df2):
    df = pd.concat([df1, df2])
    df.to_csv("monolingual_warao_all.csv", index=False)

if __name__ == "__main__":
    # combine 2 dataframes of text into one and export into
    # df1 = pd.read_csv("./input/monolingual_warao_sentences_bible-4.csv")
    # df2 = pd.read_csv("./input/monolingual_warao_sentences_bible-5.csv")
    # process_dfs(df1=df1, df2=df2)

    # load txt file + clean + save clean
    input_path = './input/monolingual_warao_final-2.txt'
    with open(input_path, 'r', encoding='utf-8') as f:
        text = f.read()

    clean_text = process_txt_file(text=text)



    # save as df 
    lines = clean_text.split("\n")            
    lines = [l.strip() for l in lines if l.strip()]  # remove empty lines
    df = pd.DataFrame(lines, columns=["warao_text"]) 

    df.to_csv("clean_monolingual_warao_final.csv", index=False, encoding="utf-8")

    # save as txt file
    output_path = './clean_monolingual_warao_final.txt'
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(clean_text)