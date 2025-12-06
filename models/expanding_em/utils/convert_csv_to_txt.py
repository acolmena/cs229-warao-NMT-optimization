import pandas as pd
import re  


def covert(input_csv_path=None, output_txt_path=None,column_name=None):
    df = pd.read_csv(input_csv_path)

    with open(output_txt_path, 'w', encoding='utf-8') as txt_file:
        for sent in df['warao_sentence']:
            sent = postprocess_text(sent)
            print(sent)
            # breakpoint()
            txt_file.write(f"{sent}\n")

    print(f"Text file converted and saved to {output_txt_path}")

def postprocess_text(text):
    # match '--- Page X ---' pattern and remove it 
    pattern = r'--- Page \d+ ---'
    return re.sub(pattern, '', text)
    

if __name__ == "__main__":
    input_csv_path = "./input/parallel_train.csv"
    output_txt_path = "./input/parallel_train.txt"

    monol_input_csv_path = "./input/monolingual_warao_sentences_bible.csv"
    monol_output_txt_path = "./input/monolingual_warao_bible.txt"
    
    column_name = 'warao_sentence'

    # covert(input_csv_path, output_txt_path, column_name)
    covert(monol_input_csv_path, monol_output_txt_path, column_name)

    