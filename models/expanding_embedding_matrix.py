"""
This code is adapted from: https://www.cs.columbia.edu/~johnhew//vocab-expansion.html 
"""

import transformers
import torch
import pandas as pd
import random

random.seed(42) # for reproducibility

MULTILINGUAL_MODEL_PATH = "facebook/mbart-large-50"
GPT_2_PATH = 'gpt2'

# load Warao vocabulary
warao_vocab = pd.read_csv('../data_exploration/output/warao_vocab.csv')
warao_vocab_lst = list(warao_vocab['warao_words'])
print(f"Preview vocab: {warao_vocab_lst[:5]}")

# randomly sample 200 words in the vocab
num_wrds = 200
rando_warao_words = random.sample(warao_vocab_lst, k=num_wrds)


tok = transformers.MBart50TokenizerFast.from_pretrained(MULTILINGUAL_MODEL_PATH, src_lang="pt_XX", tgt_lang="es_XX")
# tok = transformers.GPT2Tokenizer.from_pretrained(GPT_2_PATH)
model = transformers.MBartForConditionalGeneration.from_pretrained(MULTILINGUAL_MODEL_PATH)
# model = transformers.AutoModelForCausalLM.from_pretrained(GPT_2_PATH)

# 1) See tokens generated for a single warao word
tokens = tok.convert_ids_to_tokens(tok('janoko')['input_ids'])  # jisabaya means 'house'
print(tokens)

# can investigate the same thing for an entire sentence
warao_sent = 'Ka janoko sanukira'       # Translates to: our house is small
tokens2 = tok.convert_ids_to_tokens(tok(warao_sent)['input_ids'])
print(tokens2)



# 2) Check initial translation: new tokens aren't useful yet since new words' embeddings haven't been trained yet
# encode source Warao text
encoded_wa = tok(warao_sent, return_tensors="pt")
# Decode the translated tokens
generated_tokens = model.generate(
    **encoded_wa,
    forced_bos_token_id=tok.lang_code_to_id["es_XX"]
)
translation = tok.batch_decode(generated_tokens, skip_special_tokens=True)
print(f"First translation, no extension yet: {translation}")


# add tokens
tok.add_tokens(rando_warao_words)
model.resize_token_embeddings(len(tok))
# compute the distribution from which we’ll sample our new embeddings
tokens_added = len(warao_sent)
params = model.state_dict()
embeddings = params['model.shared.weight']
pre_expansion_embeddings = embeddings[:-3,:]
mu = torch.mean(pre_expansion_embeddings, dim=0)
n = pre_expansion_embeddings.size()[0]
sigma = ((pre_expansion_embeddings - mu).T @ (pre_expansion_embeddings - mu)) / n
dist = torch.distributions.multivariate_normal.MultivariateNormal(
        mu, covariance_matrix=1e-5*sigma)

#  load in new embeddings into model
new_embeddings = torch.stack(tuple((dist.sample() for _ in range(3))), dim=0)
embeddings[-3:,:] = new_embeddings
params['model.shared.weight'][-3:,:] = new_embeddings
model.load_state_dict(params)


# try generating Warao after adding embeddings again by initializing new token embedding as the average of all existing embeddings
# sent2 = 'Takore anebu ribu '
# generated_answer3 = tok.decode(model.generate(**tok(sent2, return_tensors='pt'), do_sample=True)[0])
# print(generated_answer3)

# try translating again after adding tokens
encoded_wa2 = tok(warao_sent, return_tensors="pt")
# Decode the translated tokens
generated_tokens2 = model.generate(
    **encoded_wa2,
    forced_bos_token_id=tok.lang_code_to_id["es_XX"]
)
translation2 = tok.batch_decode(generated_tokens2, skip_special_tokens=True)
print(f"Second translation, after extension: {translation2}")