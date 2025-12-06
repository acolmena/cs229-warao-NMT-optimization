from transformers import M2M100ForConditionalGeneration, AutoModel, AutoTokenizer
import torch

# all Frankenstein type

m2m = M2M100ForConditionalGeneration.from_pretrained("facebook/m2m100_418M")
xlmr = AutoModel.from_pretrained("./utils/input/warao-encoder-mlm")
m2m.model.encoder.load_state_dict(xlmr.state_dict(), strict=False)
m2m.save_pretrained("combined_encoder_m2m")
