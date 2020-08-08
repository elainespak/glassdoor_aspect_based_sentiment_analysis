# -*- coding: utf-8 -*-

### Following:
### https://colab.research.google.com/drive/1yFphU6PW9Uo6lmDly_ud9a6c4RCYlwdX#scrollTo=MQv0FL8VWadn


import torch
from transformers import BertTokenizer, BertModel
import logging # OPTIONAL: if you want to have more information on what's happening, activate the logger as follows
#logging.basicConfig(level=logging.INFO)
import matplotlib.pyplot as plt

% matplotlib inline
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')


text = "Here is the sentence I want embeddings for."
marked_text = "[CLS] " + text + " [SEP]"

# Tokenize our sentence with the BERT tokenizer.
tokenized_text = tokenizer.tokenize(marked_text)

# Print out the tokens.
print (tokenized_text)


# Define a new example sentence with multiple meanings of the word "bank"
text = "After stealing money from the bank vault, the bank robber was seen " \
       "fishing on the Mississippi river bank."

# Add the special tokens.
marked_text = "[CLS] " + text + " [SEP]"

# Split the sentence into tokens.
tokenized_text = tokenizer.tokenize(marked_text)

# Map the token strings to their vocabulary indeces.
indexed_tokens = tokenizer.convert_tokens_to_ids(tokenized_text)

# Display the words with their indeces.
for tup in zip(tokenized_text, indexed_tokens):
    print('{:<12} {:>6,}'.format(tup[0], tup[1]))
    
# Mark each of the 22 tokens as belonging to sentence "1".
segments_ids = [1] * len(tokenized_text)

print (segments_ids)


# Convert inputs to PyTorch tensors
tokens_tensor = torch.tensor([indexed_tokens])
segments_tensors = torch.tensor([segments_ids])

# Load pre-trained model (weights)
model = BertModel.from_pretrained('bert-base-uncased',
                                  output_hidden_states = True, # Whether the model returns all hidden-states.
                                  )

# Put the model in "evaluation" mode, meaning feed-forward operation.
model.eval()

# Run the text through BERT, and collect all of the hidden states produced
# from all 12 layers. 
with torch.no_grad():

    outputs = model(tokens_tensor, segments_tensors)

    # Evaluating the model will return a different number of objects based on 
    # how it's  configured in the `from_pretrained` call earlier. In this case, 
    # becase we set `output_hidden_states = True`, the third item will be the 
    # hidden states from all layers. See the documentation for more details:
    # https://huggingface.co/transformers/model_doc/bert.html#bertmodel
    hidden_states = outputs[2]

# Concatenate the tensors for all layers. We use `stack` here to
# create a new dimension in the tensor.
token_embeddings = torch.stack(hidden_states, dim=0)

token_embeddings.size()

# Remove dimension 1, the "batches".
token_embeddings = torch.squeeze(token_embeddings, dim=1)

token_embeddings.size()

# Swap dimensions 0 and 1.
token_embeddings = token_embeddings.permute(1,0,2)

token_embeddings.size()


# Stores the token vectors, with shape [22 x 3,072]
token_vecs_cat = []

# `token_embeddings` is a [22 x 12 x 768] tensor.

# For each token in the sentence...
for token in token_embeddings:
    
    # `token` is a [12 x 768] tensor

    # Concatenate the vectors (that is, append them together) from the last 
    # four layers.
    # Each layer vector is 768 values, so `cat_vec` is length 3,072.
    cat_vec = torch.cat((token[-1], token[-2], token[-3], token[-4]), dim=0)
    
    # Use `cat_vec` to represent `token`.
    token_vecs_cat.append(cat_vec)

print ('Shape is: %d x %d' % (len(token_vecs_cat), len(token_vecs_cat[0])))


# `hidden_states` has shape [13 x 1 x 22 x 768]

# `token_vecs` is a tensor with shape [22 x 768]
token_vecs = hidden_states[-2][0]

# Calculate the average of all 22 token vectors.
sentence_embedding = torch.mean(token_vecs, dim=0)

print ("Our final sentence embedding vector of shape:", sentence_embedding.size())




###########

class BERTEmbedding:

    def __init__(self, args):
        self.device = args.device
        self.bert, self.vocab, self.tokenizer, self.tok = self.retrieve_kobert()
        self.bert.eval()

    def retrieve_kobert(self):
        bert, vocab = get_pytorch_kobert_model()
        tokenizer = get_tokenizer()
        tok = nlp.data.BERTSPTokenizer(tokenizer, vocab, lower=False)

        return bert.to(self.device), vocab, tokenizer, tok

    def tokenize(self, sent):
        tokenized_sent = ['[CLS]'] + self.tok(sent) + ['[SEP]'] # BERTSUM은 각 문장 앞에 [CLS] 토큰을, 각 문장 뒤에 [SEP] 토큰을 더함
        return tokenized_sent

    def token_to_ids(self, sent):
        return self.tok.convert_tokens_to_ids(sent)

    def get_embeddings(self, indexed_tokens):
        # Extract KoBERT embedding from the last hidden layer
        segments_ids = [1] * len(indexed_tokens)
        tokens_tensor = torch.tensor([indexed_tokens]).to(self.device)
        segments_tensors = torch.tensor([segments_ids]).to(self.device)
        with torch.no_grad():
            encoded_layers, _ = self.bert(tokens_tensor, segments_tensors)
        token_vecs = encoded_layers[0]
        # sentence_embedding = torch.mean(encoded_layers[0],dim=0) # 문장 단위 embedding이 필요할 때 사용
        return token_vecs

is_cuda = torch.cuda.is_available()
if is_cuda and args.cuda >= 0:
     args.device = torch.device(f"cuda:{args.cuda}")
else:
     args.device = torch.device("cpu")
print(args.device)

bert_embedding = BERTEmbedding(args)
tokenized_text = bert_embedding.tokenize(sentence)
token_ids = bert_embedding.token_to_ids(tokenized_text)
sentence_emb = bert_embedding.get_embeddings(token_ids)