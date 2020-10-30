# -*- coding: utf-8 -*-
"""
Created on Thu Oct 29 20:19:14 2020

@author: Benedikt
"""

import json
import torch
import numpy as np

from keras.preprocessing.sequence import pad_sequences
from transformers import (BertForTokenClassification,
                          BertTokenizer,
                          )

SAVED_MODEL_PATH = 'data/models/28102020_2'
TAG2IDX_PATH = 'tag2idx.json'
TAG2NAME_PATH = 'tag2name.json'


def get_tag2idx():   
    with open(TAG2IDX_PATH) as json_file:
        tag2idx = json.load(json_file) 
    return tag2idx


def get_tag2name():   
    with open(TAG2NAME_PATH) as json_file:
        tag2name = json.load(json_file) 
    return tag2name
    

def test_model(test_query, t_model):
    # Tokenizer using bert-base-multilingual-cased
    tokenizer = BertTokenizer.from_pretrained('bert-base-multilingual-cased', 
                                              do_lower_case=False)
    max_len  = 75
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    tokenized_texts = []
    temp_token = []
    temp_token.append('[CLS]')
    token_list = tokenizer.tokenize(test_query)
    
    for m,token in enumerate(token_list):
        temp_token.append(token)
    
    if len(temp_token) > max_len-1:
        temp_token= temp_token[:max_len-1]
    
    temp_token.append('[SEP]')
    
    tokenized_texts.append(temp_token)
    
    input_ids = pad_sequences([tokenizer.convert_tokens_to_ids(txt) for txt in tokenized_texts],
                              maxlen=max_len, dtype="long", value=0.0, truncating="post", padding="post")
    
    attention_masks = [[int(i>0) for i in ii] for ii in input_ids]
    segment_ids = [[0] * len(input_id) for input_id in input_ids]
    
    input_ids = torch.tensor(input_ids).to(torch.int64).to(device)
    attention_masks = torch.tensor(attention_masks).to(torch.int64).to(device)
    segment_ids = torch.tensor(segment_ids).to(torch.int64).to(device)
    
    t_model.eval();
    
    with torch.no_grad():
            outputs = t_model(input_ids, token_type_ids=None, attention_mask=None,)
            # For eval mode, the first result of outputs is logits
            logits = outputs[0]
    
    predict_results = logits.detach().cpu().numpy()
    
    from scipy.special import softmax
    
    result_arrays_soft = softmax(predict_results[0])
    
    result_array = result_arrays_soft
    
    result_list = np.argmax(result_array,axis=-1)
    
    tag2name = get_tag2name()
    
    for i, mark in enumerate(attention_masks[0]):
        if mark>0:
            print("Token:%s"%(temp_token[i]))
            # print("Tag:%s"%(result_list[i]))
            print("Predict_Tag:%s"%(tag2name[str(result_list[i])]))
            # print("Posibility:%f"%(result_array[i][result_list[i]]))
            print()
            
            
# %%

        
tag2idx = get_tag2idx()
model = BertForTokenClassification.from_pretrained('bert-base-multilingual-cased',
                                                   num_labels=len(tag2idx))

model.load_state_dict(torch.load(SAVED_MODEL_PATH))

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
model.eval()

test_query = '''að sögn Ein­ars Hjör­leifs­son­ar, nátt­úru­vár­sér­fræðings á 
Veður­stofu Íslands.'''

test_model(test_query, model)