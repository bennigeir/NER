# -*- coding: utf-8 -*-
"""
Created on Thu Oct 29 20:19:14 2020

@author: Benedikt
"""

import json
import torch
import numpy as np
import configparser

from keras.preprocessing.sequence import pad_sequences
from transformers import (BertForTokenClassification,
                          BertTokenizer,
                          )


def get_tag2idx(file_path):   
    with open(file_path) as json_file:
        tag2idx = json.load(json_file) 
    return tag2idx


def get_tag2name(file_path):   
    with open(file_path) as json_file:
        tag2name = json.load(json_file) 
    return tag2name
    

def test_model(test_query, t_model, tag2name):
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
                              maxlen=max_len, dtype="long", value=0.0, 
                              truncating="post", padding="post")
    
    attention_masks = [[int(i>0) for i in ii] for ii in input_ids]
    segment_ids = [[0] * len(input_id) for input_id in input_ids]
    
    input_ids = torch.tensor(input_ids).to(torch.int64).to(device)
    attention_masks = torch.tensor(attention_masks).to(torch.int64).to(device)
    segment_ids = torch.tensor(segment_ids).to(torch.int64).to(device)
    
    t_model.eval();
    
    with torch.no_grad():
            outputs = t_model(input_ids, token_type_ids=None, attention_mask=None)
            logits = outputs[0]
    
    predict_results = logits.detach().cpu().numpy()
    
    from scipy.special import softmax
    
    result_arrays_soft = softmax(predict_results[0])
    
    result_array = result_arrays_soft
    
    result_list = np.argmax(result_array,axis=-1)
    
    test = [x[0:2] != '##' for x in temp_token]
    test = np.where(test)
    
    temp_string = ' '.join(temp_token).replace(' ##', '')
    temp_string_list = temp_string.split(' ')
    
    out = []
    for i,j in zip(range(len(temp_string_list)), test[0]):
        out.append((temp_string_list[i], tag2name[str(result_list[j])]))
    return out
            

def run_model(test_query): 

    config = configparser.ConfigParser()
    config.read('config.ini')
    
    SAVED_MODEL_PATH = config['PATHS']['model_default_load']
    TAG2IDX_PATH = config['PATHS']['tag2idx']
    TAG2NAME_PATH = config['PATHS']['tag2name']
           
    tag2idx = get_tag2idx(TAG2IDX_PATH)
    tag2name = get_tag2name(TAG2NAME_PATH)
    model = BertForTokenClassification.from_pretrained('bert-base-multilingual-cased',
                                                       num_labels=len(tag2idx))
    
    model.load_state_dict(torch.load(SAVED_MODEL_PATH))
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()
    
    out = test_model(test_query, model, tag2name)
    
    return out