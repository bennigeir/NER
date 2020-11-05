# -*- coding: utf-8 -*-
"""
Created on Thu Oct 29 20:19:14 2020

@author: Benedikt
"""

import json
import torch
import numpy as np
import io
import requests

from flask import render_template
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
            

def get_model():
    FILE_ID = '1rpyib8PbyKXkzetmwpk08Lnrzi6CHZFP'
    dm = download_file_from_google_drive(FILE_ID)
    return dm


def download_file_from_google_drive(id):
    URL = 'https://drive.google.com/uc?export=download'

    session = requests.Session()

    response = session.get(URL, params = { 'id' : id }, stream = True)
    token = get_confirm_token(response)

    if token:
        params = { 'id' : id, 'confirm' : token }
        response = session.get(URL, params = params, stream = True)

    return io.BytesIO(response.content)


def get_confirm_token(response):
    for key, value in response.cookies.items():
        if key.startswith('download_warning'):
            return value

    return None


def run_model(test_query): 
    
    stream = get_model()
           
    tag2idx = {"I-Time": 0, "I-Money": 1, "B-Date": 2, "I-Person": 3, "B-Percent": 4, "I-Location": 5, "B-Time": 6, "B-Person": 7, "B-Miscellaneous": 8, "I-Percent": 9, "I-Miscellaneous": 10, "I-Date": 11, "I-Organization": 12, "B-Organization": 13, "B-Money": 14, "B-Location": 15, "O": 16, "X": 17, "[CLS]": 18, "[SEP]": 19}
    tag2name = {0: "B-Time", 1: "B-Person", 2: "B-Percent", 3: "I-Location", 4: "O", 5: "B-Organization", 6: "I-Date", 7: "I-Time", 8: "I-Organization", 9: "I-Person", 10: "B-Date", 11: "I-Money", 12: "B-Money", 13: "B-Miscellaneous", 14: "I-Miscellaneous", 15: "I-Percent", 16: "B-Location", 17: "X", 18: "[CLS]", 19: "[SEP]"}
    tag2name = {"0": "B-Time", "1": "B-Person", "2": "B-Percent", "3": "I-Location", "4": "O", "5": "B-Organization", "6": "I-Date", "7": "I-Time", "8": "I-Organization", "9": "I-Person", "10": "B-Date", "11": "I-Money", "12": "B-Money", "13": "B-Miscellaneous", "14": "I-Miscellaneous", "15": "I-Percent", "16": "B-Location", "17": "X", "18": "[CLS]", "19": "[SEP]"}
    
    model = BertForTokenClassification.from_pretrained('bert-base-multilingual-cased',
                                                       num_labels=len(tag2idx))
    
    model.load_state_dict(torch.load(stream))
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()
    
    out = test_model(test_query, model, tag2name)
    
    return out


from flask import Flask, request, jsonify


app = Flask(__name__)


@app.route('/NER/', methods=['GET'])
def ner():
    if request.method == 'GET':
        if set(['query']).issubset(set(request.args)):
            query = request.args.get('query')
            try:
                out = run_model(query)
                response = jsonify({'results': out})
                response.status_code = 201
            except Exception as e:
                response = jsonify({'error': e})
                response.status_code = 400
        else:
            response = json.dumps({'error': 'Parameters missing'})
            response.status_code = 400
    else:
        response = json.dumps({'message': 'Method Not Allowed'})
        response.status_code = 405
    return response


@app.route('/', methods=['GET'])
def default():
    return render_template('index.html')


if __name__ == '__main__':
    
    app.run(debug=True, host='localhost',use_reloader=False)