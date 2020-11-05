# -*- coding: utf-8 -*-
"""
Created on Thu Oct 29 20:42:27 2020

@author: Benedikt
"""

import numpy as np
import torch
import math
import random
import configparser

from sklearn.model_selection import KFold
from transformers import BertForTokenClassification
from transformers import BertTokenizer
from sandbox_bert import (get_data,
                          data_preparation,
                          tokenize_labels,
                          cut_and_pad,
                          fine_tune_prep,
                          data_to_tensor,
                          data_dataloader,
                          fine_tuning,
                          train_model,
                          evaluate
                          )


BATCH_NUM = 32
N_SPLITS = 10


def main():
    
    config = configparser.ConfigParser()
    config.read('config.ini')
    
    MIM_GOLD_NER_FOLDER_PATH = config['PATHS']['gold_path']
    MIM_GOLD_NER_TSV_FILE = config['PATHS']['tsv_gold_file']
    
    data = get_data(MIM_GOLD_NER_FOLDER_PATH, MIM_GOLD_NER_TSV_FILE)
    token_sentences, tag_sentences, tag2idx, tag2name = data_preparation(data)
    
    # Set up GPU for training
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    n_gpu = torch.cuda.device_count()
    
    # Tokenizer using bert-base-multilingual-cased
    tokenizer = BertTokenizer.from_pretrained('bert-base-multilingual-cased', 
                                              do_lower_case=False)
    
    tokenized_texts, word_piece_labels = tokenize_labels(token_sentences, 
                                                     tag_sentences, 
                                                     tokenizer)
    
    # Max sentence length. Bert supports sequences of up to 512 tokens, 75 tokens 
    # and a batch size of 32 is suggested by the Bert paper.
    # max_len  = 75
    max_len = 100
    
    input_ids, tags = cut_and_pad(tokenized_texts, word_piece_labels, max_len,
                                  tokenizer, tag2idx)
    attention_masks, segment_ids = fine_tune_prep(input_ids)
    
    attention_masks = np.array(attention_masks)
    segment_ids = np.array(segment_ids)
    
    ### K-FOLD
    
    # Shuffle
    # p = np.random.permutation(len(input_ids))
    p = [x for x in range(len(input_ids))]
    random.shuffle(p)
        
    input_ids = input_ids[p]
    tags = tags[p]
    attention_masks = attention_masks[p]
    segment_ids = segment_ids[p]
    
    for train_index, test_index in KFold(N_SPLITS).split(input_ids):
        torch.cuda.empty_cache()    
        model = BertForTokenClassification.from_pretrained('bert-base-multilingual-cased',
                                                       num_labels=len(tag2idx))
        
        tr_inputs = input_ids[train_index] 
        val_inputs = input_ids[test_index]
        tr_tags = tags[train_index] 
        val_tags = tags[test_index]
        tr_masks = attention_masks[train_index]
        val_masks = attention_masks[test_index]
        tr_segs = segment_ids[train_index]
        val_segs = segment_ids[test_index]
        
        tr_inputs, val_inputs, tr_tags, val_tags,tr_masks, val_masks,tr_segs, val_segs\
        = data_to_tensor(tr_inputs, val_inputs, tr_tags, val_tags, tr_masks, 
                         val_masks, tr_segs, val_segs)
        
        train_dataloader, valid_dataloader\
            = data_dataloader(tr_inputs, val_inputs, tr_tags, val_tags, tr_masks, 
                              val_masks, tr_segs, val_segs, BATCH_NUM)
            
        model.cuda();
        if n_gpu >1:
            model = torch.nn.DataParallel(model)
        epochs = 5
        max_grad_norm = 1.0
    
        num_train_optimization_steps = int(math.ceil(len(tr_inputs) / BATCH_NUM) / 1) * epochs
        
        optimizer = fine_tuning(model)
        
        model = train_model(tr_inputs, BATCH_NUM, num_train_optimization_steps,
                            train_dataloader, device, model, max_grad_norm, 
                            optimizer, n_gpu, epochs)
            
        evaluate(model, valid_dataloader, device, tag2name, val_inputs, 
                 BATCH_NUM, write_data=True)
        
        del model
    

if __name__ == "__main__":
    main()