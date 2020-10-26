import pandas as pd
import numpy as np
import os
import torch.nn.functional as F
from seqeval.metrics import classification_report,accuracy_score,f1_score
import torch
from transformers import BertTokenizer
import math
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from sklearn.model_selection import train_test_split
from transformers import BertForTokenClassification, AdamW
from tqdm import trange
from keras.preprocessing.sequence import pad_sequences


# WARNING, EXPECT 5+ HRS RUNNING USING CUDA


# %%

# Read merged dataset
data = pd.read_csv('master_corpus.tsv', sep="\t")
print(data.head(5))

# %%

# Info on dataset

token_count = data['Token'].nunique()
tag_count = data['Tag'].nunique()
pos_count = data['POS'].nunique()
sentences_count = data['Sentence no.'].nunique()

print('Token count: {}'.format(token_count))
print('Tag count: {}'.format(tag_count))
print('POS count: {}'.format(pos_count))
print('Sentence count: {}'.format(sentences_count))

# %%

# Data needs to be parsed into sentences
class SentenceGetter(object):
    def __init__(self, data):
        self.n_sent = 1
        self.data = data
        self.empty = False
        agg_func = lambda s: [(w, p, t) for w, p, t in zip(s['Token'].values.tolist(),
                                                            s['Tag'].values.tolist(),
                                                            s['POS'].values.tolist())
                               ]        
        self.grouped = self.data.groupby("Sentence no.").apply(agg_func)
        self.sentences = [s for s in self.grouped]
    
    def get_next(self):
        try:
            s = self.grouped["Sentence: {}".format(self.n_sent)]
            self.n_sent += 1
            return s
        except:
            return None

# %%
            
getter = SentenceGetter(data)

# sentences (list of tokens / words)
token_sentences = [[s[0] for s in sent] for sent in getter.sentences]

# pos (NER tags)
tag_sentences = [[s[1] for s in sent] for sent in getter.sentences]

#tag (PoS tags)
# pos_sentences = [[s[2] for s in sent] for sent in getter.sentences]

# %%

# Tags to index
tags_vals = list(set(data['Tag'].values))

# Add X, CLS and SEP for BERT
tags_vals.append('X')
tags_vals.append('[CLS]')
tags_vals.append('[SEP]')

# Dictionary, map id to tags
tag2idx = {t: i for i, t in enumerate(tags_vals)}

# Dictionary, map index to name
tag2name = {tag2idx[key] : key for key in tag2idx.keys()}

# %%

# Set up GPU for training
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
n_gpu = torch.cuda.device_count()

# %%

# Max sentence length. Bert supports sequences of up to 512 tokens, 75 tokens 
# and a batch size of 32 is suggested by the Bert paper.
max_len  = 75

# Tokenizer using bert-base-multilingual-cased
tokenizer = BertTokenizer.from_pretrained('bert-base-multilingual-cased', 
                                          do_lower_case=False)

# %%

verbose = False

tokenized_texts = []
word_piece_labels = []

i = 0

for word_list, label in (zip(token_sentences, tag_sentences)):
    temp_label = []
    temp_token = []
    
    # Add [CLS] at the front of sentence
    temp_label.append('[CLS]')
    temp_token.append('[CLS]')
    
    for word, lab in zip(word_list, label):
        token_list = tokenizer.tokenize(word)
        
        for m, token in enumerate(token_list):
            temp_token.append(token)
            if m==0:
                temp_label.append(lab)
            else:
                # X label is set for ###something results
                temp_label.append('X')  
                
    # Add [SEP] at the end
    temp_label.append('[SEP]')
    temp_token.append('[SEP]')
    
    tokenized_texts.append(temp_token)
    word_piece_labels.append(temp_label)
    
    if 5 > i and verbose:
        print("No.%d,len:%d"%(i,len(temp_token)))
        print("texts:%s"%(" ".join(temp_token)))
        print("No.%d,len:%d"%(i,len(temp_label)))
        print("lables:%s"%(" ".join(temp_label)))
    i +=1

# %%

# Pad or trim sentences to match max sentence length
input_ids = pad_sequences([tokenizer.convert_tokens_to_ids(txt) for txt in tokenized_texts],
                          maxlen=max_len, dtype="long", value=0.0, 
                          truncating="post", padding="post")

# %%

# Pad or trim sentences to match max sentence length
tags = pad_sequences([[tag2idx.get(l) for l in lab] for lab in word_piece_labels],
                     maxlen=max_len, value=tag2idx['O'], padding="post",
                     dtype="long", truncating="post")

# %%

# For fine tune of predict, with token mask is 1,pad token is 0
attention_masks = np.array([[int(i>0) for i in ii] for ii in input_ids])

# Since only one sentence, all the segment set to 0
segment_ids = np.array([[0] * len(input_id) for input_id in input_ids])

# %%

# Shuffle
p = np.random.permutation(len(input_ids))

input_ids = input_ids[p]
tags = tags[p]
attention_masks = attention_masks[p]
segment_ids = segment_ids[p]

# %%


### K FOLD SPLIT ###


from sklearn.model_selection import KFold
n_splits = 10

batch_num = 32
# model = BertForTokenClassification.from_pretrained('bert-base-multilingual-cased',
                                                   # num_labels=len(tag2idx))

for train_index, test_index in KFold(n_splits).split(input_ids):
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
    
    # Data to tensor
    tr_inputs = torch.tensor(tr_inputs).to(torch.int64)
    val_inputs = torch.tensor(val_inputs).to(torch.int64)
    tr_tags = torch.tensor(tr_tags).to(torch.int64)
    val_tags = torch.tensor(val_tags).to(torch.int64)
    tr_masks = torch.tensor(tr_masks).to(torch.int64)
    val_masks = torch.tensor(val_masks).to(torch.int64)
    tr_segs = torch.tensor(tr_segs).to(torch.int64)
    val_segs = torch.tensor(val_segs).to(torch.int64)
    
    
    # Only set token embedding, attention embedding, no segment embedding
    train_data = TensorDataset(tr_inputs, tr_masks, tr_tags)
    train_sampler = RandomSampler(train_data)
    # Drop last can make batch training better for the last one
    train_dataloader = DataLoader(train_data, sampler=train_sampler, 
                                  batch_size=batch_num,drop_last=True)
    
    valid_data = TensorDataset(val_inputs, val_masks, val_tags)
    valid_sampler = SequentialSampler(valid_data)
    valid_dataloader = DataLoader(valid_data, sampler=valid_sampler, 
                                  batch_size=batch_num)

    # Pass parameters to GPU
    model.cuda();
    
    # Support for multiple GPUs
    if n_gpu >1:
        model = torch.nn.DataParallel(model)
    
    epochs = 5
    max_grad_norm = 1.0
    
    # Training optimization num
    num_train_optimization_steps = int( math.ceil(len(tr_inputs) / batch_num) / 1) * epochs
    
    # Fine tuning all layers
    FULL_FINETUNING = True
    
    if FULL_FINETUNING:
        # Fine tune model all layer parameters
        param_optimizer = list(model.named_parameters())
        no_decay = ['bias', 'gamma', 'beta']
        optimizer_grouped_parameters = [
            {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)],
             'weight_decay_rate': 0.01},
            {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)],
             'weight_decay_rate': 0.0}
        ]
    else:
        # Only fine tune classifier parameters
        param_optimizer = list(model.classifier.named_parameters()) 
        optimizer_grouped_parameters = [{"params": [p for n, p in param_optimizer]}]
    
    optimizer = AdamW(optimizer_grouped_parameters, lr=3e-5)
    
    
    model.train();

    print("***** Running training *****")
    print("  Num examples = %d"%(len(tr_inputs)))
    print("  Batch size = %d"%(batch_num))
    print("  Num steps = %d"%(num_train_optimization_steps))
    
    for _ in trange(epochs, desc="Epoch"):
        tr_loss = 0
        nb_tr_examples, nb_tr_steps = 0, 0
        
        for step, batch in enumerate(train_dataloader):
            # Add batch to GPU
            batch = tuple(t.to(device) for t in batch)
            b_input_ids, b_input_mask, b_labels = batch
            
            # Forward pass
            outputs = model(b_input_ids, token_type_ids=None,
            attention_mask=b_input_mask, labels=b_labels)
            loss, scores = outputs[:2]
            if n_gpu>1:
                # When multi GPU, average it
                loss = loss.mean()
            
            # Backward pass
            loss.backward()
            
            # Track train loss
            tr_loss += loss.item()
            nb_tr_examples += b_input_ids.size(0)
            nb_tr_steps += 1
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(parameters=model.parameters(), max_norm=max_grad_norm)
            
            # Update parameters
            optimizer.step()
            optimizer.zero_grad()
            
        # Print train loss per epoch
        print("Train loss: {}".format(tr_loss/nb_tr_steps))


    model.eval();

    eval_loss, eval_accuracy = 0, 0
    nb_eval_steps, nb_eval_examples = 0, 0
    y_true = []
    y_pred = []
    
    print("***** Running evaluation *****")
    print("  Num examples ={}".format(len(val_inputs)))
    print("  Batch size = {}".format(batch_num))
    for step, batch in enumerate(valid_dataloader):
        batch = tuple(t.to(device) for t in batch)
        input_ids_2, input_mask, label_ids = batch
        
    #     if step > 2:
    #         break
        
        with torch.no_grad():
            outputs = model(input_ids_2, token_type_ids=None,
            attention_mask=input_mask,)
            # For eval mode, the first result of outputs is logits
            logits = outputs[0] 
        
        # Get NER predict result
        logits = torch.argmax(F.log_softmax(logits,dim=2),dim=2)
        logits = logits.detach().cpu().numpy()
        
        
        # Get NER true result
        label_ids = label_ids.to('cpu').numpy()
        
        
        # Only predict the real word, mark=0, will not calculate
        input_mask = input_mask.to('cpu').numpy()
        
        # Compare the valuable predict result
        for i,mask in enumerate(input_mask):
            # Real one
            temp_1 = []
            # Predict one
            temp_2 = []
            
            for j, m in enumerate(mask):
                # Mark=0, meaning its a pad word, dont compare
                if m:
                    if tag2name[label_ids[i][j]] != "X" and tag2name[label_ids[i][j]] != "[CLS]" and tag2name[label_ids[i][j]] != "[SEP]" : # Exclude the X label
                        temp_1.append(tag2name[label_ids[i][j]])
                        temp_2.append(tag2name[logits[i][j]])
                else:
                    break
            
                
            y_true.append(temp_1)
            y_pred.append(temp_2)
    
            
    
    print("f1 socre: %f"%(f1_score(y_true, y_pred)))
    print("Accuracy score: %f"%(accuracy_score(y_true, y_pred)))

# %%


# %%
    
from transformers import BertConfig

torch.save(model, 'models/saved_model_cross_val')

# %%

load_model = torch.load('models/saved_model')
# load_config = BertConfig.from_json_file('./tf_model/my_tf_model_config.json')
# load_model = BertForTokenClassification.from_pretrained('./tf_model/my_tf_checkpoint.ckpt.index', from_tf=True, config=config)

# %%    

load_model.eval();
# model.eval();

eval_loss, eval_accuracy = 0, 0
nb_eval_steps, nb_eval_examples = 0, 0
y_true = []
y_pred = []

print("***** Running evaluation *****")
print("  Num examples ={}".format(len(val_inputs)))
print("  Batch size = {}".format(batch_num))
for step, batch in enumerate(valid_dataloader):
    batch = tuple(t.to(device) for t in batch)
    input_ids, input_mask, label_ids = batch
    
#     if step > 2:
#         break
    
    with torch.no_grad():
        # outputs = model(input_ids, token_type_ids=None,
        # attention_mask=input_mask,)
        outputs = load_model(input_ids, token_type_ids=None,
        attention_mask=input_mask,)
        # For eval mode, the first result of outputs is logits
        logits = outputs[0] 
    
    # Get NER predict result
    logits = torch.argmax(F.log_softmax(logits,dim=2),dim=2)
    logits = logits.detach().cpu().numpy()
    
    
    # Get NER true result
    label_ids = label_ids.to('cpu').numpy()
    
    
    # Only predict the real word, mark=0, will not calculate
    input_mask = input_mask.to('cpu').numpy()
    
    # Compare the valuable predict result
    for i,mask in enumerate(input_mask):
        # Real one
        temp_1 = []
        # Predict one
        temp_2 = []
        
        for j, m in enumerate(mask):
            # Mark=0, meaning its a pad word, dont compare
            if m:
                if tag2name[label_ids[i][j]] != "X" and tag2name[label_ids[i][j]] != "[CLS]" and tag2name[label_ids[i][j]] != "[SEP]" : # Exclude the X label
                    temp_1.append(tag2name[label_ids[i][j]])
                    temp_2.append(tag2name[logits[i][j]])
            else:
                break
        
            
        y_true.append(temp_1)
        y_pred.append(temp_2)

        

print("f1 socre: %f"%(f1_score(y_true, y_pred)))
print("Accuracy score: %f"%(accuracy_score(y_true, y_pred)))

# Get acc , recall, F1 result report
report = classification_report(y_true, y_pred,digits=4)

# Save the report into file
output_eval_file = "eval_results.txt"
with open(output_eval_file, "w") as writer:
    print("***** Eval results *****")
    print("\n%s"%(report))
    print("f1 socre: %f"%(f1_score(y_true, y_pred)))
    print("Accuracy score: %f"%(accuracy_score(y_true, y_pred)))
    
    writer.write("f1 socre:\n")
    writer.write(str(f1_score(y_true, y_pred)))
    writer.write("\n\nAccuracy score:\n")
    writer.write(str(accuracy_score(y_true, y_pred)))
    writer.write("\n\n")  
    writer.write(report)

    # %%


# LOAD
# bert_out_address = 'models/bert_out_model/'
# model = BertForTokenClassification.from_pretrained(bert_out_address,num_labels=len(tag2idx))
# Set model to GPU
load_model.cuda();

# %%

test_query = """19 skjálft­ar yfir 3 að stærð hafa verið staðfest­ir frá stóra skjálft­an­um að sögn Ein­ars Hjör­leifs­son­ar, nátt­úru­vár­sér­fræðings á Veður­stofu Íslands."""
# test_query = """Þessar ömmustelpur eru 13 ára í dag. Elsku Þórunn Elísa og Freydís Ólöf. Innilega til hamingju með daginn ykkar."""

tokenized_texts = []
temp_token = []
temp_token.append('[CLS]')
token_list = tokenizer.tokenize(test_query)

print(token_list)

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

load_model.eval();

with torch.no_grad():
        outputs = load_model(input_ids, token_type_ids=None, attention_mask=None,)
        # For eval mode, the first result of outputs is logits
        logits = outputs[0]

predict_results = logits.detach().cpu().numpy()

from scipy.special import softmax

result_arrays_soft = softmax(predict_results[0])

result_array = result_arrays_soft

result_list = np.argmax(result_array,axis=-1)

for i, mark in enumerate(attention_masks[0]):
    if mark>0:
        print("Token:%s"%(temp_token[i]))
#         print("Tag:%s"%(result_list[i]))
        print("Predict_Tag:%s"%(tag2name[result_list[i]]))
        #print("Posibility:%f"%(result_array[i][result_list[i]]))
        print()






