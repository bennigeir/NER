import pandas as pd
import torch.nn.functional as F
import torch
import math
import configparser
import calendar
import time

from seqeval.metrics import classification_report,accuracy_score,f1_score

from transformers import BertTokenizer
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from sklearn.model_selection import train_test_split
from transformers import BertForTokenClassification, AdamW
from tqdm import trange
from keras.preprocessing.sequence import pad_sequences
from utils import(read_data,
                  sentence_marker,
                  clean_data,
                  )
from SentenceGetter import SentenceGetter


VERBOSE = True
NEW_DATA = True


def get_data(ner_path, tsv_path):

    if NEW_DATA:
        raw_data = read_data(ner_path, verbose=VERBOSE)
    else:
        raw_data = pd.read_csv(tsv_path, sep="\t")

    marked_data = sentence_marker(raw_data, verbose=VERBOSE)

    data = clean_data(marked_data, verbose=VERBOSE)
    
    return data


def corpus_info(data):
    # Info on dataset
    token_count = data['Token'].shape[0]
    u_token_count = data['Token'].nunique()
    tag_count = data['Tag'].nunique()
    sentences_count = data['Sentence no.'].nunique()
    
    print('Token count: {}'.format(token_count))
    print('Unique token count: {}'.format(u_token_count))
    print('Unique tag count: {}'.format(tag_count))
    print('Sentence count: {}'.format(sentences_count))


def data_preparation(data):
    getter = SentenceGetter(data)
    # sentences (list of tokens / words)
    token_sentences = [[s[0] for s in sent] for sent in getter.sentences]
    # pos (NER tags)
    tag_sentences = [[s[1] for s in sent] for sent in getter.sentences]
    
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
    
    return token_sentences, tag_sentences, tag2idx, tag2name
    

def tokenize_labels(token_sentences, tag_sentences, tokenizer):
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
        
        if 5 > i and VERBOSE:
            print("No.%d,len:%d"%(i,len(temp_token)))
            print("texts:%s"%(" ".join(temp_token)))
            print("No.%d,len:%d"%(i,len(temp_label)))
            print("lables:%s"%(" ".join(temp_label)))
        i +=1
    
    return tokenized_texts, word_piece_labels

  
def cut_and_pad(tokenized_texts, word_piece_labels, max_len, tokenizer,
                tag2idx):
    
    # Pad or trim sentences to match max sentence length
    input_ids = pad_sequences([tokenizer.convert_tokens_to_ids(txt) for txt in tokenized_texts],
                              maxlen=max_len, dtype="long", value=0.0, 
                              truncating="post", padding="post")

    # Pad or trim sentences to match max sentence length
    tags = pad_sequences([[tag2idx.get(l) for l in lab] for lab in word_piece_labels],
                         maxlen=max_len, value=tag2idx['O'], padding="post",
                         dtype="long", truncating="post")
    
    return input_ids, tags


def fine_tune_prep(input_ids):
    # For fine tune of predict, with token mask is 1,pad token is 0
    attention_masks = [[int(i>0) for i in ii] for ii in input_ids]
    
    # Since only one sentence, all the segment set to 0
    segment_ids = [[0] * len(input_id) for input_id in input_ids]
    
    return attention_masks, segment_ids


def split_train_test(input_ids, tags,attention_masks, segment_ids):
    # Split data into train and test, 70% and 30%
    tr_inputs, val_inputs, tr_tags, val_tags,tr_masks, val_masks,tr_segs, val_segs\
        = train_test_split(input_ids, tags,attention_masks, segment_ids,
                           random_state=42, test_size=0.3)
    return tr_inputs, val_inputs, tr_tags, val_tags,tr_masks, val_masks,tr_segs, val_segs

    
def data_to_tensor(tr_inputs, val_inputs, tr_tags, val_tags,
                   tr_masks, val_masks, tr_segs, val_segs):
    # Data to tensor
    tr_inputs = torch.tensor(tr_inputs).to(torch.int64)
    val_inputs = torch.tensor(val_inputs).to(torch.int64)
    tr_tags = torch.tensor(tr_tags).to(torch.int64)
    val_tags = torch.tensor(val_tags).to(torch.int64)
    tr_masks = torch.tensor(tr_masks).to(torch.int64)
    val_masks = torch.tensor(val_masks).to(torch.int64)
    tr_segs = torch.tensor(tr_segs).to(torch.int64)
    val_segs = torch.tensor(val_segs).to(torch.int64)
    
    return tr_inputs, val_inputs, tr_tags, val_tags, \
        tr_masks, val_masks, tr_segs, val_segs
        


def data_dataloader(tr_inputs, val_inputs, tr_tags, val_tags,
                   tr_masks, val_masks, tr_segs, val_segs, batch_num):
    # Data to dataloader    
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
    
    return train_dataloader, valid_dataloader


def fine_tuning(model):
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
    
    return optimizer 


def train_model(tr_inputs, batch_num, num_train_optimization_steps,
                train_dataloader, device, model, max_grad_norm,
                optimizer, n_gpu, epochs):

    model.train();
    
    if VERBOSE:
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
            attention_mask = b_input_mask, labels=b_labels)
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

        if VERBOSE:            
            # Print train loss per epoch
            print("Train loss: {}".format(tr_loss/nb_tr_steps))
            
    return model


def save_model(model, path='data/models/saved_model'):
    # torch.save(model, path)
    torch.save(model.state_dict(), path)


def load_model(path='data/models/saved_model_hrafn'):
    return torch.load(path)


def evaluate(model, valid_dataloader, device, tag2name, val_inputs, batch_num,
             write_data=False):

    model.eval();
    # model.eval();
    
    # eval_loss, eval_accuracy = 0, 0
    # nb_eval_steps, nb_eval_examples = 0, 0
    y_true = []
    y_pred = []
    
    # if VERBOSE:
    print("***** Running evaluation *****")
    print("  Num examples ={}".format(len(val_inputs)))
    print("  Batch size = {}".format(batch_num))
        
    for step, batch in enumerate(valid_dataloader):
        batch = tuple(t.to(device) for t in batch)
        input_ids, input_mask, label_ids = batch

        with torch.no_grad():
            outputs = model(input_ids, token_type_ids=None,
            attention_mask=input_mask,)
            # outputs = load_model(input_ids, token_type_ids=None,
            # attention_mask=input_mask,)
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
        for i, mask in enumerate(input_mask):
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
    
    # if VERBOSE:
    print("f1 socre: %f"%(f1_score(y_true, y_pred)))
    print("Accuracy score: %f"%(accuracy_score(y_true, y_pred)))
    
    # Get acc , recall, F1 result report
    report = classification_report(y_true, y_pred,digits=4)
    
    if write_data:
        # Save the report into file
        ts = calendar.timegm(time.gmtime())
        output_eval_file = "eval_results_" + str(ts) + ".txt"
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


def main():
    
    config = configparser.ConfigParser()
    config.read('config.ini')
    
    MIM_GOLD_NER_FOLDER_PATH = config['PATHS']['gold_path']
    MIM_GOLD_NER_TSV_FILE = config['PATHS']['tsv_gold_file']
    
    
    data = get_data(MIM_GOLD_NER_FOLDER_PATH, MIM_GOLD_NER_TSV_FILE)
    corpus_info(data)
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
    max_len  = 75
    
    
    input_ids, tags = cut_and_pad(tokenized_texts, word_piece_labels, max_len,
                                  tokenizer, tag2idx)
    
    attention_masks, segment_ids = fine_tune_prep(input_ids)
    
    tr_inputs, val_inputs, tr_tags, val_tags,tr_masks, val_masks,tr_segs, val_segs\
        = split_train_test(input_ids, tags,attention_masks, segment_ids)
        
    tr_inputs, val_inputs, tr_tags, val_tags,tr_masks, val_masks,tr_segs, val_segs\
        = data_to_tensor(tr_inputs, val_inputs, tr_tags, val_tags, tr_masks, 
                         val_masks, tr_segs, val_segs)
    
    
    batch_num = 32
    train_dataloader, valid_dataloader = data_dataloader(tr_inputs, val_inputs, 
                                                         tr_tags, val_tags,
                                                         tr_masks, val_masks, 
                                                         tr_segs, val_segs, 
                                                         batch_num)
    
    # Token-level classifier. Load with 'bert-base-multilingual-cased'
    model = BertForTokenClassification.from_pretrained('bert-base-multilingual-cased',
                                                       num_labels=len(tag2idx))
    # Pass parameters to GPU
    model.cuda();
    # Support for multiple GPUs
    if n_gpu >1:
        model = torch.nn.DataParallel(model)
    
    epochs = 5
    max_grad_norm = 1.0
    
    # Training optimization num
    num_train_optimization_steps = int(math.ceil(len(tr_inputs) / batch_num) / 1) * epochs
    
    optimizer = fine_tuning(model)
    model = train_model(tr_inputs, batch_num, num_train_optimization_steps,
                        train_dataloader, device, model, max_grad_norm,
                        optimizer, n_gpu, epochs)
    
    evaluate(model, valid_dataloader, device, tag2name, val_inputs, batch_num, 
             write_data=True)


if __name__ == "__main__":
    main()