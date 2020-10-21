import os
import pandas as pd


MIM_GOLD_NER_PATH = 'data/MIM-GOLD-NER/'
MIM_GOLD_PATH = 'data/MIM-GOLD-1_0/'

FILES = [f for f in os.listdir(MIM_GOLD_NER_PATH)]


def read_data():
    dataframe = pd.DataFrame()

    # Iterate through files in MIM GOLD NER and MIM GOLD 1.0 and merge them
    try:
        for file in FILES:
            merged = merge_files(file)
            dataframe = pd.concat([dataframe, merged])
        
        return dataframe.reset_index()
    except FileNotFoundError:
        return None


def merge_files(file):
    ner_data = pd.read_csv(os.path.join(MIM_GOLD_NER_PATH, file), 
                           sep="\t", 
                           header=None)
    column_names = ['Token','Tag']
    ner_data.columns = column_names
    
    pos_data = pd.read_csv(os.path.join(MIM_GOLD_PATH, file), 
                           sep="\t", 
                           header=None)
    column_names = ['Token','POS']
    pos_data.columns = column_names
    
    merged = ner_data.merge(pos_data['POS'], left_index=True, right_index=True)
    
    return merged


def sentence_marker(dataframe):
    # Group tokens/words together and mark which belong to the same sentence
    sentence_no = 0
    sentences = []
    
    for index, row in dataframe.iterrows():
        sentences.append(sentence_no)
        if row['Token'] == '.':
            sentence_no += 1
            
    dataframe['Sentence no.'] = sentences


def clean_data(dataframe):
    # Remove rows containing nan values, if there are any
    rows_with_nan = [index for index, row in dataframe.iterrows() 
                     if row.isnull().any()]
    
    for i in rows_with_nan[::-1]:
        dataframe = dataframe.drop(i)


def write_data(dataframe, filename):
    # Write dataframe to tsv file
    try:
        dataframe.to_csv(filename, sep='\t', index=False, header=True)
    except:
        return None


if __name__ == "__main__":
    dataframe = read_data()
    sentence_marker(dataframe)
    clean_data(dataframe)
    write_data(dataframe, 'merged_mim_gold_corpus.tsv')