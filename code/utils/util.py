import os
import pandas as pd


MIM_GOLD_NER_PATH = '../data/MIM-GOLD-NER/'
FILES = [f for f in os.listdir(MIM_GOLD_NER_PATH)]


def read_data(verbose=False):
    dataframe = pd.DataFrame()
    if verbose:
        print('Running read_data...')
    
    # Iterate through files in MIM GOLD NER and MIM GOLD 1.0 and merge them
    try:
        for file in FILES:
            if verbose:
                print('Reading {}...'.format(file))
                print(os.path.join(MIM_GOLD_NER_PATH, file))
            ner_data = pd.read_csv(os.path.join(MIM_GOLD_NER_PATH, file), 
                                   sep="\t", 
                                   header=None)
            
            column_names = ['Token','Tag']
            ner_data.columns = column_names
            
            dataframe = pd.concat([dataframe, ner_data])
        
        return dataframe.reset_index()
    except FileNotFoundError:
        return None


def sentence_marker(dataframe, verbose=False):
    if verbose:
        print('Running sentence_marker...')
    # Group tokens/words together and mark which belong to the same sentence
    sentence_no = 0
    sentences = []
    
    for index, row in dataframe.iterrows():
        sentences.append(sentence_no)
        if row['Token'] == '.':
            sentence_no += 1
            
    dataframe['Sentence no.'] = sentences
    if verbose:
        print('Done')


def clean_data(dataframe, verbose=False):
    if verbose:
        print('Running clean_data...')
        
    # Remove rows containing nan values, if there are any
    rows_with_nan = [index for index, row in dataframe.iterrows() 
                     if row.isnull().any()]
    
    if verbose:
        print('Number of rows containing nan values: {}'.format(len(rows_with_nan)))
    
    for i in rows_with_nan[::-1]:
        dataframe = dataframe.drop(i)


def write_data(dataframe, filename, verbose=False):
    if verbose:
        print('Running write_data...')
    # Write dataframe to tsv file
    try:
        dataframe.to_csv(filename, sep='\t', index=False, header=True)
        if verbose:
            print('Dataframe can be found at: {}'.format(filename))
    except:
        return None