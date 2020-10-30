import os
import pandas as pd


def read_data(path, verbose=False):
    dataframe = pd.DataFrame()
    if verbose:
        print('Running read_data...')
    
    # Iterate through files in MIM GOLD NER and MIM GOLD 1.0 and merge them
    try:
        files = [f for f in os.listdir(path)]
        # files = [os.path.join(path, 'blog.txt')]
        for file in files:
            if verbose:
                print('Reading {}...'.format(file))
            ner_data = pd.read_csv(os.path.join(path, file), 
                                   sep="\t", 
                                   header=None)
            
            column_names = ['Token','Tag']
            ner_data.columns = column_names
            
            dataframe = pd.concat([dataframe, ner_data[['Token', 'Tag']]])
        
        if verbose:
            print('Done')
        
        return dataframe.reset_index()
    except FileNotFoundError:
        return None


def sentence_marker(dataframe, verbose=False):
    out = dataframe.copy()
    if verbose:
        print('Running sentence_marker...')
    # Group tokens/words together and mark which belong to the same sentence
    sentence_no = 0
    sentences = []
    
    for index, row in out.iterrows():
        sentences.append(sentence_no)
        if row['Token'] == '.':
            sentence_no += 1
            
    out['Sentence no.'] = sentences
    if verbose:
        print('Done')
        
    out = out.drop(columns=['index'])
        
    return out


def clean_data(dataframe, verbose=False):
    out = dataframe.copy()
    if verbose:
        print('Running clean_data...')
        
    # Remove rows containing nan values, if there are any
    rows_with_nan = [index for index, row in out.iterrows() 
                     if row.isnull().any()]
    
    if verbose:
        print('Number of rows containing nan values: {}'.format(len(rows_with_nan)))
    
    for i in rows_with_nan[::-1]:
        out = out.drop(i)
        
    if verbose:
        print('Done')
        
    return out


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