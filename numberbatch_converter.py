import gensim

PATH = 'files/numberbatch-en-17.04b.txt'

if __name__ == '__main__':
    print('Trying to load the model...')

    try:
        numberbatch = gensim.models.KeyedVectors.load_word2vec_format(PATH, binary=False, unicode_errors='ignore')
        print('Converting the model...')
        numberbatch.init_sims(replace=True)
        numberbatch.save('conceptNet')
        print('Convertion successful')
    except FileNotFoundError:
        print('Incorrect file path. Please open the code and modify the PATH variable.')
    except ValueError:
        print('The provided file is not a gensim model.')

    print('Conversion failed.')
