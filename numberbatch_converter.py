import gensim

PATH = 'files/numberbatch-en-17.04b.txt'

if __name__ == '__main__':
    print('Loading the model...')
    try:
        numberbatch = gensim.models.KeyedVectors.load_word2vec_format(PATH, binary=False, unicode_errors='ignore')
    except FileNotFoundError:
        print('Incorrect file path. Please open the code and modify the PATH variable.')

    print('Converting the model...')
    numberbatch.init_sims(replace=True)
    numberbatch.save('conceptNet')
    print('Convertion successful')