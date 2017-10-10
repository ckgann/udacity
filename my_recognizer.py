import warnings
from asl_data import SinglesData


def recognize(models: dict, test_set: SinglesData):
    """ Recognize test word sequences from word models set

   :param models: dict of trained models
       {'SOMEWORD': GaussianHMM model object, 'SOMEOTHERWORD': GaussianHMM model object, ...}
   :param test_set: SinglesData object
   :return: (list, list)  as probabilities, guesses
       both lists are ordered by the test set word_id
       probabilities is a list of dictionaries where each key a word and value is Log Liklihood
           [{SOMEWORD': LogLvalue, 'SOMEOTHERWORD' LogLvalue, ... },
            {SOMEWORD': LogLvalue, 'SOMEOTHERWORD' LogLvalue, ... },
            ]
       guesses is a list of the best guess words ordered by the test set word_id
           ['WORDGUESS0', 'WORDGUESS1', 'WORDGUESS2',...]
   """
    warnings.filterwarnings("ignore", category=DeprecationWarning)
    import operator
    probabilities = []
    guesses = []
    # TODO implement the recognizer
    # return probabilities, guesses
    X_all = test_set.get_all_sequences
    lengths_all = test_set.get_all_Xlengths
    words = []
    ddict = {}
   
    for word_id in range(0,len(test_set.get_all_Xlengths())):
        X, lengths = test_set.get_all_Xlengths()[word_id]          
        for word, model in models.items():
            try:
                LogLvalue = model.score(X, lengths)
                ddict[word] = logLvalue               
                words.append(word_id)
                
            except:
                ddict[word] = np.NINF
                words.append(word_id)
                pass
            
            probabilities.append(ddict)           
            
            
    for dictionary in probabilities:
        guesses.append(max(dictionary.iteritems(), key=operator.itemgetter(1))[0]
        
    print('guesses',len(guesses),guesses)
                       
    return (probabilities,guesses)













