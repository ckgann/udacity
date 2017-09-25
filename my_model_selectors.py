import math
import statistics
import warnings

import numpy as np
from hmmlearn.hmm import GaussianHMM
from sklearn.model_selection import KFold
from asl_utils import combine_sequences


class ModelSelector(object):
    '''
    base class for model selection (strategy design pattern)
    '''

    def __init__(self, all_word_sequences: dict, all_word_Xlengths: dict, this_word: str,
                 n_constant=3,
                 min_n_components=2, max_n_components=10,
                 random_state=14, verbose=False):
        self.words = all_word_sequences
        self.hwords = all_word_Xlengths
        self.sequences = all_word_sequences[this_word]
        self.X, self.lengths = all_word_Xlengths[this_word]
        self.this_word = this_word
        self.n_constant = n_constant
        self.min_n_components = min_n_components
        self.max_n_components = max_n_components
        self.random_state = random_state
        self.verbose = verbose

    def select(self):
        raise NotImplementedError

    def base_model(self, num_states):
        # with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=DeprecationWarning)
        # warnings.filterwarnings("ignore", category=RuntimeWarning)
        try:
            hmm_model = GaussianHMM(n_components=num_states, covariance_type="diag", n_iter=1000,
                                    random_state=self.random_state, verbose=False).fit(self.X, self.lengths)
            if self.verbose:
                print("model created for {} with {} states".format(self.this_word, num_states))
            return hmm_model
        except:
            if self.verbose:
                print("failure on {} with {} states".format(self.this_word, num_states))
            return None


class SelectorConstant(ModelSelector):
    """ select the model with value self.n_constant

    """

    def select(self):
        """ select based on n_constant value

        :return: GaussianHMM object
        """
        best_num_components = self.n_constant
        return self.base_model(best_num_components)


class SelectorBIC(ModelSelector):
    """ select the model with the lowest Bayesian Information Criterion(BIC) score

    http://www2.imm.dtu.dk/courses/02433/doc/ch6_slides.pdf
    Bayesian information criteria: BIC = -2 * logL + p * logN
    """

    def select(self):
        """ select the best model for self.this_word based on
        BIC score for n between self.min_n_components and self.max_n_components

        :return: GaussianHMM object
        """
        warnings.filterwarnings("ignore", category=DeprecationWarning)

        # TODO implement model selection based on BIC scores
        
        bic_score = []
        n_range = range(self.min_n_components, self.max_n_components+1)
        for num_states in n_range:
            hmm_model = GaussianHMM(n_components=num_states, covariance_type="diag", n_iter=1000,
                                    random_state=self.random_state, verbose=False).fit(self.X, self.lengths)
            logL = hmm_model.score(X, self.lengths)
            p1 =  num_states*(num_states-1)
            p2 = num_states-1
            p3 = num_states*self.n_features
            p4 = num_states*self.n_features
            p = p1 + p2 + p3 + p4
            n = len(self._generate_sample_from_state(self.random_state))
            BIC = -2 * logL + p * np.log(n) 
            bic_score.append(BIC)
        
        min_value = min(bic_score)
        best_n = bic_score.index(min_value)
        return GaussianHMM(n_components=best_n, covariance_type="diag", n_iter=1000,
                                    random_state=self.random_state, verbose=False).fit(self.X, self.lengths)
     

        
  
        raise NotImplementedError


class SelectorDIC(ModelSelector):
    ''' select best model based on Discriminative Information Criterion

    Biem, Alain. "A model selection criterion for classification: Application to hmm topology optimization."
    Document Analysis and Recognition, 2003. Proceedings. Seventh International Conference on. IEEE, 2003.
    http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.58.6208&rep=rep1&type=pdf
    https://pdfs.semanticscholar.org/ed3d/7c4a5f607201f3848d4c02dd9ba17c791fc2.pdf
    DIC = log(P(X(i)) - 1/(M-1)SUM(log(P(X(all but i))
    '''

    def select(self):
        warnings.filterwarnings("ignore", category=DeprecationWarning)

        # TODO implement model selection based on DIC scores
        raise NotImplementedError


class SelectorCV(ModelSelector):
    ''' select best model based on average log Likelihood of cross-validation folds
            model = SelectorCV(sequences, Xlengths, word, 
                    min_n_components=2, max_n_components=15, random_state = 14).select()
            model = GaussianHMM(n_components=num_states, covariance_type="diag", n_iter=1000,
                    random_state=self.random_state, verbose=False).fit(self.X, self.lengths)
    '''
     
    
    
    
    
    
    def select(self):
        warnings.filterwarnings("ignore", category=DeprecationWarning)
        split_method = KFold()

        # TODO implement model selection using CV
        # with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=RuntimeWarning)
        # warnings.filterwarnings("ignore", category=RuntimeWarning)
        try:          
            cv_score = []
            n_range = range(self.min_n_components, self.max_n_components+1)
            split_method = KFold()

            for num_states in n_range:
            
                for cv_train_idx, cv_test_idx in split_method.split(self.sequences):
                    logscores = []
                    training = asl.build_training(cv_train_idx)  
                    X, lengths = training.get_word_Xlengths(word)
                    train_combined = combine_sequence(cv_train_idx, self.sequences)
                    model = GaussianHMM(n_components=num_states, n_iter=1000).fit(train_combined, lengths)
                    logscores.append(model.score(combine_sequence(cv_train_idx, self.sequences), lengths))

                cv_score.append(np.mean(logscores))
        
            min_value = min(cv_score)
            best_n = cv_score.index(min_value)
            return GaussianHMM(n_components=best_n).fit(self.X, self.lengths)     
        
        if self.verbose:
                print("model created for {} with {} states".format(self.this_word, num_states))
            return hmm_model
        except:
            if self.verbose:
                print("failure on {} with {} states".format(self.this_word, num_states))
            return None
