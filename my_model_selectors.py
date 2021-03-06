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
        
        
        bic_score = np.inf
        n_range = range(self.min_n_components, self.max_n_components+1)

        try:
            for num_states in n_range:
                try:
                    #print('num_states=',num_states)
                    model = self.base_model(num_states)
                    #print('transmat_',model.transmat_)
                    if np.round(model.transmat_.sum()) == model.transmat_.shape[0]:                  
                        #print('if stmt triggered****')
                        logL = model.score(self.X, self.lengths)
                        #print('logL',logL)
                        p1 =  num_states*(num_states-1)
                        p2 = num_states-1
                        #print('p2', p2)
                        p3 = num_states*model.n_features
                        p4 = num_states*model.n_features
                        p = p1 + p2 + p3 + p4
                        #print('ps', p, p1, p2, p3, p4)
                        n = len(self.X)
                        #print('len(x)',len(self.X))
                        BIC = -2 * logL + p * np.log(n) 
                        #print('BIC',BIC)
                        if BIC < bic_score:
                            bic_score , best_n = BIC , num_states


                except:
                    if self.verbose:
                        print("failure on {} with {} states".format(self.this_word, num_states))


            return GaussianHMM(n_components=best_n, covariance_type="diag", n_iter=1000,
                                        random_state=self.random_state, verbose=False).fit(self.X, self.lengths)
        except:
            return None

        
  
        #raise NotImplementedError


class SelectorDIC(ModelSelector):
    ''' select best model based on Discriminative Information Criterion
    
        self.words = all_word_sequences
        self.hwords = all_word_Xlengths
        self.sequences = all_word_sequences[this_word]
        self.X, self.lengths = all_word_Xlengths[this_word]
        self.this_word = this_word

    Biem, Alain. "A model selection criterion for classification: Application to hmm topology optimization."
    Document Analysis and Recognition, 2003. Proceedings. Seventh International Conference on. IEEE, 2003.
    http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.58.6208&rep=rep1&type=pdf
    https://pdfs.semanticscholar.org/ed3d/7c4a5f607201f3848d4c02dd9ba17c791fc2.pdf
    DIC = log(P(X(i)) - 1/(M-1)SUM(log(P(X(all but i))
    '''

    def select(self):
        warnings.filterwarnings("ignore", category=DeprecationWarning)

        # TODO implement model selection based on DIC scores
        
        dic_score = np.NINF
        n_range = range(self.min_n_components, self.max_n_components+1)
        j_scores = []
        i_scores = []
          
        for num_states in n_range:

            try:
                model = self.base_model(num_states)                
                X, lengths = self.hwords[self.this_word]
                logL_i = model.score(X, lengths)
                #print('logL_i',logL_i,'numstates',num_states,'word',self.this_word)               
                logL_j_other = []
                
                for word in self.words:                      
                    if word != self.this_word:
                        X, lengths = self.hwords[word]                   
                        try:                      
                            logL_j_other.append(model.score(X, lengths))
                        except:
                            #print('j-score failed', num_states, word)
                            pass

                #print('len of logj',len(logL_j_other))
                logL_j = np.mean(logL_j_other)
                DIC = logL_i - logL_j
                j_scores.append(logL_j)
                i_scores.append(logL_i)
                #print(num_states, 'bic-len',len(logL_j_other),'I',logL_i,'J',logL_j,'DIC',DIC)
                
                if DIC > dic_score:
                    dic_score , best_n = DIC , num_states
                    #print('numstates',num_states,'best_n',best_n,'dic_score',dic_score)
            except:
                pass
                #print('model fit failed for', num_states)
            
        #for i in range(len(i_scores)):           
        #    print('results','i=',i_scores[i],'j=',j_scores[i],'diff=',i_scores[i]-j_scores[i])
            
        return GaussianHMM(n_components=best_n, covariance_type="diag", n_iter=1000,
                                    random_state=self.random_state, verbose=False).fit(self.X, self.lengths)
     
        #raise NotImplementedError


class SelectorCV(ModelSelector):
    ''' select best model based on average log Likelihood of cross-validation folds
            model = SelectorCV(sequences, Xlengths, word, 
                    min_n_components=2, max_n_components=15, random_state = 14).select()
            model = GaussianHMM(n_components=num_states, covariance_type="diag", n_iter=1000,
                    random_state=self.random_state, verbose=False).fit(self.X, self.lengths)
    # TODO implement model selection using CV
        # with warnings.catch_warnings():
       
        # warnings.filterwarnings("ignore", category=RuntimeWarning)
                    #warnings.filterwarnings("ignore", category=DeprecationWarning)

    '''
    
    def select(self):
        warnings.filterwarnings("ignore", category=DeprecationWarning)
        warnings.filterwarnings("ignore", category=RuntimeWarning)
        #print('min/max',self.min_n_components, self.max_n_components)
        #return
 
            #cv_score = []
        best_n=0
        CV = np.PINF
        

        try:
            split_method = KFold(n_splits=min(3,len(self.sequences)))
            n_range = range(self.min_n_components, self.max_n_components+1)
            for num_states in n_range:
                cv_score = []
                try:
                    for cv_train_idx, cv_test_idx in split_method.split(self.sequences):
                        train_x, train_legnths = combine_sequences(cv_train_idx, self.sequences)
                        test_x, test_legnths = combine_sequences(cv_test_idx, self.sequences)
                        #print('p1','train_legnths=',len(train_legnths),train_legnths, 'num_states=',num_states)  
                        if train_legnths[0] > num_states:   
                            #print('p2','enough samples',train_legnths[0] ,num_states)
                            try:
                                model = GaussianHMM(n_components=num_states, n_iter=1000).fit(train_x, train_legnths)
                                if model.transmat_.sum() == model.transmat_.shape[0]: 
                                    #print('p3','good tmat','score=',model.score(test_x, test_legnths))
                                    try:
                                        cv_score.append(model.score(test_x, test_legnths))
                                        #print('p4','cvscore=',cv_score)
                                    except:
                                        pass
                                        #print('p5','score failed')                        

                            except:
                                pass
                                #print('p6','fit model failed',    train_legnths[0], num_states)  


                    if np.mean(cv_score) < CV :
                        CV, best_n = np.mean(cv_score), num_states                             

                except:
                    if self.verbose:
                        print('p7',"failure on {} with {} states".format(self.this_word, num_states)) 
            
        except:
            return None
        #print('p8',best_n, CV)
        return GaussianHMM(n_components=best_n).fit(self.X, self.lengths)     
        
  #          if self.verbose:
  #              print("model created for {} with {} states".format(self.this_word, num_states))
  #              return hmm_model
  #      except Exception as e:
  #          print(e)
  #          import traceback
  #          if self.verbose:
  #              #print("failure on {} with {} states".format(self.this_word, num_states))               
  #              print("".join(traceback.format_exception(etype=type(e),value=e,tb=e.__traceback__)))

    #        return None
