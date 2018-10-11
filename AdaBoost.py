'''
implementation of AdaBoost.

 - D is a probability vector. Initially we assume that
    each sample has an equal probability of being chosen.
    The distributution is then modified each iteration
    to increase the probaility mass on samples which
    the weak classifier gets incorrect and decrease 
    the probability mass on samples which the weak
    classifier gets correct.


'''

class AdaBoost(object):
    def __init__(self,X):
        self.X = X
        self.N = X.shape[0]
        self.D = np.ones(self.N)/self.N
        self.Dt= []

    
