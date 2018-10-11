import numpy as np

'''
An implementation of the "ERM for Decision Stumps" algorithm 
that appears in the book "Understanding Machine Learning".

- We assume that the training matrix X is given as a m times d + 1 matrix.
    Where d is the dimension of the data and m is the size of
    training set. The last collumn corresponds to the correct classifier.
    Hence the last column consists of 1's and -1's.

- D is a probability vector. This is initialised by AdaBoost and 
    the modified each each iteration by AdaBoost
'''


class DecisionStump(object):
    def __init__(self):
        self.theta = 0
        self.j     = 0

    def calculateDecisionStump(self,X,D):
        Fstar = np.inf
        m,dp1 = X.shape
        d = dp1 - 1
        for j in range(d):
# append a row of the data matrix that 
# is equal to the last row +1 and with 
# the last coordinate set to zero 
            X = np.vstack([X,X[-1]+1]); X[-1,-1] = 0;
# sort the the m rows X using the jth coordinate 
            X = X[X[:j].argsort()]
# get last column of X 
            y = X[:,-1][:-1]
# obtain the sum of the components of the 
            ysum = (y==1).astype(int)
            F = D.dot(ysum)
# we apply the trick oulined on page 104
            if(F<Fstar):
                Fstar = F; self.theta = X[0,j]-1; self.j = j;
            for i in range(m):
                F = F-y[i]*D[i];
                if(F < Fstar and X[i,j]!=X[i+1,j]):
                    Fstar = F
                    self.theta = 0.5*(X[i,j]+X[i+1,j])
                    self.j = j
 
