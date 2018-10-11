import numpy as np
import DecisionStump as ds

# m is the number of data points
# d is the dimension of the data
m = 100;d = 5


# generate some data
data = np.random.rand(m,d)
X = np.zeros((m,d+1))
X[:,:-1] = data
X[:,-1] = np.random.randint(2,size=100)

D = np.ones()

stumps = ds.DecisionStump()
stumps.calculateDecisionStump(X,)

