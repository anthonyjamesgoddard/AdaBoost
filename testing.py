import numpy as np
import DecisionStump as ds

# n is the number of data points
# d is the dimension of the data
# m is the number of points in our testing set
n = 100;d = 10
m = 20


# generate some data
data = np.random.rand(n,d)
X = np.zeros((n,d+1))
X[:,:-1] = data
X[:,-1] = np.random.randint(2,size=100)

D = np.ones(m)*(1.0/m)

stumps = ds.DecisionStump()
stumps.calculateDecisionStump(X[:m,:],D)
print stumps.j
