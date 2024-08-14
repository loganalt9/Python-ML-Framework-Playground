import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.neural_network import MLPClassifier, MLPRegressor

x = np.random.randint(0,10, (10, 2))
y = np.sum(x,axis=1)

modelR = MLPRegressor(hidden_layer_sizes=20, max_iter=1000)
modelR.fit(x,y)

test = np.array([5,8]).reshape(1,-1)

print (modelR.predict(test))

modelC = MLPClassifier(hidden_layer_sizes=20, max_iter=1000)