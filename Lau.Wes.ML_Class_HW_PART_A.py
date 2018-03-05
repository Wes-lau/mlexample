
# coding: utf-8

# In[28]:

# Load libraries
import pandas
from pandas.tools.plotting import scatter_matrix
import matplotlib.pyplot as plt
from sklearn import model_selection

url = "http://archive.ics.uci.edu/ml/machine-learning-databases/abalone/abalone.data"
names = ['Sex (M, F, I)','Length', 'Diameter','Height','Whole weight','Shucked weight','Viscera weight','Shell weight','Rings']

dataset = pandas.read_csv(url, names=names)





# In[37]:

dataset.corr()["Rings"]
#Shell weight correlates with age (rings)


# In[105]:

columns = dataset.columns.tolist()
columns = [c for c in columns if c not in ['Sex (M, F, I)', 'Rings']]
target = "Rings"
columns

#Filther out the irrelevant columns


# In[73]:

from sklearn.model_selection  import train_test_split
train = dataset.sample(frac=0.8, random_state=1)
test = dataset.loc[~dataset.index.isin(train.index)]

print(train.shape)
print(test.shape)
#splits the training and test into 80/20


# In[104]:

from sklearn.linear_model import LinearRegression
model = LinearRegression()
model.fit(train[columns], train[target])

from sklearn.metrics import mean_squared_error


predictions = model.predict(test[columns])


mean_squared_error(predictions, test[target])
#MSE IS TOO HIGH


# In[107]:

from sklearn.ensemble import RandomForestRegressor

model = RandomForestRegressor(n_estimators=100, min_samples_leaf=10, random_state=1)

model.fit(train[columns], train[target])

predictions = model.predict(test[columns])

mean_squared_error(predictions, test[target])

#MSE is a little lower


# In[109]:

from sklearn.linear_model import LogisticRegression
#Assumed you have, X (predictor) and Y (target) for training data set and x_test(predictor) of test_dataset
# Create logistic regression object
model = LogisticRegression()
# Train the model using the training sets and check score
model.fit(train[columns], train[target])
model.score(train[columns], train[target])
#Equation coefficient and Intercept
print('Coefficient: \n', model.coef_)
print('Intercept: \n', model.intercept_)
#Predict Output
predicted= model.predict(test[columns])
mean_squared_error(predictions, test[target])


# In[ ]:




# In[106]:

from sklearn.cluster import KMeans
#Assumed you have, X (attributes) for training data set and x_test(attributes) of test_dataset
# Create KNeighbors classifier object model 
k_means = KMeans(n_clusters=3, random_state=0)
# Train the model using the training sets and check score
model.fit(train[columns], train[target])
#Predict Output
predictions = model.predict(test[columns])
mean_squared_error(predictions, test[target])

