#!/usr/bin/env python
# coding: utf-8

# In[3]:


from sklearn.neural_network import MLPRegressor
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import BayesianRidge, LinearRegression
from sklearn.model_selection import GridSearchCV
from sklearn.externals import joblib
import scipy.io


# ## Auto adjusting Subrogated model 

# In[4]:


def AutoSB( inputs_train,outputs_train):
    # firstmodel
    inputs_train=list(inputs_train)
    outputs_train=list(outputs_train)
    model_SVM=SVR()
    param_grid = { 'kernel':['linear','rbf'],
                  'C':[0.01,0.1,1,10,100],
                  'gamma':[1,10,100,1000,10000,'auto']}
    grid = GridSearchCV(model_SVM,param_grid,refit=True,verbose=3)
    grid.fit(inputs_train,outputs_train)
    print(grid.best_params_)
    print(grid.best_score_)
    
    return  grid.best_estimator_
    
    
    


# In[5]:


#mat = scipy.io.loadmat('experimentosMC.mat') 


# In[6]:


#modelo=AutoSB( mat['Experiments'],mat['Expout'])


# In[7]:


#modelo


# In[ ]:




