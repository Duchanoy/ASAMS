#!/usr/bin/env python
# coding: utf-8

# In[1]:
def TestData(data):
    print(type(data))
    print(data)
    print(data.strides(32,7))


    


# In[2]:


def AutoSB( inputs_train,outputs_train):
    # firstmodel
    from sklearn.neural_network import MLPRegressor
    from sklearn.svm import SVR
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.linear_model import BayesianRidge, LinearRegression
    from sklearn.model_selection import GridSearchCV
    from joblib import dump, load
    import scipy.io
    import numpy as np
    from sklearn.metrics import r2_score
    from sklearn.metrics import mean_squared_error
    from scipy.spatial import Voronoi,voronoi_plot_2d
    import numpy as np
    
    model_SVM=SVR()
    param_grid_SVM = { 'kernel':['linear','rbf'],
                  'C':[0.01,0.1,1,10,100],
                  'gamma':[1,10,100,1000,10000,'auto']}
    gridSVM = GridSearchCV(model_SVM,param_grid_SVM,refit=True,verbose=0)
    gridSVM.fit(inputs_train,outputs_train)
    print(gridSVM.best_params_)
    print(gridSVM.best_score_)
    
    model_RFR=RandomForestRegressor()
    param_grid_RFR = { 'n_estimators':[200, 400, 600, 800, 1000, 1200, 1400, 1600, 1800, 2000],
                       'max_features':['auto', 'sqrt', 'log2'],
                       'max_depth': [10, 20, 30, 40, 50, 60, 70, 80, 90, 100, None],
                       'min_samples_leaf': [1, 2, 4],
                       'min_samples_split': [2, 5, 10]}
    
    gridRFR = GridSearchCV(model_RFR,param_grid_RFR,refit=True,verbose=0)
    gridRFR.fit(inputs_train,outputs_train)
    print(gridRFR.best_params_)
    print(gridRFR.best_score_)
          
    model_BR= BayesianRidge(n_iter=500)
    param_grid_BR = {'alpha_1':[1e-4,1e-5,1e-6,1e-7],
                     'alpha_2':[1e-4,1e-5,1e-6,1e-7],
                     'lambda_1':[1e-4,1e-5,1e-6,1e-7],
                     'lambda_2':[1e-4,1e-5,1e-6,1e-7]}
    gridBR= GridSearchCV(model_BR,param_grid_BR,refit=True,verbose=0)
    gridBR.fit(inputs_train,outputs_train)
    print(gridBR.best_params_)
    print(gridBR.best_score_)     

          
    return  gridSVM,gridRFR,gridBR
    
    
    


# In[3]:


def modelEvaluation(moldelList,Experiments,Expout):
    from sklearn.neural_network import MLPRegressor
    from sklearn.svm import SVR
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.linear_model import BayesianRidge, LinearRegression
    from sklearn.model_selection import GridSearchCV
    from joblib import dump, load
    import scipy.io
    import numpy as np
    from sklearn.metrics import r2_score
    from sklearn.metrics import mean_squared_error
    from scipy.spatial import Voronoi,voronoi_plot_2d
    import numpy as np
    
    LScore=[];
    MSElist = []
    for model in moldelList:
        MSE = mean_squared_error(Expout, model.predict(Experiments), multioutput='raw_values')
        Score = r2_score(model.predict(Experiments),Expout);
        LScore.append(Score);
        MSElist.append(MSE);
    return LScore,MSElist


# In[ ]:


def AdaptativeSampling(model,Experimentos,Salidas,NumExpe):
    from sklearn.neural_network import MLPRegressor
    from sklearn.svm import SVR
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.linear_model import BayesianRidge, LinearRegression
    from sklearn.model_selection import GridSearchCV
    from joblib import dump, load
    import scipy.io
    import numpy as np
    from sklearn.metrics import r2_score
    from sklearn.metrics import mean_squared_error
    from scipy.spatial import Voronoi,voronoi_plot_2d
    import numpy as np
    
    [Ne,Dim]=Experimentos.shape
    scores =cross_val_score(model, Experimentos,Salidas, cv=Ne,scoring='neg_mean_gamma_deviance')
    vor = Voronoi(Experimentos,qhull_options="QJ")
    worst_regions=np.argsort(scores)
    LenVertex=len(vor.vertices)
    VoroniScoresExp=[]
    VoroniScores=[]
    for i,vertex in enumerate(vor.vertices):
        vetexscore=[]
        for j, case in enumerate(vor.regions):
            if i in case:
                vetexscore.append(scores[j])
        VoroniScores.append(np.mean(np.array(vetexscore)))
        VoroniScoresExp.append(vetexscore)
        point_selection_order=np.argsort(np.array(VoroniScores))
        New_points=vor.vertices[:NumExpe,:]
    return New_points,VoroniScores, VoroniScoresExp

