#!/usr/bin/env python
# coding: utf-8

# In[1]:


def TestData(data):
    print(type(data))
    print(data)
    castdata=list(data)
    print(type(castdata))
    print(castdata)
    return False
    


# In[ ]:


def AutoSBTest( inputs_train,outputs_train):
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
    param_grid_SVM = { 'kernel':['rbf'],
                  'C':[100],
                  'gamma':[1]}
    gridSVM = GridSearchCV(model_SVM,param_grid_SVM,refit=True,verbose=0,scoring='neg_mean_squared_error')
    gridSVM.fit(inputs_train,outputs_train)
    print(gridSVM.best_params_)
    print(gridSVM.best_score_)
    
    model_RFR=RandomForestRegressor()
    param_grid_RFR = { 'n_estimators':[200],
                       'max_features':['auto'],
                       'max_depth': [10],
                       'min_samples_leaf': [1],
                       'min_samples_split': [2]}
    
    gridRFR = GridSearchCV(model_RFR,param_grid_RFR,refit=True,verbose=0,scoring='neg_mean_squared_error')
    gridRFR.fit(inputs_train,outputs_train)
    print(gridRFR.best_params_)
    print(gridRFR.best_score_)
          
    model_BR= BayesianRidge(n_iter=500)
    param_grid_BR = {'alpha_1':[1e-4],
                     'alpha_2':[1e-4],
                     'lambda_1':[1e-4],
                     'lambda_2':[1e-4]}
    gridBR= GridSearchCV(model_BR,param_grid_BR,refit=True,verbose=0,scoring='neg_mean_squared_error')
    gridBR.fit(inputs_train,outputs_train)
    print(gridBR.best_params_)
    print(gridBR.best_score_)     

          
    return  gridSVM,gridRFR,gridBR


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
    gridSVM = GridSearchCV(model_SVM,param_grid_SVM,refit=True,verbose=0,scoring='neg_mean_squared_error')
    gridSVM.fit(inputs_train,outputs_train)
    print(gridSVM.best_params_)
    print(gridSVM.best_score_)
    
    model_RFR=RandomForestRegressor()
    param_grid_RFR = { 'n_estimators':[200, 600, 1000, 1400, 1800],
                       'max_features':['auto', 'log2'],
                       'max_depth': [10, 50, 90, None],
                       'min_samples_leaf': [1, 2, 4],
                       'min_samples_split': [2, 5, 10]}
    
    gridRFR = GridSearchCV(model_RFR,param_grid_RFR,refit=True,verbose=0,scoring='neg_mean_squared_error')
    gridRFR.fit(inputs_train,outputs_train)
    print(gridRFR.best_params_)
    print(gridRFR.best_score_)
          
    model_BR= BayesianRidge(n_iter=500)
    param_grid_BR = {'alpha_1':[1e-4,1e-6,1e-7],
                     'alpha_2':[1e-4,1e-6,1e-7],
                     'lambda_1':[1e-4,1e-6,1e-7],
                     'lambda_2':[1e-4,1e-6,1e-7]}
    gridBR= GridSearchCV(model_BR,param_grid_BR,refit=True,verbose=0,scoring='neg_mean_squared_error')
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
    return LScore


# In[ ]:


def modelReducedFiltering(modelList,keepRate):
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
    
    gridList=list()
    for model in modelList:
        parOut=dict()
        jsonModelSVM=model.cv_results_
        for key in list(jsonModelSVM['params'][0].keys()):
            parOut[key]=list()
        mdlScores=jsonModelSVM["std_test_score"]
        mdlSOrder=np.argsort(mdlScores)
        NumMdl=round(len(mdlScores)*keepRate)
        minnum=max([NumMdl,1])
        SubmdlScores=mdlSOrder[:minnum]
        for case in SubmdlScores:
            dictpar=jsonModelSVM['params'][case]
            for key in list(dictpar.keys()):
                parOut[key].append(dictpar[key])
        for key in list(parOut.keys()):     
            parOut[key]=list(set(parOut[key]))
        gridList.append(parOut)
    return gridList


# In[ ]:


def AdaptativeSampling(modelList,Experimentos,Salidas,NumExpe):
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
    from sklearn.model_selection import cross_val_score
    
    [Ne,Dim]=Experimentos.shape
    scoreList=list()
    for model in modelList:
        scoresMld =cross_val_score(model.best_estimator_, Experimentos,Salidas, cv=Ne,scoring='neg_mean_squared_error')
        scoreList.append(scoresMld)
    scoreArray=np.array(scoreList)
    scores=np.mean(scoreArray,axis=0)
    vor = Voronoi(Experimentos,qhull_options="QJ")
    LenVertex=len(vor.vertices)
    minlimit=Experimentos.min(axis=0)
    maxlimit=Experimentos.max(axis=0)
    usefulVertex=list()
    usefulVerIndex=list()
    for verIndex,vertex in enumerate(vor.vertices):
        flag=True
        for index,parm in enumerate(vertex):
            if parm<minlimit[index] or parm>maxlimit[index]:
                flag=False
        if flag:
            usefulVertex.append(vertex)
            usefulVerIndex.append(verIndex)
    #print(usefulVertex)
    VoroniScoresExp=[]
    VoroniScores=[]
    for i,vertex in enumerate(vor.vertices):
        vetexscore=[]
        if i in usefulVerIndex:
            for j, case in enumerate(vor.regions):
                if i in case:
                    vetexscore.append(scores[j])
        else:
            vetexscore.append(-1000)
        
        VoroniScores.append(np.mean(np.array(vetexscore)))
        VoroniScoresExp.append(vetexscore)
    #print(np.array(VoroniScores).shape)    
    point_selection_order=np.argsort(np.array(VoroniScores))
    point_selValues=np.sort(np.array(VoroniScores))
    #print(np.sort(np.array(VoroniScores)*-1))
    #print(point_selection_order)
    New_points=vor.vertices[point_selection_order[:NumExpe]]
    return New_points


# In[ ]:


def stopCondition(modelList,TargetScore):
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
    
    performance = list()
    for model in modelList:
        score=model.best_score_
        performance.append(score)
    BestModel=np.array(performance).argmax()
    if performance[BestModel]>TargetScore:
        return False
    else:
        return True


# In[ ]:


def ModelReduceSelection(inputs_train,outputs_train,ModelList,ParamGridList):
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
    model_RFR=RandomForestRegressor()
    model_BR= BayesianRidge(n_iter=500)
    modelType=[model_SVM,model_RFR,model_BR]
    outModels=list()
    for index,model in enumerate(ModelList):
        #modelB=model.best_estimator_
        param_grid = ParamGridList[index]
        #print()
        grid = GridSearchCV(modelType[index],param_grid,refit=True,verbose=0,scoring='neg_mean_squared_error')
        grid.fit(inputs_train,outputs_train)
        outModels.append(grid)
    return outModels 


# In[ ]:


def BestModel(modelList,name):
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
    
    
    performance = list()
    for model in modelList:
        score=model.best_score_
        performance.append(score)
    BestModel=np.array(performance).argmax()
    dump(modelList[BestModel],name+'.joblib')
    return performance[BestModel]


# In[ ]:


def LogRegister(jsonName,modelList,Experiments,ExpOuts,IteratioNum):
    import json
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
    
    interjson=dict()
    try:
        
        log=json.load(open(jsonName+"_log.json"))
        
    except:
        log=dict()
    
    
    interjson["models"]=list()
    interjson["modelsScore"]=list()
    for model in modelList:
        modelRepot=dict()
        modelRepot["ModelsScore"]=model.cv_results_['mean_test_score'].tolist()
        modelRepot["ModelsParams"]=model.cv_results_['params']
        modelRepot["BestScore"]=model.best_score_
        modelRepot["BestParams"]=model.best_params_
        interjson["models"].append(modelRepot)    
        interjson["modelsScore"].append(model.best_score_)
        
    
    interjson["PointNumber"]=len(Experiments)
    interjson["Experiments"]=Experiments.tolist()
    interjson["Outs"]=ExpOuts.tolist()
    interjson["Iteration"]=IteratioNum
    log.update({str(IteratioNum):interjson})
    MaxScore=np.array(interjson["modelsScore"]).max()
    MaxArgmax=np.array(interjson["modelsScore"]).argmax()
    bestModel=interjson["models"][MaxArgmax]["BestParams"]
    print("--------------------------------------------------------")
    print("Iteration {}".format(IteratioNum))
    print("Best model {}".format(bestModel))
    print("Best model score  {}".format(MaxScore))
    print("Number of experiments {}".format(len(Experiments)))
    s = json.dumps(log)
    open(jsonName+"_log.json","w").write(s)

