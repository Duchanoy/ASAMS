Exfunction [Expout,Experimments,bestScore,FinalScore]=Model_trainig(name,levels,problem,Numout,hashCell,file,KeepRate,NumExpe,iterations,maxExp,TargetScore)
fprintf('Start Experiment Design\n')
[Cases,Labels,Experimments]=ExperimentDesign(name,levels);
[Experimments,Expout]=experimentRun(Labels,Experimments,problem,Numout,hashCell);
[TRS,vnum]=size(Expout);
TES=((TRS*100)/80)-TRS;
Test=TestingRandom(name,TES);
fprintf('Evaluating new points\n')
[Test,Testout]=experimentRun(Labels,Test,problem,Numout,hashCell);
save(strcat(file,'train','.mat'),'Experimments','Expout')
save(strcat(file,'test','.mat'),'Test','Testout')
exp=py.numpy.array(Experimments);
expOut=py.numpy.array(Expout);
fprintf('Start iteration %d \n',1)
disp('Starting Meta-Learning')
models=py.AutoML_SM.AutoSB(exp,expOut);
LogicVal=py.AutoML_SM.stopCondition(models,TargetScore);
counter=1;
IteratioNum=py.int(counter);
disp('Saving data log ')
py.AutoML_SM.LogRegister(name,models,exp,expOut,IteratioNum);
while(LogicVal)
    fprintf('\n\n\n_____________________________________\n')
    fprintf('Start iteration %d \n',counter+1)
    disp('Reduce models')
    gridParm=py.AutoML_SM.modelReducedFiltering(models,KeepRate);
    disp('Sampling new points')
    nExp=py.int(NumExpe);
    samplePoints=py.AutoML_SM.AdaptativeSampling(models,exp,expOut,nExp);
    newPoint=double(samplePoints);
    disp('Evaluating new points')
    [newPoint,NewExpout]=experimentRun(Labels,newPoint,problem,Numout,hashCell);
    Experimments=[Experimments;newPoint];
    Expout=[Expout;NewExpout];
    exp=py.numpy.array( Experimments);
    expOut=py.numpy.array(Expout);
    disp('Training model')
    models=py.AutoML_SM.ModelReduceSelection(exp,expOut,models,gridParm);
    counter=counter+1;
    IteratioNum=py.int(counter);
    disp('Saving data log ')
    py.AutoML_SM.LogRegister(name,models,exp,expOut,IteratioNum);
    nex=size(Expout);
    %number of iteratios stop condiction 
    if counter>=iterations
        LogicVal=false;
    %number of experiments stop condiction
    elseif nex(1)>=maxExp  
        nex=size(Expout);
        LogicVal=false;
    else
        LogicVal=py.AutoML_SM.stopCondition(models,TargetScore);
    end
end
bestScore = py.AutoML_SM.BestModel(models,name);
FinalScore = py.AutoML_SM.modelEvaluation(moldels,Test,Testout);
fprintf('TestDataScore %f \n',FinalScore)
