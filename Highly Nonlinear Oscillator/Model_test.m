function Model_test(name,levels,problem,Numout,hashCell,file,TES)
fprintf('Start Experiment Design\n')
[Cases,Labels,Experimments]=ExperimentDesign(name,levels);
%[Experimments,Expout]=experimentRun(Labels,Experimments,problem,Numout,hashCell);
%[TRS,vnum]=size(Expout);
%TES=((TRS*100)/80)-TRS;
Test=TestingRandom(name,TES);
fprintf('Evaluating new points\n')
%[Test,Testout]=experimentRun(Labels,Test,problem,Numout,hashCell);
save(strcat(file,'gridTest','.mat'),'Experimments')
save(strcat(file,'test','.mat'),'Test')
save(strcat(file,'Labels','.mat'),'Labels')
%save(strcat(file,'gridTest','.mat'),'Experimments','Expout')
%save(strcat(file,'test','.mat'),'Test','Testout')

%@flujo_magnetico,1,hashCell
%MagnetiCircuit
