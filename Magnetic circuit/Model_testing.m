function [Test,Testout]=Model_testing(name,TRS,problem,Numout,hashCell,file)
TES=((TRS*100)/80)-TRS;
Test=TestingRandom(name,TES);
[Testout]=experimentRun(Labels,Test,problem,Numout,hashCell);
save(strcat(file,'test','.mat'),'Test','Testout')