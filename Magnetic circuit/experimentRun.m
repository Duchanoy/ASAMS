function [experiments,Expout]=experimentRun(labels,experiments,problem,Numout,hashCell)
[exp,var]=size(experiments);
Fx=problem;
Expout=zeros(1,Numout);
Error=[];
for i=1:exp
    
    FROW=hashTable(labels,hashCell,experiments(i,:));
    try 
        row=Fx(FROW);
    catch 
        Error=[Error,i];
        disp('Error en ')
        disp(i)
        row=zeros(1,Numout);
        
    end 
    
    Expout=[Expout;row];
end
[nu,var]=size(Error);
Errortag=sort(Error,'descend');
for j =1:var
    experiments(Errortag(j),:)=[];
    Expout(Errortag(j),:)=[];
end
Expout(1)=[];