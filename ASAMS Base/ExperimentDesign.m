function [Casos,label,Outs]=ExperimentDesign(name,levels)
Problem=redjson(name);
Variables=fieldnames(Problem);
[N,s]=size(Variables);
Casos(1)=struct('name','none','value','none');
ListDV=[];
label=[];
index=1;
for i =1:N
    
    tipo=Problem.(Variables{i}).Type;
    %Problem.(Variables{i});
    if strcmp(tipo,'Continuo')
        minimo=Problem.(Variables{i}).min;
        maximo=Problem.(Variables{i}).max;
        ListDV=[ListDV,i];
        Casos(end +1)=struct('name',Variables{i},'value',minimo:(maximo-minimo)/(levels-1):maximo);
    elseif strcmp(tipo,'ID')
        grid = fieldnames(Problem.(Variables{i}).Table);
        [z,w]=size(grid);
        [f,g]=size(Problem.(Variables{i}).set);
        discLevels=min(levels,f);
        discLevels;
        DLV=1:(f-1)/(discLevels-1):f;
        DLV;
        for j=1:z
            ListDV=[ListDV,i];
            Casos(end +1)=struct('name',grid{j},'value',Problem.(Variables{i}).Table.(grid{j})(int8(DLV)));
        end
    elseif strcmp(tipo, 'DiscreteID')
        minimo=Problem.(Variables{i}).min;
        maximo=Problem.(Variables{i}).max;
        Dsclim=minimo:(maximo-minimo)/(levels-1):maximo;
        eval([Variables{i},'= Dsclim']);
        [z,w]=size(Problem.(Variables{i}).Equation);
        for j=1:z  
            ListDV=[ListDV,i];
            Casos(end +1)=struct('name',Problem.(Variables{i}).Equation(j).name,'value',eval(Problem.(Variables{i}).Equation(j).eq));
        end
    end
end 
Casos(1)=[];       
        
DOE=[];
for factors=1:N
    DOE=[DOE,levels];
end
dFF=fullfact(DOE);
[z,w]=size(ListDV);
label=[];
for k =1:w
     label=[label,',',Casos(k).name];
end
label=split(label,',');
label(1)=[];
[x,y] = size(dFF);
ou=[];
for h=1:w
    ou=[ou,(Casos(h).value(levels)-Casos(h).value(1))/2+Casos(h).value(1)];
end
%Outs=zeros(1,w);
Outs=ou;
for i=1:x
    row=[];
    for j=1:y
        for k =1:w
            if ListDV(k) == j
                row=[row,Casos(k).value(dFF(i,j))];
            end
        end
    end
    Outs=[Outs;row];
end
%Outs(1,:)=[]; 



