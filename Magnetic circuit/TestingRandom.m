function [Outs]=TestingRandom(name,numExp)
levels=2;
Problem=redjson(name);
Variables=fieldnames(Problem);
[N,s]=size(Variables);
Casos(1)=struct('name','none','value','none');
ListDV=[];
typesVAR=[];
label=[];
index=1;
for i =1:N
    
    tipo=Problem.(Variables{i}).Type;
    %Problem.(Variables{i})
    if strcmp(tipo,'Continuo')
        minimo=Problem.(Variables{i}).min;
        maximo=Problem.(Variables{i}).max;
        ListDV=[ListDV,i];
        typesVAR=[typesVAR,1];
        Casos(end +1)=struct('name',Variables{i},'value',minimo:(maximo-minimo)/(levels-1):maximo);
    elseif strcmp(tipo,'ID')
        grid = fieldnames(Problem.(Variables{i}).Table);
        [z,w]=size(grid);
        [f,g]=size(Problem.(Variables{i}).set);
        DLV=1:(f-1)/(f-1):f;
        DLV;
        for j=1:z
            ListDV=[ListDV,i];
            typesVAR=[typesVAR,2];
            Casos(end +1)=struct('name',grid{j},'value',Problem.(Variables{i}).Table.(grid{j})(int8(DLV)));
        end
    elseif strcmp(tipo, 'DiscreteID')
        minimo=Problem.(Variables{i}).min;
        maximo=Problem.(Variables{i}).max;
        Dsclim=minimo:1:maximo;
        eval([Variables{i},'= Dsclim']);
        [z,w]=size(Problem.(Variables{i}).Equation);
        for j=1:z  
            ListDV=[ListDV,i];
            typesVAR=[typesVAR,3];
            Casos(end +1)=struct('name',Problem.(Variables{i}).Equation(j).name,'value',eval(Problem.(Variables{i}).Equation(j).eq));
        end
    end
end
[z,w]=size(ListDV);
Casos(1)=[]; 

Outs=zeros(1,w);
for i=1:numExp
    row=[];
    ref=0;
    for k =1:w    
           if typesVAR(k)==1
              m = Casos(k).value(2)- Casos(k).value(1);
              b = Casos(k).value(2)-m;
              y=m*rand+b;      
            elseif typesVAR(k)==2
              if ref~=ListDV(k)
                 ref=ListDV(k)
                [nd,max]=size(Casos(k).value);
                 nr=randi([1,nd]);
              end
              y=Casos(k).value(nr);
            elseif typesVAR(k)==3
              [nd,max]=size(Casos(k).value);
              nr=randi([1,max]);
              y=Casos(k).value(nr);     
           end
        
        row=[row,y];
    end
            
    Outs=[Outs;row];
end
    
Outs(1,:)=[];
