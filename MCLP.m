tic;                %tic-toc to compute elasped execution time of program
MCLP_GA(20,5);      % Calling the Function
toc;
% Maximal Coverage Linkage Problem, with Local Refinement using Genetic Algorithm
% Inputs:- P: Size of population of mating pool, K: Number of facilities opened (No. of strings in a chromosome)
% Encoding scheme: string
% Selection: Binary Tournament
% Crossover scheme: Random Single-point crossover with crossover probability 0.9
% Mutation scheme:  Element to be mutated is replaced randomly from  N%
% nearest neighbour, here N=5 chosen, with mutation probability=0.01 if result not saturated for 50 generations,
% else mutation probability =0.8
% Refinement: Local refinement is used for early generations (non-saturated generations (50))
% refinement incorporates: each facility location is updated by the point which
% has minimum weighted sum of distances to the other points
% within the cluster corresponding to that facility

function []=MCLP_GA(P,K)
    data=readmatrix("C:\MCLP_GA\SJC818.txt");         %Data Set Coordinate file path
    %file can be downloaded from: http://www.lac.inpe.br/~lorena/instances/mcover/Coord/coord818.txt
    len=size(data);                                 
    nrows=len(1);
    demand=readmatrix("C:\MCLP_GA\SJC818Demand.txt"); %Data Set Demand file path
    %http://www.lac.inpe.br/~lorena/instances/mcover/Demanda/demanda818.dat
    totalDemand=sum(demand,'all');
    population=initialize(P,K,nrows);
    r=1600;
    i=1;
    j=1;
    mutprob=0.01;
    [saturatedHundred,saturatedFifty]=notTerminated(population,P,K,r,i,totalDemand,demand,data,nrows);
    while saturatedHundred
            M=selection(population(:,:,i),P,K,r,totalDemand,demand,data,nrows);
            C=crossover(M,P,K);
            T=mutation(C,P,K,data,nrows,mutprob);
            if(saturatedFifty)
                j=0;
            end
            if i==1|| j==1
                mutprob=0.01;
                T=refinement(T,P,K,r,data,nrows,demand);
            else
                mutprob=0.8;
            end
            population(:,:,i+1)=createNextGenerationFrom(population(:,:,i),T,P,K,r,totalDemand,demand,data,nrows);         
            [saturatedHundred,saturatedFifty]=notTerminated(population,P,K,r,i,totalDemand,demand,data,nrows);
            i=i+1;  
    end
    fitnessOriginal=evaluate(population(:,:,i-1),P,K,r,totalDemand,demand,data,nrows);
    [x,y]=max(fitnessOriginal);
    fprintf("\nRadius: %d, Facilities opened: %d",r,K);
    fprintf("\nCoverage: %f Percentage",x*100);
    'Facility Cordinates',data(population(y,:,i-1),:)
end

function[flag,saturatedFifty]=notTerminated(population,P,K,r,noOfGenerations,totalDemand,demand,data,nrows)
    flag=true;
    saturatedFifty=false;
    if(noOfGenerations==1)
        flag=true;
    else
        x=size(population);
        count=0;
        n=x(3);
        if(n>100)
            val=evaluate(population(:,:,noOfGenerations),P,K,r,totalDemand,demand,data,nrows);
            for i=(n-1):-1:(n-100)
                temp=evaluate(population(:,:,i),P,K,r,totalDemand,demand,data,nrows);
                if (max(temp)==max(val))
                    count=count+1;
                    flag=false;
                else
                    flag=true;
                end
            end
        end
        if count>=50
            saturatedFifty=true;
        end
    end
    
end

function [population] = initialize(P,K,nrow)
    population=randi([1 nrow],P,K,1);
end


function [cover]=evaluate(population,P,K,r,totalDemand,demand,data,nrows)
    coverage=zeros(1,P);
    for i=1:P
        covered=zeros(1,nrows);
        val=0;
        for j=1:K
            for k=1:nrows
                if covered(k)==0
                    d=((data(k,1)-data(population(i,j),1))^2+(data(k,2)-data(population(i,j),2))^2)^0.5;
                    if d<=r
                        val=val+demand(k);
                        covered(k)=1;
                    end
                end
            end
        end
        coverage(1,i)=val/totalDemand;
    end
    cover=coverage(:);
end
function[fitness]=getFitness(chromosome,K,r,totalDemand,demand,data,nrows)
    val=0;
    covered=zeros(nrows,1);
    for i=1:K
        for j=1:nrows
          if covered(j)==0
            d=((data(j,1)-data(chromosome(i),1))^2+(data(j,2)-data(chromosome(i),2))^2)^0.5;
            if d<=r
                val=val+demand(j);
                covered(j)=1;
            end
          end
        end
    end
    fitness=val/totalDemand;
end
function [mating_pool]=selection(population,P,K,r,totalDemand,demand,data,nrows)
    mating_pool=zeros(P,K);
    for i=1:P
        p1=randi([1 P],1);
        p2=randi([1 P],1);
        f1=getFitness(population(p1,:),K,r,totalDemand,demand,data,nrows);
        f2=getFitness(population(p2,:),K,r,totalDemand,demand,data,nrows);
        if(f1>f2)
            mating_pool(i,:)=population(p1,:);
        else
            mating_pool(i,:)=population(p2,:);
        end
    end
end

function [C]=crossover(M,P,K)
    C=zeros(P,K);
    crossprob=0.9;
    x=1;
    off1=zeros(1,K);
    off2=zeros(1,K);
    for i=1:(P/2)
             r=rand;
             p1=randi([1 P],1);
             p2=randi([1 P],1);
            if(r<=crossprob)
                crosspoint=randi([1 K],1);
                for m=1:crosspoint
                    off1(1,m)=M(p1,m);
                    off2(1,m)=M(p2,m);
                end
                for m=crosspoint+1:K
                    off1(1,m)=M(p2,m);
                    off2(1,m)=M(p1,m);
                end
            else
                  for m=1:K
                    off1(1,m)=M(p1,m);
                    off2(1,m)=M(p2,m);
                  end
            end
            C(x,:)=off1(1,:);
            C(x+1,:)=off2(1,:);
            x=x+2;
            off1=[];
            off2=[];
    end   
end
function[neighbour]=getNeighbours(facility,N,data,nrows)
    distance=randi([0 0],nrows,1);
    for i=1:nrows
        if(facility ~=i)
            distance(i)=((data(facility,1)-data(i,1))^2+(data(facility,2)-data(i,2))^2)^0.5;
        end
    end
    k=round(nrows*N/100);
    [B,neighbour]=mink(distance,k);
end
function[T]=mutation(C,P,K,data,nrows,mutprob)
    N=5;
    T=C;
    for i=1:P
        for j=1:K
            r=rand;
            if(r<=mutprob)
                neighbour=getNeighbours(C(i,j),N,data,nrows);
                mutated=randsample(neighbour,1);
                T(i,j)=mutated;
            end   
        end
    end
end

function[population]=refinement(population,P,K,r,data,nrows,demand)
distance=NaN*zeros(nrows,K);
    for m=1:P
        for i=1:nrows
            for j=1:K
                d=((data(i,1)-data(population(m,j),1))^2+(data(i,2)-data(population(m,j),2))^2)^0.5;
                if d<=r
                    distance(i,j)=d;
                else
                    distance(i,j)=NaN;
                end
            end
        end
        minDis=zeros(1,nrows);
        facilityNo=zeros(1,nrows);
        for i=1:nrows
            [minDis(1,i),facilityNo(1,i)]=min(distance(i,:),[],'omitnan');
            if isnan(min(distance(i,:)))
                facilityNo(1,i)=NaN;
            end
        end
        clusters=[];
        for i=1:K
            for j=1:nrows
               if ~(isnan(facilityNo(1,j)))
                if(facilityNo(1,j))==i
                    clusters(1,end+1)=j;
                end
               end
            end
            x=size(clusters);
            weightedDistance=NaN*ones(1,x(2));
            for j=1:x(2)
                sum=0;
                for k=1:x(2)
                    if j~=k
                        d=((data(clusters(1,j),1)-data(clusters(1,k),1))^2+(data(clusters(1,j),2)-data(clusters(1,k),2))^2)^0.5;
                        if(d<=r)
                            wd=demand(clusters(1,k))*d;
                        else
                            wd=NaN;
                        end
                        sum=sum+wd;
                    end
                end
                weightedDistance(1,j)=sum;
            end
            [minWD,updatedIndexFacility]=min(weightedDistance(1,:),[],'omitnan');
            if ~isempty(updatedIndexFacility)
                population(m,i)=clusters(1,updatedIndexFacility);
            end
            clusters=[];
        end
    end
    
end
function[pool]=createNextGenerationFrom(population,T,P,K,r,totalDemand,demand,data,nrows)
    pool=zeros(P,K);
    mergedPop=cat(1,population,T);
    fitness=evaluate(mergedPop,2*P,K,r,totalDemand,demand,data,nrows);
    [val,index]=maxk(fitness,P);
    for i=1:P
        pool(i,:)=mergedPop(index(i),:);
    end
end