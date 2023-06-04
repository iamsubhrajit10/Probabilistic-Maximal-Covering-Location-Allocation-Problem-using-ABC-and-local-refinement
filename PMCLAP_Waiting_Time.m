% provide the path of instance file e.g. 818 file
%file can be downloaded from: http://www.lac.inpe.br/~lorena/instances/mcover/Coord/coord818.txt
filepath="C:\MCLP_GA\818.txt";


sumFitness=0;
totalTime=0;
global counters;
global cumProbabilites;
counters=zeros(1,3);
cumProbabilites=zeros(1,4);

%provide the name of instance to be executed
instance='818_50_1_48_90';
bestFitness=0;
achieveCount=0;
bestTime=0;

%set number of times the instance should be executed
noOfExecution=2;
fitnessTrack=zeros(1,noOfExecution);
timeTrack=zeros(1,noOfExecution);
bestEpochs=0;
totalEpochs=0;
for i=1:noOfExecution
    tic;
    [currentAllocation,currentFacilityIndices,fitness,currentEpochs]=PMCLAP_ABC(filepath,50,48,0.90);
    currentTime=toc;
    sumFitness=sumFitness+fitness;
    totalTime=totalTime+currentTime;
    totalEpochs=totalEpochs+currentEpochs;
    if bestFitness<=fitness
        bestFitness=fitness;
        bestAllocation=currentAllocation;
        bestFacilityIndices=currentFacilityIndices;
        if i == 1 || currentTime<bestTime
           if i==1 || bestEpochs<currentEpochs
                bestEpcohs=currentEpochs;
           end
           bestTime=currentTime;
        end
    end
    if fitness >=134600
        achieveCount=achieveCount+1;
    end
    
    fitnessTrack(1,i)=fitness;
    timeTrack(1,i)=currentTime;
end
averageFitness=sumFitness/noOfExecution;

% computation of std. dev. of fitness
y=0;
for i=1:noOfExecution
    x=(fitnessTrack(1,i)-averageFitness)^2;
    y=y+x;
end
standardDevFitness=(y/noOfExecution)^0.5;
averageTime=totalTime/noOfExecution;

% computation of std. dev. of time
y=0;
for i=1:noOfExecution
    x=(timeTrack(1,i)-averageTime)^2;
    y=y+x;
end
standardDevTime=(y/noOfExecution)^0.5;
averageEpochs=totalEpochs/noOfExecution;

% set the path where output results to be written
baseDirectory = 'C:\MCLP_GA\818R\';
indiceFileName = sprintf('%s_%s.txt',  'indice', instance);
allocationFileName=sprintf('%s_%s.txt',  'allocation', instance);
metadataFileName=sprintf('%s_%s.txt',  'metadata', instance);
if ~isfolder(baseDirectory)
    mkdir(baseDirectory);
end
fullFilePath = fullfile(baseDirectory, indiceFileName);
fileID = fopen(fullFilePath, 'w');
% Write the facilities opened to the file
fprintf(fileID, '%d\t', bestFacilityIndices);
% Close the file
fclose(fileID);

fullFilePath = fullfile(baseDirectory, allocationFileName);
fileID = fopen(fullFilePath, 'w');
% Write the allocation matrix data to the file
fprintf(fileID, '%d\t', bestAllocation);
% Close the file
fclose(fileID);

fullFilePath = fullfile(baseDirectory, metadataFileName);
fileID = fopen(fullFilePath, 'w');
% Write the metadata to the file
fprintf(fileID, 'Best Fitness: %f\n', bestFitness);
fprintf(fileID, 'Average Fitness: %f\n', averageFitness);
fprintf(fileID, 'Std. Dev Fitness: %f\n', standardDevFitness);
fprintf(fileID, 'Best Instance Min Time: %f\n', bestTime);
fprintf(fileID, 'Average Time: %f\n', averageTime);
fprintf(fileID, 'Std. Dev Time: %f\n', standardDevTime);
fprintf(fileID, 'Best Fitness Count: %d\n', achieveCount);
fprintf(fileID, 'Epoch of Best Sol: %d\n',bestEpochs);
fprintf(fileID, 'Average Epochs: %f\n',averageEpochs);
% Close the file
fclose(fileID);
fprintf('\nSuccessfully Executed %s\n',instance);

%PMCLAP_ABC funtion implements the proposed strategy for solving PMCLAP
%Inputs:- filepath: path to the dataset of instance e.g. 818.txt and data is of size mx1
%Inputs:- K: number of facilities to be opened
%Inputs:- tau: queue waiting time in terms of minutes, a conversion 
           %is done to express it according to poisson distribution 
           %i.e. (24*60)/tau is used in the formulation 
%Inputs:- alpha: probability in terms of percentage e.g. 0.85
%Outputs:- bestAllocation: allocation matrix of 1xm size correspomding to
           %allocation of best solution achieved so far. Each coloumn
           %corresponds to a customer, and if the customer is allocated to
           %a facility then it contains the facility indice otherwise 0
%Outputs:- bestFacilityIndices: 1xK matrix, contains facility indices opened for best solution
%Outputs:- maxNectar: contains the nectar of best soultion achieved
%Outputs:- epochs: contains the number of iterations executed till convergence
function[bestAllocation,bestFacilityIndices,maxNectar,epochs]=PMCLAP_ABC(filepath,K,tau,alpha)
    P=20;   %Colony size
    mu=96;  %mu that appears in the formulation
    r=750;  %r radius in m
    x=mu+((log(1-alpha))*(1440/tau));   %RHS constraint calculation of constraint of waiting time
    x = formatToTwoDecimalPlaces(x);    %Precision to two decimal places
    data=readmatrix(filepath);          %Reading the data matrix of customers; 
                                        %It contains m rows of <x y demand> where x,y are the coordinates  

    len=size(data);                                 
    m=len(1);   % number of customers
    demand=data(:,3); % Demands of customers
    distance=zeros(m,m);
    
    % Calculation of distance matrix using euclidian distance and precision
    % is set to upto two decimal places
    for i=1:m
        for j=1:m
            if i~=j 
                distance(i,j)=formatToTwoDecimalPlaces((formatToTwoDecimalPlaces((data(i,1)-data(j,1))^2)+formatToTwoDecimalPlaces((data(i,2)-data(j,2))^2))^0.5);
            end
        end
    end
    distance=formatToTwoDecimalPlaces(distance);
   
    bestAllocation=zeros(1,m);      %to hold allocation of customers to facilities of best solution
    bestFacilityIndices=zeros(1,K); %to hold facility indices of best solution
    maxNectar=0;                       %to hold best fitness achieved so far
    
    %initialization of initial colony
    colony=initialize(P,K,m);   %PxK matrix each row holding a possible candidate solution of K facilities
    epochs=1;                       %counts the number of epochs executed
    
    %computes the fitness for each of the solutions in the colony, and also
    %returns the allocation and facilities opened of best solution in the
    %colony
    [nectar,currentAllocation,currentFacilityIndices]=computePopulationFitness(colony(:,:,epochs),P,K,r,demand,distance,m,x,epochs);
    
    %updates the best allocation, facility indices and max fitness achieved
    %with the best soultion achieved so far
    [bestAllocation,bestFacilityIndices,maxNectar]=updateBestSolution(currentAllocation,currentFacilityIndices,bestAllocation,bestFacilityIndices,m,demand,maxNectar);
    
    % fitness matrix to hold the fitness of all the solutions so far
    % generated of the populations across generations, as program is
    % limtied to max 1000 epochs, it's a 1000xPx1 matrix
    nectarMatrix=zeros(1000,P,1);
    nectarMatrix(1,:,:)=nectar;
    
    %to hold the abondant counter values of the P solutions of the colony
    abandonmentCounter=zeros(1,P);
    
    %execute till stopping criterion is met
    %limited to 1000 epochs max
    %notTerminated func. checks the stopping criterion is met or not
    while epochs<=1000 && notTerminated(nectarMatrix,epochs)
        currentColony=colony(:,:,epochs);  % holds the current colony of the current generation/iteration
        epochs=epochs+1;                    % as new colony to be generated, epochs is increased by 1
        
        %Employed Bees Phase
        [eBColony,abandonmentCounter]=employeedBees(currentColony,P,K,distance,demand,r,m,x,epochs,abandonmentCounter);
        
        %Onlooker Bees Phase
        [oBColony,abandonmentCounter]=onlookerBees(eBColony,P,K,distance,demand,r,m,x,epochs,abandonmentCounter);
        
        %Scout Bees Phase
        [sBColony,abandonmentCounter]=scoutBees(oBColony,P,K,distance,demand,r,m,x,epochs,abandonmentCounter);
        
        %Best solutions achieved so far are kept intact
        [updatedColony,~]=createNextGenerationFrom(sBColony,currentColony,P,K,r,demand,distance,m,x,epochs);
        
        % Regional Facility Enhancement Procedure after the bees are done
        enhancedColony=enhanceSolutionVector(updatedColony, P, K, distance, demand, r, m, x,epochs);
        
        %Best solutions achieved so far are kept intact
        [colony(:,:,epochs),nectar,currentAllocation,currentFacilityIndices]=createNextGenerationFrom(enhancedColony,updatedColony,P,K,r,demand,distance,m,x,epochs);
        
        %Updating the allocation, facility indices, fitness of the best
        %solution achieved so far
        [bestAllocation,bestFacilityIndices,maxNectar]=updateBestSolution(currentAllocation,currentFacilityIndices,bestAllocation,bestFacilityIndices,m,demand,maxNectar);
        
        nectarMatrix(epochs,:,:)=nectar;  %fitness matrix holding the fitness of all the solutions in the updated colony
    end 
end

% function to check the stopping criterion is met or not
% Inputs:- fitM: holding the fitness of all the solutions till n epochs
% Outputs:- n: number of epochs executed
% If the best solution so far achieved is not changed for the last 100
% iterations it returns false otherwise true
function[flag]=notTerminated(nectarMatrix,noOfIteration)
    flag=true;
    if noOfIteration <=100  % as checking for last 100 gen, so if n<=100 it just returns true
        return;
    end
    bestNectarOfLastIteration=max(nectarMatrix(noOfIteration,:));
    for i=noOfIteration-1:-1:noOfIteration-100
        thisIterationNectar=max(nectarMatrix(i,:));
        if thisIterationNectar<=bestNectarOfLastIteration
            flag=false;
        else
            flag=true;
            break;
        end
    end
end

%func to initialize the colony with random facility indices of size PxK
%returns the initialized colony
function [colony] = initialize(P,K,m)
    colony=zeros(P,K,1);
    
    %for each solution vector, it initializes with K possible candidate
    %facilities randomly from 1 to m i.e. within the customers
    for i=1:P
        colony(i,:,1)=randperm(m,K);
    end
end

%function to update the allocation matrix, facility indices opened and max
%nectar of the best solution till date
%Inputs:- currentAllocation, currentFacilityIndices, bestAllocation, bestFacilityIndices, m, demand, maxNectar
%Outputs:- updated bestAllocation, bestFacilityIndices, maxNectar
function[bestAllocation,bestFacilityIndices,maxNectar] = updateBestSolution(currentAllocation,currentFacilityIndices,bestAllocation,bestFacilityIndices,m,demand,maxNectar)
    nectarBest=0;
    nectarCurrent=0;
    for i=1:m
        if currentAllocation(1,i) ~=0
            nectarCurrent=nectarCurrent+demand(i);
        end
        if bestAllocation(1,i) ~=0
            nectarBest=nectarBest+demand(i);
        end
    end
    if nectarBest<nectarCurrent
        maxNectar=nectarCurrent;
        bestAllocation=currentAllocation;
        bestFacilityIndices=currentFacilityIndices;
    end
end

% function to apply the proposed regional facility enhancement procedure
% for each facility of each solution vector of the colony, it greedily
% chooses between one candidate neighbour facility from 10% neighbours and
% forms the enhanced colony
% It returns the enhanced colony of the regional facility enhancement procedure

function [enhancedColony] = enhanceSolutionVector(enhancedColony, P, K, distance, demand, r, m, x,epochs)
    N =round(m/10);
    for i = 1:P
        for j = 1:K
            neighbours = getNeighbours(enhancedColony(i,j), N, distance, m);
            
            neighbourhood = randsample(neighbours, 1);
            newPopulation = enhancedColony;
            newPopulation(i,j) = neighbourhood;
            if getFitness(newPopulation(i,:), K, r, demand, distance, m, x,epochs) >= getFitness(enhancedColony(i,:), K, r, demand, distance, m, x,epochs)
                enhancedColony(i,:) = newPopulation(i,:);
            end
        end
    end
end


function [population,counter] = employeedBees(population, P, K, distance, demand, r, nrows, x,epochs,counter)
    newPopulation = population;
    for i=1:P
        for j=1:K
            k=i;
            while k == i
                k = randi(P);  % Generate a random number between 1 and P
            end
            phi=randi([-1 1]);
            v=round(population(i,j)+phi*(population(i,j)-population(k,j)));
            if v>0 && v<=nrows
                newPopulation(i,j) = v;
            end
        end
        if getFitness(newPopulation(i,:), K, r, demand, distance, nrows, x,epochs) > getFitness(population(i,:), K, r, demand, distance, nrows, x,epochs)
              population(i,:) = newPopulation(i,:);
        else
              counter(1,i)=counter(1,i)+1;
        end
    end
end



function [newPopulation,counter] = onlookerBees(population, P, K, distance, demand, r, nrows, x,epochs,counter)
    chosenPopulation = population;
    probabilities = zeros(P, 1);
    newPopulation=population;
    sumFitness = sum(computePopulationFitness(population,P,K,r,demand,distance,nrows,x,epochs));
    for i = 1:P
       probabilities(i) =getFitness(population(i,:), K, r, demand, distance, nrows, x,epochs) / sumFitness;
    end
    probabilities=formatToTwoDecimalPlaces(probabilities);
    cumulativeProb=cumsum(probabilities);
    cumulativeProb=formatToTwoDecimalPlaces(cumulativeProb);
    for k = 1:P
        i = find(cumulativeProb >= rand(), 1);
        if ~(i==0)
            for q=1:K
                j=i;
                while j == i
                    j = randi(P);  % Generate a random number between 1 and P
                end
                phi=randi([-1 1]);
                v=round(population(i,q)+phi*(population(i,q)-population(j,q)));
                if v>0 && v<=nrows
                    chosenPopulation(k, :) = v;
                end
            end
            if getFitness(chosenPopulation(k,:), K, r, demand, distance, nrows, x,epochs) >= getFitness(population(k,:), K, r, demand, distance, nrows, x,epochs)
              newPopulation(i,:) = chosenPopulation(k,:);
            else
              counter(1,i)=counter(1,i)+1;
            end
        else
            chosenPopulation(k, :) = population(k, :);
        end
    end
end

function [population,counter] = scoutBees(population,P, K, distance, demand, r, nrows, x,epochs,counter)
    L = floor(0.1*K*P);
    for i=1:P
        if counter(1,i)> L
            population(i, :) = randperm(nrows,K);
            counter(1,i)=0;
        end
    end

end


function [pool,fitnessBestPop,allocation,facilityIndice] = createNextGenerationFrom(population, T, P, K, r, demand, distance, nrows, x,epochs)
    mergedPop = cat(1, population, T);
    [fitnessMergedPop,allocation,facilityIndice] = computePopulationFitness(mergedPop, 2 * P, K, r, demand, distance, nrows, x,epochs);
    [fitnessBestPop, indices] = maxk(fitnessMergedPop, P);
    pool = mergedPop(indices,:);
end


function [fit,allocation,facilityIndice]=computePopulationFitness(population,P,K,r,demand,distance,nrows,x,epochs)
    fit=zeros(P,1);
    allocationPopulation=zeros(P,1,nrows);
    for i=1:P
        [fit(i,1),allocationMatrix]=getFitness(population(i,:),K,r,demand,distance,nrows,x,epochs);
        allocationPopulation(i,1,:)=allocationMatrix;
    end
    [~,best]=max(fit);
    allocation=allocationPopulation(best,:,:);
    allocation = reshape(allocation, [1, nrows]);
    facilityIndice=population(best,:);
end
function[fitness,allocationMatrix]=getFitness(solution,K,r,demand,distance,nrows,x,epochs)
   global counters;
   global cumProbabilites;
   tempAllocation=zeros(1,3,nrows);
   if epochs<50
        [fit1,tempAllocation(1,1,:)]=getFitness1(solution,K,r,demand,distance,nrows,x,epochs);
        [fit2,tempAllocation(1,2,:)]=getFitness2(solution,K,r,demand,distance,nrows,x);
        [fit3,tempAllocation(1,3,:)]=getFitness3(solution,K,r,demand,distance,nrows,x);
        [fitness,i]=max([fit1 fit2 fit3]);
        allocationMatrix=tempAllocation(1,i,:);
        counters(1,i)=counters(1,i)+1;
   else
               % initialize eps to a small positive constant
        eps = 1e-10;

        if epochs ==50
            % compute probabilities using roulette wheel selection
            total = sum(counters(1,:));
            probabilites(1) = (counters(1,1) + eps) / (total + 4*eps);
            probabilites(2) = (counters(1,2) + eps) / (total + 4*eps); 
            probabilites(3) = (counters(1,3) + eps) / (total + 4*eps); 

            % compute cumulative probabilities
            probabilites=formatToTwoDecimalPlaces(probabilites);
            cumProbabilites = cumsum(probabilites);
            cumProbabilites=formatToTwoDecimalPlaces(cumProbabilites);
        end
        
        % select a solution based on the computed probabilities
        randProb = rand();
        randProb = formatToTwoDecimalPlaces(randProb);
        if randProb < cumProbabilites(1)
            index=1;
            [fitness,allocationMatrix] = getFitness1(solution, K, r, demand, distance, nrows, x,epochs);
            
        elseif randProb < cumProbabilites(2)
            index=2;
            [fitness,allocationMatrix] = getFitness2(solution, K, r, demand, distance, nrows, x);
        elseif randProb < cumProbabilites(3)
            index=3;
            [fitness,allocationMatrix] = getFitness3(solution, K, r, demand, distance, nrows, x);
        else
            index=1;
            [fitness,allocationMatrix] = getFitness1(solution, K, r, demand, distance, nrows, x);
        end
       counters(1,index)=counters(1,index)+1;
   end
end

function[fitness,allocation]=getFitness1(solution,K,r,demand,distance,nrows,x,epochs)
    val=0;
    yM=zeros(K,1);
    allocation=zeros(1,nrows);
    for i=1:nrows
        if allocation(1,i) ==0
            [yM,facilityNo,flag] = getLessCongestedFacility(solution,i,distance,r,yM,x,demand,K);
             if flag
                val=val+demand(i);
                allocation(1,i)=solution(facilityNo);
            end
        end
    end
    fitness=val;
end
function[fitness,allocation]=getFitness2(solution,K,r,demand,distance,nrows,x)
    val=0;
    yM=zeros(K,1);
    allocation=zeros(1,nrows);
    for i=1:nrows
        if allocation(1,i) ==0
            [yM,facilityNo,flag] = getRandomFacility(solution,i,distance,r,yM,x,demand,K);
             if flag
                val=val+demand(i);
                allocation(1,i)=solution(facilityNo);
            end
        end
    end
    fitness=val;
end

function[fitness,allocation]=getFitness3(solution,K,r,demand,distance,nrows,x)
    val=0;
    yM=zeros(K,1);
    allocation=zeros(1,nrows);  
    for j=1:nrows
        weightedMatrix=zeros(1,K);
        f=0.01*demand(j);
        f=formatToTwoDecimalPlaces(f);
        for i=1:K
            if ~allocation(1,j) && formatToTwoDecimalPlaces(yM(i,1)+f)<=x && distance(solution(i),j)<=r
                weightedMatrix(1,i)=demand(j)/distance(solution(i),j);
            end
        end
            weightedMatrix=formatToTwoDecimalPlaces(weightedMatrix);
            [maxW,index]=max(weightedMatrix(1,:));
            if maxW ~=0
                allocation(1,j)=solution(index);
                val=val+demand(j);
                yM(index,1)=formatToTwoDecimalPlaces(yM(index,1)+f);
            end
    end
    fitness=val;
end

function[yM,facilityNo,flag]=getRandomFacility(solution,customer,distance,r,yM,x,demand,K)
  availableFacility=[];
  flag=false;
  facilityNo=-1;
  f=0.01*demand(customer);
  f=formatToTwoDecimalPlaces(f);
  j=1;
  for i=1:K
      if distance(solution(i),customer)<=r && formatToTwoDecimalPlaces(yM(i,1)+f)<=x
          availableFacility(end+1)=i;
          j=j+1;
          flag=true;
      end
  end

  if flag
    randomIndex = randi(numel(availableFacility));
    facilityNo=availableFacility(randomIndex);
    yM(facilityNo,1)=formatToTwoDecimalPlaces(yM(facilityNo,1)+f);
  end
end
function[yM,facilityNo,flag]=getLessCongestedFacility(solution,customer,distance,r,yM,x,demand,K)
  availableFacility=zeros(1,K);
  flag=false;
  facilityNo=-1;
  f=0.01*demand(customer);
  f=formatToTwoDecimalPlaces(f);
  min=-1;
  for i=1:K
      if distance(solution(i),customer)<=r && formatToTwoDecimalPlaces(yM(i,1)+f)<=x
          min=yM(i,1);
          facilityNo=i;
          availableFacility(1,i)=1;
      end
  end
  if min~=-1
      for i=1:K
          if availableFacility(1,i)==1 && min>yM(i,1)
              min=yM(i,1);
              facilityNo=i;
          end
      end
  end
  if facilityNo ~=-1
      yM(facilityNo,1)=formatToTwoDecimalPlaces(yM(facilityNo,1)+f);
      flag=true;
  end
end


function [neighbour] = getNeighbours(facility, N, distance, nrows)
    d = Inf(nrows, 1);
    for i = 1:nrows
        if facility ~= i
            d(i) = distance(facility, i);
        end
    end
    
    if N >= nrows
        neighbour = setdiff(1:nrows, facility);
    else
        k = round(nrows * N / 100);
        [~, indices] = mink(d, k);
        neighbour = indices;
    end
end
function result = formatToTwoDecimalPlaces(value)
    result = floor(value * 100) / 100;
end

