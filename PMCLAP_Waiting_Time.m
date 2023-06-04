filepath="C:\MCLP_GA\818.txt";

sumFitness=0;
totalTime=0;
global counters;
global cumProbabilites;
counters=zeros(1,3);
cumProbabilites=zeros(1,4);
instance='818_20_1_42_85';
bestFitness=0;
achieveCount=0;
bestTime=0;
noOfExecution=30;
fitnessTrack=zeros(1,noOfExecution);
timeTrack=zeros(1,noOfExecution);
bestEpochs=0;
totalEpochs=0;
for i=1:noOfExecution
    tic;
    [currentAllocation,currentFacilityIndices,fitness,currentEpochs]=PMCLAP_ABC(filepath,20,42,0.85);
    currentTime=toc;
    sumFitness=sumFitness+fitness;
    totalTime=totalTime+currentTime;
    totalEpochs=totalEpochs+currentEpochs;
    if bestFitness<=fitness
        bestFitness=fitness;
        bestAllocation=currentAllocation;
        bestFacilityIndices=currentFacilityIndices;
        if i == 1 || currentTime<bestTime
           if bestEpochs<currentEpochs
                bestEpcohs=currentEpochs;
           end
           bestTime=currentTime;
        end
    end
    if fitness >=61900
        achieveCount=achieveCount+1;
    end
    
    fitnessTrack(1,i)=fitness;
    timeTrack(1,i)=currentTime;
end
averageFitness=sumFitness/noOfExecution;
y=0;
for i=1:noOfExecution
    x=(fitnessTrack(1,i)-averageFitness)^2;
    y=y+x;
end
standardDevFitness=(y/noOfExecution)^0.5;
averageTime=totalTime/noOfExecution;
y=0;
for i=1:noOfExecution
    x=(timeTrack(1,i)-averageTime)^2;
    y=y+x;
end
standardDevTime=(y/noOfExecution)^0.5;
averageEpochs=totalEpochs/noOfExecution;
baseDirectory = 'C:\MCLP_GA\818R\';
indiceFileName = sprintf('%s_%s.txt',  'indice', instance);
allocationFileName=sprintf('%s_%s.txt',  'allocation', instance);
metadataFileName=sprintf('%s_%s.txt',  'metadata', instance);
if ~isfolder(baseDirectory)
    mkdir(baseDirectory);
end
fullFilePath = fullfile(baseDirectory, indiceFileName);
fileID = fopen(fullFilePath, 'w');
% Write the matrix data to the file
fprintf(fileID, '%d\t', bestFacilityIndices);
% Close the file
fclose(fileID);

fullFilePath = fullfile(baseDirectory, allocationFileName);
fileID = fopen(fullFilePath, 'w');
% Write the matrix data to the file
fprintf(fileID, '%d\t', bestAllocation);
% Close the file
fclose(fileID);

fullFilePath = fullfile(baseDirectory, metadataFileName);
fileID = fopen(fullFilePath, 'w');
% Write the matrix data to the file
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

function[bestAllocation,bestFacilityIndices,fitmax,epochs]=PMCLAP_ABC(filepath,K,tau,alpha)
    P=20;
    mu=96;
    r=750;
    x=mu+((log(1-alpha))*(1440/tau));
    x = formatToTwoDecimalPlaces(x);
    data=readmatrix(filepath);         %Data Set Coordinate file path
    %file can be downloaded from: http://www.lac.inpe.br/~lorena/instances/mcover/Coord/coord818.txt
    len=size(data);                                 
    m=len(1);   % number of customers
    demand=data(:,3); %Data Set Demand file path
    distance=zeros(m,m);
    for i=1:m
        for j=1:m
            if i~=j 
                distance(i,j)=((formatToTwoDecimalPlaces(((data(i,1)-data(j,1)))^2)+formatToTwoDecimalPlaces(((data(i,2)-data(j,2)))^2))^0.5);
            end
        end
    end
    distance = formatToTwoDecimalPlaces(distance);
    bestAllocation=zeros(1,m);
    bestFacilityIndices=zeros(1,K);
    fitmax=0;
    population=initialize(P,K,m); 
    epochs=1;
    [fitness,currentAllocation,currentFacilityIndices]=computePopulationFitness(population(:,:,epochs),P,K,r,demand,distance,m,x,epochs);
    [bestAllocation,bestFacilityIndices,fitmax]=updateBestSolution(currentAllocation,currentFacilityIndices,bestAllocation,bestFacilityIndices,m,demand,fitmax);
    fitM=zeros(1000,P,1);
    fitM(1,:,:)=fitness;
    counter=zeros(1,P);
    
    while epochs<=1000 && notTerminated(fitM,epochs)                
        currentPop=population(:,:,epochs);
        epochs=epochs+1;
        [eB,counter]=employeedBees(currentPop,P,K,distance,demand,r,m,x,epochs,counter);
        [oB,counter]=onlookerBees(eB,P,K,distance,demand,r,m,x,epochs,counter);
        [SP,counter]=scoutBees(oB,P,K,distance,demand,r,m,x,epochs,counter);
        [modifiedPop,~]=createNextGenerationFrom(SP,currentPop,P,K,r,demand,distance,m,x,epochs);
        eN=enhanceSolutionVector(modifiedPop, P, K, distance, demand, r, m, x,epochs);
        [population(:,:,epochs),fitness,currentAllocation,currentFacilityIndices]=createNextGenerationFrom(eN,modifiedPop,P,K,r,demand,distance,m,x,epochs);
        [bestAllocation,bestFacilityIndices,fitmax]=updateBestSolution(currentAllocation,currentFacilityIndices,bestAllocation,bestFacilityIndices,m,demand,fitmax);
        fitM(epochs,:,:)=fitness;  
    end 
end


function[flag]=notTerminated(fitM,n)
    flag=true;
    if n <100
        return;
    end
    mx=max(fitM(n,:));
    for i=n-1:-1:n-100
        temp=max(fitM(i,:));
        if temp<=mx
            flag=false;
        else
            flag=true;
            break;
        end
    end
end

function [population] = initialize(P,K,nrows)
    population=zeros(P,K,1);
    for i=1:P
        population(i,:,1)=randperm(nrows,K);
    end
end
function[bestAllocation,bestFacilityIndices,fitmax] = updateBestSolution(currentAllocation,currentFacilityIndices,bestAllocation,bestFacilityIndices,m,demand,fitmax)
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
        fitmax=nectarCurrent;
        bestAllocation=currentAllocation;
        bestFacilityIndices=currentFacilityIndices;
    end
end

function [population] = enhanceSolutionVector(population, P, K, distance, demand, r, nrows, x,epochs)
    N =round(nrows/10);
    for i = 1:P
        for j = 1:K
            neighbours = getNeighbours(population(i,j), N, distance, nrows);
            
            neighbourhood = randsample(neighbours, 1);
            newPopulation = population;
            newPopulation(i,j) = neighbourhood;
            if getFitness(newPopulation(i,:), K, r, demand, distance, nrows, x,epochs) >= getFitness(population(i,:), K, r, demand, distance, nrows, x,epochs)
                population(i,:) = newPopulation(i,:);
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
            if ~allocation(1,j) && (yM(i,1)+f)<=x && distance(solution(i),j)<=r
                weightedMatrix(1,i)=demand(j)/distance(solution(i),j);
            end
        end
            weightedMatrix=round(weightedMatrix,2);
            [maxW,index]=max(weightedMatrix(1,:));
            if maxW ~=0
                allocation(1,j)=solution(index);
                val=val+demand(j);
                yM(index,1)=yM(index,1)+f;
            end
    end
    yM=formatToTwoDecimalPlaces(yM);
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
      if distance(solution(i),customer)<=r && (yM(i,1)+f)<=x
          availableFacility(end+1)=i;
          j=j+1;
          flag=true;
      end
  end

  if flag
    randomIndex = randi(numel(availableFacility));
    facilityNo=availableFacility(randomIndex);
    yM(facilityNo,1)=yM(facilityNo,1)+f;
  end
  yM=formatToTwoDecimalPlaces(yM);
end
function[yM,facilityNo,flag]=getLessCongestedFacility(solution,customer,distance,r,yM,x,demand,K)
  availableFacility=zeros(1,K);
  flag=false;
  facilityNo=-1;
  f=0.01*demand(customer);
  f=formatToTwoDecimalPlaces(f);
  min=-1;
  for i=1:K
      if distance(solution(i),customer)<=r && (yM(i,1)+f)<=x
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
      yM(facilityNo,1)=yM(facilityNo,1)+f;
      flag=true;
  end
  yM=formatToTwoDecimalPlaces(yM);
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

