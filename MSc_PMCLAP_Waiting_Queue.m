filepath="C:\MCLP_GA\30.txt";

sumFitness=0;
totalTime=0;
global counters;
global cumProbabilites;
counters=zeros(1,3);
cumProbabilites=zeros(1,4);
instance='30_2_0_2_95';
bestFitness=0;
achieveCount=0;
bestTime=0;
noOfExecution=10;
fitnessTrack=zeros(1,noOfExecution);
timeTrack=zeros(1,noOfExecution);
bestEpochs=0;
totalEpochs=0;
for i=1:noOfExecution
    tic;
    [currentAllocation,currentFacilityIndices,fitness,currentEpochs]=PMCLAP_ABC(filepath,2,2,0.95);
    currentTime=toc;
    sumFitness=sumFitness+fitness;
    totalTime=totalTime+currentTime;
    totalEpochs=totalEpochs+currentEpochs;
    if bestFitness<=fitness
        bestFitness=fitness;
        bestAllocation=currentAllocation;
        bestFacilityIndices=currentFacilityIndices;
        if i == 1 || currentTime<bestTime
           bestTime=currentTime;
        end
        if i==1 || bestEpochs>currentEpochs
            bestEpochs=currentEpochs;
        end
    end
    if fitness >=4520
        achieveCount=achieveCount+1;
    end
    
    fitnessTrack(1,i)=fitness;
    timeTrack(1,i)=currentTime;
    fprintf('\n Executed %dth\n',i);
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
baseDirectory = 'C:\MCLP_GA\30R\';
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
function[bestAllocation,bestFacilityIndices,maxNectar,epochs]=PMCLAP_ABC(filepath,K,b,alpha)
    P=20;
    mu=72;
    r=1.5;
    x=mu*((1-alpha)^(1/(b+2)));
    x = setPrecision(x);
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
                distance(i,j)=setPrecision((setPrecision((data(i,1)-data(j,1))^2)+setPrecision((data(i,2)-data(j,2))^2))^0.5);
            end
        end
    end
    distance=setPrecision(distance);
   
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
    
    %to hold the abondonment counter values of the P solutions of the colony
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
        [updatedColony,~,~,~]=createNextGenerationFrom(sBColony,currentColony,P,K,r,demand,distance,m,x,epochs);
        % Regional Facility Enhancement Procedure after the bees are done
        [enhancedColony]=enhanceSolutionVector(updatedColony, P, K, distance, demand, r, m, x,epochs);
        %Best solutions achieved so far are kept intact
        [colony(:,:,epochs),nectar,currentAllocation,currentFacilityIndices]=createNextGenerationFrom(enhancedColony,updatedColony,P,K,r,demand,distance,m,x,epochs);
        %Updating the allocation, facility indices, fitness of the best
        %solution achieved so far
        [bestAllocation,bestFacilityIndices,maxNectar]=updateBestSolution(currentAllocation,currentFacilityIndices,bestAllocation,bestFacilityIndices,m,demand,maxNectar);
        nectarMatrix(epochs,:,:)=nectar;  %fitness matrix holding the fitness of all the solutions in the updated colony
    end 
end

% function to check the stopping criterion is met or not
% Inputs:- fitM: holding the fitness of all the solutions till
            % noOfIteration time
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

function [colony] = enhanceSolutionVector(colony, P, K, distance, demand, r, m, x,epochs)
    N =round(m/10);
    for i = 1:P
        for j = 1:K
            neighbours = getNeighbours(colony(i,j),colony(i,:), N, distance, m);
            neighbourhood = randsample(neighbours, 1);
            newColony = colony;
            newColony(i,j) = neighbourhood;
            if getFitness1(newColony(i,:), K, r, demand, distance, m, x,epochs) >= getFitness1(colony(i,:), K, r, demand, distance, m, x,epochs)
                colony(i,:) = newColony(i,:);
            end
        end
    end
end

% employeed bees phase works as per standard procedure explained in the
% paper
% returns the updated colony and abondonmentCounter values after employeed bees phase is done
function [eBColony,abandonmentCounter] = employeedBees(eBColony, P, K, distance, demand, r, m, x,epochs,abandonmentCounter)
    newColony = eBColony;
    for i=1:P
        for j=1:K
            k=i;
            while k == i
                k = randi(P);  % Generate a random number between 1 and P
            end
            phi=randi([-1 1]);
            %creates a new solution vector as per employe bee phase
            y=round(eBColony(i,j)+phi*(eBColony(i,j)-eBColony(k,j)));
            if y>0 && y<=m
                newColony(i,j) = y;
            end
        end
        %greedily chooses between newColony and old eBColony
        if getFitness1(newColony(i,:), K, r, demand, distance, m, x,epochs) > getFitness1(eBColony(i,:), K, r, demand, distance, m, x,epochs)
              eBColony(i,:) = newColony(i,:);
        else
              abandonmentCounter(1,i)=abandonmentCounter(1,i)+1;    % updates the abandonment counter
        end
    end
end

% onlooker bees phase as per standard onlooker bees procedure explained in
% the paper
% returns the updated colony and abaondoment counter after onlooker bees phase

function [oBColony,abandonmentCounter] = onlookerBees(oBColony, P, K, distance, demand, r, m, x,epochs,abandonmentCounter)
    newColony = oBColony;
    probabilities = zeros(P, 1);
    sumFitness = sum(computePopulationFitness(oBColony,P,K,r,demand,distance,m,x,epochs));
    for i = 1:P
       probabilities(i) =getFitness(oBColony(i,:), K, r, demand, distance, m, x,epochs) / sumFitness;
    end
    %computes the probability of each solution in the colony with upto 2
    %decimal precision
    probabilities=setPrecision(probabilities);
    %calculates cumulative probabilities and applies roulette wheel
    %selection
    cumulativeProb=cumsum(probabilities);
    cumulativeProb=setPrecision(cumulativeProb);
    for k = 1:P
        i = find(cumulativeProb >= rand(), 1);
        if ~(i==0)
            for q=1:K
                j=i;
                while j == i
                    j = randi(P);  % Generate a random number between 1 and P
                end
                phi=randi([-1 1]);
                v=round(oBColony(i,q)+phi*(oBColony(i,q)-oBColony(j,q)));
                if v>0 && v<=m
                    newColony(k, :) = v;
                end
            end
            if getFitness1(newColony(k,:), K, r, demand, distance, m, x,epochs) >= getFitness1(oBColony(k,:), K, r, demand, distance, m, x,epochs)
              oBColony(i,:) = newColony(k,:);
            else
              abandonmentCounter(1,i)=abandonmentCounter(1,i)+1;
            end
        end
    end
end

% Scout Bees Phase as per the standard procedure explained in the paper
% Returns the updated colony and abandonement counter
function [sBColony,abandonmentCounter] = scoutBees(sBColony,P, K, distance, demand, r, m, x,epochs,abandonmentCounter)
    L = floor(0.6*K*P); % Abandonment limit
    for i=1:P
        if abandonmentCounter(1,i)> L
            sBColony(i, :) = randperm(m,K);
            abandonmentCounter(1,i)=0;
        end
    end

end

% function to keep intanct the best solutions achieved so far
% it merges the two colonies provided as argument and returns the top P solutions
% from the merged colony of the both
function [bestColony,fitnessBestPop,allocation,facilityIndice] = createNextGenerationFrom(colony1, colony2, P, K, r, demand, distance, m, x,epochs)
    mergedColony = cat(1, colony1, colony2);
    [fitnessMergedPop,allocation,facilityIndice] = computePopulationFitness(mergedColony, 2 * P, K, r, demand, distance, m, x,epochs);
    [fitnessBestPop, indices] = maxk(fitnessMergedPop, P);
    bestColony = mergedColony(indices,:);
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
            cumProbabilites = cumsum(probabilites);
        end

        % select a solution based on the computed probabilities
        randProb = rand();
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

function[fitness,allocation]=getFitness1(solution,K,r,demand,distance,m,x,epochs)
    val=0;
    yM=zeros(m,1);
    allocation=zeros(1,m);
    for i=1:m
        if allocation(1,i) ==0
            [yM,facilityNo,flag] = getLessCongestedFacility(solution,i,distance,r,yM,x,demand,K);
             if flag
                val=val+demand(i);
                demand(i)=0;
                allocation(1,i)=solution(facilityNo);
            end
        end
    end
    fitness=val;
end
function[fitness,allocation]=getFitness2(solution,K,r,demand,distance,m,x)
    val=0;
    yM=zeros(m,1);
    allocation=zeros(1,m);
    for i=1:m
        if allocation(1,i) ==0
            [yM,facilityNo,flag] = getRandomFacility(solution,i,distance,r,yM,x,demand,K);
             if flag
                val=val+demand(i);
                demand(i)=0;
                allocation(1,i)=solution(facilityNo);
            end
        end
    end
    fitness=val;
end

function[fitness,allocation]=getFitness3(solution,K,r,demand,distance,m,x)
%     val=0;
%     yM=zeros(m,1);
%     allocation=zeros(1,m);  
%     for j=1:K
%         weightedMatrix=zeros(1,m);
%         for i=1:m
%             f=0.015*demand(i);
%             f=setPrecision(f);
%             if ~allocation(1,i) && setPrecision(yM(solution(j),1)+f)<=x && distance(solution(j),i)<=r
%                 weightedMatrix(1,i)=demand(i)/distance(solution(j),i);
%             end
%         end
%         [~,sortedIndices]=sort(weightedMatrix(:),'descend');
%         count=1;
%         while count <= m
%             if distance(solution(j),sortedIndices(count))>r ||~demand(sortedIndices(count))
%                 count=count+1;
%             else
%                 f=setPrecision(0.01*demand(sortedIndices(count)));
%                 if ~allocation(1,sortedIndices(count)) &&setPrecision(yM(solution(j),1)+f)<=x 
%                     val=val+demand(sortedIndices(count));
%                     demand(sortedIndices(count))=0;
%                     allocation(1,sortedIndices(count))=solution(j);
%                     yM(solution(j))=setPrecision(yM(solution(j),1)+f);
%                 end
%                 count=count+1;
%             end
%         end
%     end
%     fitness=val;
    val=0;
    yM=zeros(m,1);
    allocation=zeros(1,m);  
    for j=1:m
        weightedMatrix=zeros(1,K);
        f=0.015*demand(j);
        f=setPrecision(f);
        for i=1:K
            if ~allocation(1,j) && setPrecision(yM(solution(i),1)+f)<=x && distance(solution(i),j)<=r
                weightedMatrix(1,i)=demand(j)/distance(solution(i),j);
            end
        end
            weightedMatrix=setPrecision(weightedMatrix);
            [maxW,index]=max(weightedMatrix(1,:));
            if maxW ~=0
                allocation(1,j)=solution(index);
                val=val+demand(j);
                demand(i)=0;
                yM(solution(index),1)=setPrecision(yM(solution(index),1)+f);
            end
    end
    fitness=val;
end

function[yM,facilityNo,flag]=getRandomFacility(solution,customer,distance,r,yM,x,demand,K)
  availableFacility=[];
  flag=false;
  facilityNo=-1;
  f=0.015*demand(customer);
  f=setPrecision(f);
  j=1;
  for i=1:K
      if distance(solution(i),customer)<=r && setPrecision(yM(solution(i),1)+f)<=x
          availableFacility(end+1)=i;
          j=j+1;
          flag=true;
      end
  end

  if flag
    randomIndex = randi(numel(availableFacility));
    facilityNo=availableFacility(randomIndex);
    yM(solution(facilityNo),1)=setPrecision(yM(solution(facilityNo),1)+f);
  end
end
function[yM,facilityNo,flag]=getLessCongestedFacility(solution,customer,distance,r,yM,x,demand,K)
  availableFacility=zeros(1,K);
  flag=false;
  facilityNo=-1;
  f=0.015*demand(customer);
  f=setPrecision(f);
  min=-1;
  for i=1:K
      if distance(solution(i),customer)<=r && setPrecision(yM(solution(i),1)+f)<=x
          min=yM(solution(i),1);
          facilityNo=i;
          availableFacility(1,i)=1;
      end
  end
  if min~=-1
      for i=1:K
          if availableFacility(1,i)==1 && min>yM(solution(i),1)
              min=yM(solution(i),1);
              facilityNo=i;
          end
      end
  end
  if facilityNo ~=-1
      yM(solution(facilityNo),1)=setPrecision(yM(solution(facilityNo),1)+f);
      flag=true;
  end
end


function [neighbour] = getNeighbours(facility, solution, N, distance, m)
    d = Inf(m, 1);
    for i = 1:m
        if facility ~= i && ~ismember(i, solution)
            d(i) = distance(facility, i);
        end
    end
    
    if N >= m
        neighbour = setdiff(1:m, facility);
    else
        k = N;
        [~, indices] = mink(d, k);
        neighbour = indices;
    end
end
function result = setPrecision(value)
    result = round(value,3);
end

