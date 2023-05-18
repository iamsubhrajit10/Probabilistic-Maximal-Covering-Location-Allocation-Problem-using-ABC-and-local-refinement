sum=0;
fitM=[];
totalTime=0;
global counters;
counters=zeros(1,4);
global cumProbabilites;
cumProbabilites=zeros(1,4);
for i=1:10
    tic;
    fit=PMCLAP_ABC(30,10,750,96,0.95,0);
    totalTime=totalTime+toc;
    sum=sum+fit;
    fitM(end+1)=fit;
end
avg=sum/10;
fprintf("\nAverage Fitness: %f ",avg);
fprintf("+\nMax Fitness: %f ",max(fitM));
y=0;
for i=1:10
    x=(fitM(1,i)-avg)^2;
    y=y+x;
end
standardDeviation=(y/10)^0.5;
fprintf("\nStandard Deviation: %f ",standardDeviation);
fprintf("\nAverage Time: %f ",totalTime/10);
counters


sum=0;
fitM=[];
totalTime=0;
counters=zeros(1,4);
cumProbabilites=zeros(1,4);
for i=1:10
    tic;
    fit=PMCLAP_ABC(30,10,750,96,0.95,1);
    totalTime=totalTime+toc;
    sum=sum+fit;
    fitM(end+1)=fit;
end
avg=sum/10;
fprintf("\nAverage Fitness: %f ",avg);
fprintf("+\nMax Fitness: %f ",max(fitM));
y=0;
for i=1:10
    x=(fitM(1,i)-avg)^2;
    y=y+x;
end
standardDeviation=(y/10)^0.5;
fprintf("\nStandard Deviation: %f ",standardDeviation);
fprintf("\nAverage Time: %f ",totalTime/10);
counters

sum=0;
fitM=[];
totalTime=0;
counters=zeros(1,4);
cumProbabilites=zeros(1,4);
for i=1:10
    tic;
    fit=PMCLAP_ABC(30,10,750,96,0.95,2);
    totalTime=totalTime+toc;
    sum=sum+fit;
    fitM(end+1)=fit;
end
avg=sum/10;
fprintf("\nAverage Fitness: %f ",avg);
fprintf("+\nMax Fitness: %f ",max(fitM));
y=0;
for i=1:10
    x=(fitM(1,i)-avg)^2;
    y=y+x;
end
standardDeviation=(y/10)^0.5;
fprintf("\nStandard Deviation: %f ",standardDeviation);
fprintf("\nAverage Time: %f ",totalTime/10);
counters
sum=0;
fitM=[];
totalTime=0;
counters=zeros(1,4);
cumProbabilites=zeros(1,4);
for i=1:10
    tic;
    fit=PMCLAP_ABC(30,20,750,96,0.85,0);
    totalTime=totalTime+toc;
    sum=sum+fit;
    fitM(end+1)=fit;
end
avg=sum/10;
fprintf("\nAverage Fitness: %f ",avg);
fprintf("+\nMax Fitness: %f ",max(fitM));
y=0;
for i=1:10
    x=(fitM(1,i)-avg)^2;
    y=y+x;
end
standardDeviation=(y/10)^0.5;
fprintf("\nStandard Deviation: %f ",standardDeviation);
fprintf("\nAverage Time: %f ",totalTime/10);
counters

sum=0;
fitM=[];
totalTime=0;
counters=zeros(1,4);
cumProbabilites=zeros(1,4);
for i=1:10
    tic;
    fit=PMCLAP_ABC(30,20,750,96,0.95,0);
    totalTime=totalTime+toc;
    sum=sum+fit;
    fitM(end+1)=fit;
end
avg=sum/10;
fprintf("\nAverage Fitness: %f ",avg);
fprintf("+\nMax Fitness: %f ",max(fitM));
y=0;
for i=1:10
    x=(fitM(1,i)-avg)^2;
    y=y+x;
end
standardDeviation=(y/10)^0.5;
fprintf("\nStandard Deviation: %f ",standardDeviation);
fprintf("\nAverage Time: %f ",totalTime/10);
counters

sum=0;
fitM=[];
totalTime=0;
counters=zeros(1,4);
cumProbabilites=zeros(1,4);
for i=1:10
    tic;
    fit=PMCLAP_ABC(30,20,750,96,0.85,1);
    totalTime=totalTime+toc;
    sum=sum+fit;
    fitM(end+1)=fit;
end
avg=sum/10;
fprintf("\nAverage Fitness: %f ",avg);
fprintf("+\nMax Fitness: %f ",max(fitM));
y=0;
for i=1:10
    x=(fitM(1,i)-avg)^2;
    y=y+x;
end
standardDeviation=(y/10)^0.5;
fprintf("\nStandard Deviation: %f ",standardDeviation);
fprintf("\nAverage Time: %f ",totalTime/10);
counters

sum=0;
fitM=[];
totalTime=0;
counters=zeros(1,4);
cumProbabilites=zeros(1,4);
for i=1:10
    tic;
    fit=PMCLAP_ABC(30,20,750,96,0.95,1);
    totalTime=totalTime+toc;
    sum=sum+fit;
    fitM(end+1)=fit;
end
avg=sum/10;
fprintf("\nAverage Fitness: %f ",avg);
fprintf("+\nMax Fitness: %f ",max(fitM));
y=0;
for i=1:10
    x=(fitM(1,i)-avg)^2;
    y=y+x;
end
standardDeviation=(y/10)^0.5;
fprintf("\nStandard Deviation: %f ",standardDeviation);
fprintf("\nAverage Time: %f ",totalTime/10);
counters


sum=0;
fitM=[];
totalTime=0;
counters=zeros(1,4);
cumProbabilites=zeros(1,4);
for i=1:10
    tic;
    fit=PMCLAP_ABC(30,20,750,96,0.85,2);
    totalTime=totalTime+toc;
    sum=sum+fit;
    fitM(end+1)=fit;
end
avg=sum/10;
fprintf("\nAverage Fitness: %f ",avg);
fprintf("+\nMax Fitness: %f ",max(fitM));
y=0;
for i=1:10
    x=(fitM(1,i)-avg)^2;
    y=y+x;
end
standardDeviation=(y/10)^0.5;
fprintf("\nStandard Deviation: %f ",standardDeviation);
fprintf("\nAverage Time: %f ",totalTime/10);
counters


sum=0;
fitM=[];
totalTime=0;
counters=zeros(1,4);
cumProbabilites=zeros(1,4);
for i=1:10
    tic;
    fit=PMCLAP_ABC(30,20,750,96,0.85,1);
    totalTime=totalTime+toc;
    sum=sum+fit;
    fitM(end+1)=fit;
end
avg=sum/10;
fprintf("\nAverage Fitness: %f ",avg);
fprintf("+\nMax Fitness: %f ",max(fitM));
y=0;
for i=1:10
    x=(fitM(1,i)-avg)^2;
    y=y+x;
end
standardDeviation=(y/10)^0.5;
fprintf("\nStandard Deviation: %f ",standardDeviation);
fprintf("\nAverage Time: %f ",totalTime/10);
counters



sum=0;
fitM=[];
totalTime=0;
counters=zeros(1,4);
cumProbabilites=zeros(1,4);
for i=1:10
    tic;
    fit=PMCLAP_ABC(30,50,750,96,0.85,0);
    totalTime=totalTime+toc;
    sum=sum+fit;
    fitM(end+1)=fit;
end
avg=sum/10;
fprintf("\nAverage Fitness: %f ",avg);
fprintf("+\nMax Fitness: %f ",max(fitM));
y=0;
for i=1:10
    x=(fitM(1,i)-avg)^2;
    y=y+x;
end
standardDeviation=(y/10)^0.5;
fprintf("\nStandard Deviation: %f ",standardDeviation);
fprintf("\nAverage Time: %f ",totalTime/10);
counters


sum=0;
fitM=[];
totalTime=0;
counters=zeros(1,4);
cumProbabilites=zeros(1,4);
for i=1:10
    tic;
    fit=PMCLAP_ABC(30,50,750,96,0.85,1);
    totalTime=totalTime+toc;
    sum=sum+fit;
    fitM(end+1)=fit;
end
avg=sum/10;
fprintf("\nAverage Fitness: %f ",avg);
fprintf("+\nMax Fitness: %f ",max(fitM));
y=0;
for i=1:10
    x=(fitM(1,i)-avg)^2;
    y=y+x;
end
standardDeviation=(y/10)^0.5;
fprintf("\nStandard Deviation: %f ",standardDeviation);
fprintf("\nAverage Time: %f ",totalTime/10);
counters


sum=0;
fitM=[];
totalTime=0;
counters=zeros(1,4);
cumProbabilites=zeros(1,4);
for i=1:10
    tic;
    fit=PMCLAP_ABC(30,50,750,96,0.85,2);
    totalTime=totalTime+toc;
    sum=sum+fit;
    fitM(end+1)=fit;
end
avg=sum/10;
fprintf("\nAverage Fitness: %f ",avg);
fprintf("+\nMax Fitness: %f ",max(fitM));
y=0;
for i=1:10
    x=(fitM(1,i)-avg)^2;
    y=y+x;
end
standardDeviation=(y/10)^0.5;
fprintf("\nStandard Deviation: %f ",standardDeviation);
fprintf("\nAverage Time: %f ",totalTime/10);
counters
function[fitmax]=PMCLAP_ABC(P,K,r,mu,alpha,b)
    x=mu*((1-alpha)^(1/(b+2)));
    data=readmatrix("C:\MCLP_GA\818.txt");         %Data Set Coordinate file path
    %file can be downloaded from: http://www.lac.inpe.br/~lorena/instances/mcover/Coord/coord818.txt
    len=size(data);                                 
    nrows=len(1);
    demand=data(:,3); %Data Set Demand file path
    distance = pdist2(data(:, 1:2), data(:, 1:2));
    population=initialize(P,K,nrows);
    epochs=1;
    fitness=computePopulationFitness(population(:,:,epochs),P,K,r,demand,distance,nrows,x,epochs);
    fitM=zeros(1000,P,1);
    fitM(1,:,:)=fitness;
    counter=zeros(1,P);
    while epochs<=1000 && notTerminated(fitM,epochs)                
        currentPop=population(:,:,epochs);
        epochs=epochs+1;
        eN=enhanceSolutionVector(currentPop, P, K, distance, demand, r, nrows, x,epochs);
        [eB,counter]=employeedBees(eN,P,K,distance,demand,r,nrows,x,epochs,counter);
        [oB,counter]=onlookerBees(eB,P,K,distance,demand,r,nrows,x,epochs,counter);
% %         if epochs<=50
% %             RP=refinement(oB,P,K,r,distance,nrows,demand,data,x,epochs);
% %         else
%         RP=oB;
        [SP,counter]=scoutBees(oB,P,K,distance,demand,r,nrows,x,epochs,counter);
        population(:,:,epochs)=SP;
        [population(:,:,epochs),fitness]=createNextGenerationFrom(population(:,:,epochs),currentPop,P,K,r,demand,distance,nrows,x,epochs);
        fitM(epochs,:,:)=fitness;
    end
    fitmax=max(fitM(:));
    fprintf("\nRadius: %d, Facilities opened: %d",r,K);
    fprintf("\nEpochs: %d",epochs);
    fprintf("\nFitness: %f ",fitmax);
    'Facility Coordinates',data(population(1,:,epochs),1:2)
end


function[flag]=notTerminated(fitM,n)
    flag=true;
    if n <= 100
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
function [population] = enhanceSolutionVector(population, P, K, distance, demand, r, nrows, x,epochs)
    N = 82;
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
    chosenPopulation = zeros(P, K);
    probabilities = zeros(P, 1);
    newPopulation=population;
    sumFitness = sum(computePopulationFitness(population,P,K,r,demand,distance,nrows,x,epochs));
    for i = 1:P
       probabilities(i) =getFitness(population(i,:), K, r, demand, distance, nrows, x,epochs) / sumFitness;
    end
    cumulativeProb=cumsum(probabilities);
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
    L = 20;
    for i=1:P
        if counter(1,i)> L
            population(i, :) = randperm(nrows,K);
            counter(1,i)=0;
        end
    end

end








function [pool,fitness] = createNextGenerationFrom(population, T, P, K, r, demand, distance, nrows, x,epochs)
    mergedPop = cat(1, population, T);
    fitness = computePopulationFitness(mergedPop, 2 * P, K, r, demand, distance, nrows, x,epochs);
    [fitness, indices] = maxk(fitness, P);
    pool = mergedPop(indices,:);

end


function [fit]=computePopulationFitness(population,P,K,r,demand,distance,nrows,x,epochs)
    fit=zeros(P,1);
    for i=1:P
        fit(i,1)=getFitness(population(i,:),K,r,demand,distance,nrows,x,epochs);
    end
end
function[fitness]=getFitness(solution,K,r,demand,distance,nrows,x,epochs)
   global counters;
   global cumProbabilites;
   
   if epochs<50
        fit1=getFitness1(solution,K,r,demand,distance,nrows,x,epochs);
        fit2=getFitness2(solution,K,r,demand,distance,nrows,x);
        fit3=getFitness3(solution,K,r,demand,distance,nrows,x);
        fit4=getFitness4(solution,K,r,demand,distance,nrows,x);
        [fitness,i]=max([fit1 fit2 fit3 fit4]);
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
            probabilites(4) = (counters(1,4) + eps) / (total + 4*eps); 

            % compute cumulative probabilities
            cumProbabilites = cumsum(probabilites);
        end

        % select a solution based on the computed probabilities
        randProb = rand();
        if randProb < cumProbabilites(1)
            index=1;
            fitness = getFitness1(solution, K, r, demand, distance, nrows, x,epochs);
        elseif randProb < cumProbabilites(2)
            index=2;
            fitness = getFitness2(solution, K, r, demand, distance, nrows, x);
        elseif randProb < cumProbabilites(3)
            index=3;
            fitness = getFitness3(solution, K, r, demand, distance, nrows, x);
        elseif randProb < cumProbabilites(4)
            index=4;
            fitness = getFitness4(solution, K, r, demand, distance, nrows, x);
        else
            index=1;
            fitness = getFitness2(solution, K, r, demand, distance, nrows, x);
        end
       counters(1,index)=counters(1,index)+1;
   end
end
function[fitness]=getFitness1(solution,K,r,demand,distance,nrows,x,epochs)
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
function[fitness]=getFitness2(solution,K,r,demand,distance,nrows,x)
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
function[fitness]=getFitness3(solution,K,r,demand,distance,nrows,x)
    val=0;
    yM=zeros(K,1);
    allocation=zeros(1,nrows);
    for i=1:nrows
        if allocation(1,i) ==0
            [yM,facilityNo,flag] = getNearestFacility(solution,i,distance,r,yM,x,demand,K);
             if flag
                val=val+demand(i);
                allocation(1,i)=solution(facilityNo);
            end
        end
    end
    fitness=val;
end

function[fitness]=getFitness4(solution,K,r,demand,distance,nrows,x)
    val=0;
    yM=zeros(K,1);
    allocation=zeros(1,nrows);
    weightedMatrix=zeros(1,nrows);
    myArray = 1:K;
    permIndices = randperm(numel(myArray));
    uniqueValues = unique(myArray(permIndices));
    for i=1:numel(uniqueValues)
        currentValue = uniqueValues(i);
        for j=1:nrows
            f=0.01*demand(j);
            if ~allocation(1,j) && (yM(currentValue,1)+f)<=x && distance(solution(currentValue),j)<=r
                weightedMatrix(1,j)=demand(j)/distance(solution(currentValue),j);
            end
        end
        [maxW,index]=max(weightedMatrix(1,:));
        if maxW ~=0
            allocation(1,index)=currentValue;
            val=val+demand(index);
            f=0.01*demand(index);
            yM(currentValue,1)=yM(currentValue,1)+f;
        end
    end
    fitness=val;
end

function[yM,facilityNo,flag]=getRandomFacility(solution,customer,distance,r,yM,x,demand,K)
  availableFacility=[];
  flag=false;
  facilityNo=-1;
  f=0.01*demand(customer);
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
end
function[yM,facilityNo,flag]=getLessCongestedFacility(solution,customer,distance,r,yM,x,demand,K)
  availableFacility=zeros(1,K);
  flag=false;
  facilityNo=-1;
  f=0.01*demand(customer);
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
end
function[yM,facilityNo,flag]=getNearestFacility(solution,customer,distance,r,yM,x,demand,K)
  availableFacility=zeros(1,K);
  flag=false;
  facilityNo=-1;
  f=0.01*demand(customer);
  min=distance(solution(1),customer);
  for i=1:K
      if distance(solution(i),customer)<=r && (yM(i,1)+f)<=x && distance(solution(i),customer)<min
          min=distance(solution(i),customer);
          facilityNo=i;
          availableFacility(1,i)=1;
      end
  end
  if facilityNo ~=-1
      yM(facilityNo,1)=yM(facilityNo,1)+f;
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

