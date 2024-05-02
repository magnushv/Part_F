%% Task 1: Generating random input data
meanDelayRange = [0, 0.7];
numServersRange = [0, 1]; 
repairUtilizationRange = [0, 1]; 

numRows = 1000;
randomMeanDelay = rand(numRows, 1) * (meanDelayRange(2) - meanDelayRange(1)) + meanDelayRange(1);
randomNumServers = rand(numRows, 1) * (numServersRange(2) - numServersRange(1)) + numServersRange(1);
randomRepairUtilization = rand(numRows, 1) * (repairUtilizationRange(2) - repairUtilizationRange(1)) + repairUtilizationRange(1);

disp("Snippet of randomly generated input data:");
disp([randomMeanDelay(1:5), randomNumServers(1:5), randomRepairUtilization(1:5)]);

%% Task 2: Generate targets using fuzzy inference system
spares = evalfis(fis, [randomMeanDelay, randomNumServers, randomRepairUtilization]);

disp("Generated targets (Number of spares):");
disp(spares(1:5));

%% Task 3: Split dataset into training and test sets
dataset = [randomMeanDelay, randomNumServers, randomRepairUtilization, spares];

trainRatio = 0.7;

numRows = size(dataset, 1);
trainIndices = randperm(numRows, round(trainRatio * numRows));
testIndices = setdiff(1:numRows, trainIndices);

trainData = dataset(trainIndices, :);
testData = dataset(testIndices, :);

disp("Number of rows in training set:");
disp(size(trainData, 1));
disp("Number of rows in test set:");
disp(size(testData, 1));

%% Task 4: Create and train neuro-fuzzy inference system
trainInput = trainData(:, 1:3); 
trainTarget = trainData(:, 4);   

numMFs = [3 3 3]; 

opt = genfisOptions('GridPartition');
opt.NumMembershipFunctions = numMFs;
opt.InputMembershipFunctionType = 'gaussmf';

anfisSystem = genfis(trainInput, trainTarget, opt);

epochNum = 50; 
anfisSystem = anfis(trainData, anfisSystem, epochNum);

% Handle warnings for input values outside the specified ranges
outOfRange1 = sum(trainInput(:, 1) < meanDelayRange(1) | trainInput(:, 1) > meanDelayRange(2));
outOfRange2 = sum(trainInput(:, 2) < numServersRange(1) | trainInput(:, 2) > numServersRange(2));
outOfRange3 = sum(trainInput(:, 3) < repairUtilizationRange(1) | trainInput(:, 3) > repairUtilizationRange(2));

if outOfRange1 > 0 || outOfRange2 > 0 || outOfRange3 > 0
    warning(['Some input values are outside the specified ranges: Input 1: ' num2str(outOfRange1) ', Input 2: ' num2str(outOfRange2) ', Input 3: ' num2str(outOfRange3)]);
end

%% Task 5: Evaluate neuro-fuzzy inference system
predictedTargets = evalfis(anfisSystem, testData(:, 1:3));
actualTargets = testData(:, 4);
errors = actualTargets - predictedTargets;

MSE = mean(errors.^2);
RMSE = sqrt(MSE);
meanError = mean(errors);
stdError = std(errors);

disp('Errors:');
disp(errors);
disp(['Mean Squared Error (MSE): ', num2str(MSE)]);
disp(['Root Mean Squared Error (RMSE): ', num2str(RMSE)]);
disp(['Mean of Errors: ', num2str(meanError)]);
disp(['Standard Deviation of Errors: ', num2str(stdError)]);

%% Task 6: Plot between test target and test output
figure;
plot(testData(:, 4), predictedTargets, 'bo');
hold on;
plot([min(testData(:, 4)), max(testData(:, 4))], [min(testData(:, 4)), max(testData(:, 4))], 'r--');
text(max(testData(:, 4)), min(testData(:, 4)), ['RMSE: ' num2str(RMSE)], 'HorizontalAlignment', 'right');
hold off;
xlabel('Test Target');
ylabel('Test Output');
title('Plot between Test Target and Test Output');
legend('Predicted Output', 'Ideal Output', 'Location', 'northwest');
grid on;

%% Task 7: Histogram of errors
figure;
histogram(errors, 'Normalization', 'probability');
xlabel('Error');
ylabel('Probability');
title('Histogram of Errors');

%% Task 8: Providing input beyond the universe of discourse and comparing outputs
extremeInputs = [1.2, 1.2, 1.2; -0.1, -0.1, -0.1];  %extreme value, 2 sets

extremeOutputsFIS = zeros(size(extremeInputs, 1), 1);
extremeOutputsANFIS = zeros(size(extremeInputs, 1), 1);

for i = 1:size(extremeInputs, 1)
    extremeOutputsFIS(i) = evalfis(fis, extremeInputs(i, :));
    extremeOutputsANFIS(i) = evalfis(anfisSystem, extremeInputs(i, :));
end

disp('Extreme outputs (FIS vs ANFIS):');
disp(table(extremeInputs, extremeOutputsFIS, extremeOutputsANFIS, ...
    'VariableNames', {'ExtremeInputs', 'FIS_Output', 'ANFIS_Output'}));
