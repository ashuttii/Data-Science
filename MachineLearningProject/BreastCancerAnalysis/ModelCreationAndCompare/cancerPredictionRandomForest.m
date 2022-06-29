%% Random Forest - Breast Cancer

%%
features = size(train_X);
numTree = [50, 100,200, 500, 1000];
leaf = [5 10 20 50];
NumPredictorstoSample = sqrt(features(2));

len = length(numTree)*length(leaf);
accuracy_mdl = zeros(len, 3);

%%
%Creating multiple models with all combinations of trees and leaf to find
%the best model to train

figure;
hold on;
rng(1);
for i = 1 : length(numTree)
    for j = 1:length(leaf)

        rowIndex = ((i-1) * length(leaf)) + j;
              
        Mdl_1 = TreeBagger(numTree(i),train_X, train_Y,'Method','classification', 'OOBPrediction','On',...
        'OOBPredictorImportance','off','NumPredictorsToSample',NumPredictorstoSample,'MinLeafSize',leaf(j)); %, 'MinLeafSize',leaf(j));
    
        oobErrorBaggedEnsemble = oobError(Mdl_1);
        plot(oobErrorBaggedEnsemble,'Color',rand(1,3),'LineWidth',2);
        
        % Predict and obtain the confusion matrix and accuracy
        [Yfit,Sfit] = oobPredict(Mdl_1);
        conf_mat = confusionmat(train_Y,str2double(Yfit));
        acc = trace(conf_mat)/sum(conf_mat, 'all');
        
        accuracy_mdl(rowIndex,:) = [numTree(i) leaf(j) acc];      
    end
end

% there is no direct way to plot legends with two for loop, ref -
% https://uk.mathworks.com/matlabcentral/answers/258113-how-to-use-legend-in-two-for-loops 
lh = legend({'50 5' '50 10' '50 20' '50 50' '100 5' '100 10' '100 20' '100 50' ...
    '200 5' '200 10' '200 20' '200 50' '500 5' '500 10' '500 20' '500 50' '1000 5' '1000 10' '1000 20' '1000 50'});
lh.Location='NorthEastOutside';
xlabel 'Number of grown trees (tree - leaf)';
ylabel 'Out-of-bag classification error';
hold off; 

fprintf('Model for Random forest created and looped with different combinations of tree and leaf size\n') 
%%
%Model with 1000 Tree and 5 leaf size - performed best

tic
Mdl_FI = TreeBagger(1000,train_X, train_Y,'Method','classification', 'OOBPrediction','On',...
        'OOBPredictorImportance','on','NumPredictorsToSample',NumPredictorstoSample,'MinLeafSize',5);
toc

[Yfit,Sfit] = oobPredict(Mdl_FI);
conf_mat = confusionmat(train_Y,str2double(Yfit));
acc = trace(conf_mat)/sum(conf_mat, 'all');    

fprintf('Model with 1000 Tree and 5 leaf size - performed best\n') 
fprintf('Model for RF created with best tree-leaf and with important predictors\n') 
%%
%To identify important predictors 

figure
bar(Mdl_FI.OOBPermutedPredictorDeltaError);
xlabel('Feature Number') ;
ylabel('Out-of-Bag Feature Importance');

fprintf('Predictor importance threshold set at 0.5 \n')

%%
% Threshold for feature importance set at 0.5 as it yeilded best result on
% analysis

idxvar = find(Mdl_FI.OOBPermutedPredictorDeltaError>0.5);

Mdl_FITrain = TreeBagger(1000,train_X(:,idxvar),train_Y,'Method','classification','OOBPredictorImportance','on','OOBPrediction','on');
figure
plot(oobError(Mdl_FITrain));
xlabel('Number of Grown Trees');
ylabel('Out-of-Bag Classification Error');

[Yfit,Sfit] = oobPredict(Mdl_FITrain);
conf_mat = confusionmat(train_Y,str2double(Yfit));
accFI = trace(conf_mat)/sum(conf_mat, 'all');
fprintf('Final model consist of 1000 trees and 5 leafs with important predictors \n')

%%

[XtrRF,YtrRF,TtrRF,AUCtrRF] = perfcurve(train_Y,Sfit(:,2),'1');

fprintf('Final LR model trained with binomial distribution and features selected from lasso \n')
fprintf('AUC ofTraining data : %4.3f \n',AUCtrRF)

%%
%Testing unseen data onto best model

tic
[testPredictRF,scoreRF] = predict(Mdl_FITrain,test_X(:,idxvar));
toc
%%
%Accuracy
v1RF = (testPredictRF == string(test_Y));
v2RF = 1- sum(v1RF)/size(testPredictRF,1);
accuracyTestRF = sum(v1RF)/size(testPredictRF,1)*100;

%Calculating model score of RF to identify AUC
[XteRF,YteRF,TteRF,AUCteRF] = perfcurve(test_Y,scoreRF(:,2),'1');

fprintf('Accuracy of the RF model in predicting test data : %4.3f \n',accuracyTestRF)
fprintf('AUC of Test data : %4.3f \n',AUCteRF)

%%
figure
plot(XtrRF,YtrRF,'LineWidth',2); hold on;
plot(XteRF,YteRF,'LineWidth',2, 'LineStyle','-.'); 
legend('Training','Test');
xlabel('False positive rate') ;
ylabel('True positive rate');
title('ROC for Classification by RandomForest');
hold off;


%%
total_MRF = sum(double(test_Y)==1);
predicted_MRF = sum(double(testPredictRF == "1"));

%%
% Precision - Recall - F1 score for Malignant cases

TP_MRF = sum((double(test_Y)==1) .* double(testPredictRF == "1"));
precision_MRF = TP_MRF/predicted_MRF;
recall_MRF = TP_MRF/total_MRF;

F1Score_MRF = 2 * precision_MRF * recall_MRF/(precision_MRF+recall_MRF);

fprintf('Malignant - RF \n')
fprintf('Precision : %4.3f \n',precision_MRF)
fprintf('Recall : %4.3f \n',recall_MRF)
fprintf('F1 score : %4.3f \n',F1Score_MRF)



%%
% Precision - Recall - F1 score for Begnin cases

total_BRF = sum(double(test_Y)==0);
predicted_BRF = sum(double(testPredictRF == "0"));

TP_BRF = sum((double(test_Y)==0) .* double(testPredictRF == "0"));
precision_BRF = TP_BRF/predicted_BRF;
recall_BRF = TP_BRF/total_BRF;

F1Score_BRF = 2 * precision_BRF * recall_BRF/(precision_BRF+recall_BRF);

fprintf('Begnin\n')
fprintf('Precision : %4.3f \n',precision_BRF)
fprintf('Recall : %4.3f \n',recall_BRF)
fprintf('F1 score : %4.3f \n',F1Score_BRF)


%%
% calculate accuracy
figure; 
pred =  str2double(testPredictRF);
cm = confusionchart(test_Y,pred);
cm.Title = 'Breast cancer - Random Forest';
cm.RowSummary = 'row-normalized';
cm.ColumnSummary = 'column-normalized';

%%