%% Breast cancer - Logistic regression

%% 
model_basic =  fitglm(train_X,train_Y,'linear','Distribution','binomial','Link','log'); % Logistic regression
fprintf('Model trained on basic model\n')        

%%
%Cross validation - on training data
%Cross val splits the data into 10 folds and for each run one of the fold
%will be a test set, the rest will be trained to predict the test results.
%Model trained on Binomial distribution

P1 = 0.5; 
num_folds = 10;
accuracyTrainingModel = zeros(1,num_folds);

indices = crossvalind('Kfold',train_Y,num_folds);
    for i = 1:num_folds
        test_c = (indices == i); train_c = ~test_c;
        model_c =  fitglm(train_X(train_c,:),train_Y(train_c),'linear','Distribution','binomial', 'Link', 'logit'); % Logistic regression
        predict_c = logical(predict(model_c,train_X(test_c,:))>=P1);
        v2_c = (predict_c == train_Y(test_c));
        v3_c = 1- sum(v2_c)/size(predict_c,1);
        accuracyTrainingModel(i) = sum(v2_c)/size(predict_c,1)*100;
    end
fprintf('Model trained with binary distribution \n')
avgAccuracyTrainModel = mean(accuracyTrainingModel);
fprintf('Binary Distribution model accuracy : %4.3f, error : %4.3f \n',avgAccuracyTrainModel,v3_c)

%%
%Identifing important features using Lasso Generalized linear model
%B - importance of each feature in each iteration, 
%lam - the best feature selection

[B,FitInfo] = lassoglm(train_X,train_Y,'binomial','CV',10);

hold on;
lassoPlot(B,FitInfo,'plottype','CV'); 
legend('show') % Show legend
hold off;
lam = FitInfo.Index1SE;
featuresB = B(:,lam);

%looping to fecth best features from lassoglm, ie feature importance not 0
newFeatures = zeros();
for i = 1 : length(featuresB)
    if featuresB(i) ~= 0
        newFeatures(end+1) = i;
    end
end
fprintf('Feture selected using Lasso Generalized linear model \n')

%%
%Training model - and cross validation if the new fetures (from lassoglm)
%is improving the accuracy.

accuracyTrainLasso = zeros(1,num_folds);
indices = crossvalind('Kfold',train_Y,num_folds);
    for i = 1:num_folds
        test_c = (indices == i); train_c = ~test_c;
        model_las =  fitglm(train_X(train_c,newFeatures(:,2:end)),train_Y(train_c),'linear','Distribution','binomial', 'Link', 'logit'); % Logistic regression
        predict_las = logical(predict(model_las,train_X(test_c,newFeatures(:,2:end)))>=P1);
        v2_las = (predict_las == train_Y(test_c));
        v3_las = 1- sum(v2_las)/size(predict_las,1);
        accuracyTrainLasso(i) = sum(v2_las)/size(predict_las,1)*100;
    end


fprintf('Model trained with Features selected from Lasso Generalized linear model\n')
avgAccuracyLassoFI = mean(accuracyTrainLasso);
fprintf('Lasso Generalized linear model accuracy : %4.3f, error : %4.3f \n',avgAccuracyLassoFI,v3_las)

    
%%
%Finalizing a model for LR - binomial distribution with feature importance
%from lassoglm
tic
modelLogisticRegression =  fitglm(train_X(:,newFeatures(:,2:end)),train_Y,'linear','Distribution','binomial','Link','logit');
toc
%%
figure;
plotResiduals(modelLogisticRegression,'probability');
%%
%Calculating model score of LR to identify AUC
model_score = modelLogisticRegression.Fitted.Response;
[XtrLR,YtrLR,TtrLR,AUCtrLR] = perfcurve(train_Y,model_score,'1');

fprintf('Final LR model trained with binomial distribution and features selected from lasso \n')
fprintf('AUC of Training data : %4.3f \n',AUCtrLR)

%%
%Model creation completed and cross validated.
%Model is ready to test new/unseen data
%selecting lasso features from testset and then predicting results 

test_X_newFeaturesLR  = test_X(:,newFeatures(:,2:end));
tic
[testPredictLR,scoreLR] = predict(modelLogisticRegression,test_X_newFeaturesLR );
toc
%%
% %% Testset accuracy is calculated, probablity threshold set - 0.5, ie
% any prediction over 0.5 is Malignent

P1 = 0.5 
v1LR = logical(testPredictLR >= P1);
v2LR = (v1LR == test_Y);
v3LR = 1- sum(v2LR)/size(v1LR,1);
accuracyTestLR = sum(v2LR)/size(v1LR,1)*100;


%Calculating model score of LR to identify AUC
[XteLR,YteLR,TteLR,AUCteLR] = perfcurve(test_Y,scoreLR(:,1),'1');

fprintf('Accuracy of the LR model in predicting test data : %4.3f \n',accuracyTestLR)
fprintf('AUC of Test data : %4.3f \n',AUCteLR)

%%
%Plote Area Under Curve for training and test data. 

figure; hold on;
plot(XtrLR,YtrLR,'LineWidth',2); hold on;
plot(XteLR,YteLR,'LineWidth',2, 'LineStyle','-.'); 
legend('Training','Test')
xlabel('False positive rate') 
ylabel('True positive rate')
title('ROC for Classification by Logistic Regression')
hold off;
%%
% Precision - Recall - F1 score for Malignent cases 

total_MLR = sum(double(test_Y)==1);
predicted_MLR = sum(double(v1LR == 1));

TP_MLR = sum((double(test_Y)==1) .* double(v1LR == 1));
precision_MLR = TP_MLR/predicted_MLR;
recall_MLR = TP_MLR/total_MLR;

F1Score_MLR = 2 * precision_MLR * recall_MLR/(precision_MLR+recall_MLR);

fprintf('Malignant\n')
fprintf('Precision : %4.3f \n',precision_MLR)
fprintf('Recall : %4.3f \n',recall_MLR)
fprintf('F1 score : %4.3f \n',F1Score_MLR)

%%
%Precision - Recall - F1 score for Begnin cases

total_BLR = sum(double(test_Y)==0);
predicted_BLR = sum(double(v1LR == 0));

TP_BLR = sum((double(test_Y)==0) .* double(v1LR == 0));
precision_BLR = TP_BLR/predicted_BLR;
recall_BLR = TP_BLR/total_BLR;

F1Score_BLR = 2 * precision_BLR * recall_BLR/(precision_BLR+recall_BLR);

fprintf('Begnin\n')
fprintf('Precision : %4.3f \n',precision_BLR)
fprintf('Recall : %4.3f \n',recall_BLR)
fprintf('F1 score : %4.3f \n',F1Score_BLR)


%%
%Confusion Matrix for Logistic Regression
figure; 
cc = confusionchart(test_Y, double(v1LR))
cc.Title = 'Breast cancer - Logistic Regression';
cc.RowSummary = 'row-normalized';
cc.ColumnSummary = 'column-normalized';
