%Initialization
close all
clear all
clc

%%
%Importing data to train and test models
data_c = importdata('cancerDataforModel.csv');
data_cancer = data_c.data;
fprintf('Data read completed \n')

%%
%Partitioning data into training and test sets using Holdout
rng(1)
cancer_cval = cvpartition(data_cancer(:,1),'HoldOut',0.3);
idx = cancer_cval.test;

%split data into train and test sets
train_data = data_cancer(~idx,:);    
test_data = data_cancer(idx,:);
fprintf('Data partition completed \n')

%%
%SMOTE - Synthetic Minority Oversampling Technique
%Initial dataset consist of only 569 unique records of patients, Increasing
%the dataset size to train the model

[train_data_smote,C,Xn,Cn] = smote(train_data,[5],'Class',train_data(:,1));

%%
%write training and test data to file

%writematrix(train_data_smote,'CancerTrainData.csv');
writematrix(test_data, 'CancerTestData.csv');
fprintf('Data write completed \n')

%%
% split data and class

train_X = train_data_smote(:,[2:31]);
train_Y = train_data_smote(:,1);

test_X = test_data(:,[2:31]);
test_Y = test_data(:,1);
%%
%adding noise to train data
%Initial analysis of dataset was resulting in high accuracy, adding gaussian noise. 


xnoise = randn(size(train_X) ,'like',train_X);
train_X = train_X + (7.7 * xnoise) + 6.3 ;


%%

%Logistic Regression
%Creating the best model that fit the data
fprintf('-------------------------------------- \n')
fprintf('LOGISTIC REGRESSION to train model \n')
cancerPredictionLogisticRegression;
fprintf('-------------------------------------- \n')


%%

%Random Forest
%Creating the best model that fit the data
fprintf('-------------------------------------- \n')
fprintf('RANDOM FOREST to train model \n')
cancerPredictionRandomForest; 
fprintf('-------------------------------------- \n')

%%
%AUC for LR - RF test data

figure
plot(XteLR,YteLR,'LineWidth',2); hold on;
plot(XteRF,YteRF,'LineWidth',2, 'LineStyle','-.'); 
legend('LogisticRegression','RandomForest')
xlabel('False positive rate') 
ylabel('True positive rate')
title('ROC for LogisticRegression - RandomForest over test(unseen) data')
hold off;






