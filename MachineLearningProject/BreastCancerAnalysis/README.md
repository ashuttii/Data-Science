Breast Cancer Wisconsin Diagnosis 
---------------------------------------------------------------------------------------------------
**Overview**
------------------------

Records of unique patients whose features are computed from a digitized image of the cell nuclei of breast mass. There are 30 features and a target (Diagnosis) classifying if the cancer is B-Benign or M-malignant.

The aim is to compare Logistic Regression and Random Forest Machine learning techniques, to identify accuracy and overall performance.

---------------------------------------------------------------------------------------------------
**Prerequisites** 
-------------------------

Python 
Anaconda Distribution - Jupyter Notebook
MATLAB

Download Python from - https://www.python.org/downloads/

**Python 3.8.8**

Install Anaconda from - https://docs.anaconda.com/anaconda/install/index.html

**conda 4.10.3**

Install pandas ref - https://pandas.pydata.org/docs/getting_started/install.html

**Pandas version - 1.2.4**

**$ conda install seaborn**

MATLAB - https://uk.mathworks.com/login?uri=%2Fdownloads%2Fweb_downloads
MATLAB version – 

**MATLAB Version: 9.10.0.1739362** (R2021a) Update 5

MATLAB License Number: *****

Operating System: macOS  Version: 11.5.1 Build: 20G80

Java Version: Java 1.8.0_202-b08 with Oracle Corporation Java HotSpot(TM) 64-Bit Server VM mixed mode


MATLAB                                                Version 9.10        (R2021a)

Bioinformatics Toolbox                                Version 4.15.1      (R2021a)

Statistics and Machine Learning Toolbox               Version 12.1        (R2021a)
>>

-----------------------------------------------

About this repo
---------------------------------

This repository contains the following files:
1) RawData
2) ExploratoryDataAnalysis
3) ModelCreationAndCompare

======================

RawData - Consist of unprocessed data from Kaggle(link below) 

ExploratoryDataAnalysis - Jupyter notebook on which the initial analysis of data is performed.

ModelCreationAndCompare - MATLAB codes where multiple models are created to derive at the final best model in both Logistic Regression and Random Forest and compare the performance and accuracy.

-- cancerDataforModel.csv --> Analysed and transformed data from Jupyter notebook

-- cancerData.m --> Main MATLAB file where data is split to test and train, SMOTE technique is used to multiply imbalanced data, Gaussian noise is added to avoid overfitting
This will inturn call ancerPredictionLogisticRegression.m and cancerPredictionRandomForest.m

-- cancerPredictionLogisticRegression.m --> Logistic regression model using fitglm() created to identify best model. 

--cancerPredictionRandomForest.m --> Random Forest model using Treebagger with multiple tree and leaf size created to identify best fit model. 



**Running Code**
-----------------------------------------------
For EDA
---------------------

•	Upload **BreastCancerEDA.ipynb** file from **BreastCancerAnalysis/ExploratoryDataAnalysis/** onto Jupyter Notebook. 

•	The code is already linked to the dataset, All that is needed is to ‘Run All’.


For Model creation and Comparison
---------------------

•	Load **BreastCancerAnalysis/ModelCreationAndCompare project** in MATLAB. 

•	Open **cancerData.m** file, all data is already linked to the code.

•	Run (all)

 

**Data**
-----------------------------------------------

Link for Data:
https://www.kaggle.com/uciml/breast-cancer-wisconsin-data

Also available in,
https://archive.ics.uci.edu/ml/machine-learning-databases/breast-cancer-wisconsin/wdbc.data

