Higgs Boson Detection - MLP vs SVM
---------------------------------------------------------------------------------------------------
**Overview**
------------------------

Data Set Information:

The data has been produced using Monte Carlo simulations. The first 21 features (columns 2-22) are kinematic properties measured by the particle detectors in the accelerator. The last seven features are functions of the first 21 features; these are high-level features derived by physicists to help discriminate between the two classes. There is an interest in using deep learning methods to obviate the need for physicists to manually develop such features. Benchmark results using Bayesian Decision Trees from a standard physics package and 5-layer neural networks are presented in the original paper. The last 500,000 examples are used as a test set.

Attribute Information:

The first column is the class label (1 for signal, 0 for background), followed by the 28 features (21 low-level features then 7 high-level features): lepton pT, lepton eta, lepton phi, missing energy magnitude, missing energy phi, jet 1 pt, jet 1 eta, jet 1 phi, jet 1 b-tag, jet 2 pt, jet 2 eta, jet 2 phi, jet 2 b-tag, jet 3 pt, jet 3 eta, jet 3 phi, jet 3 b-tag, jet 4 pt, jet 4 eta, jet 4 phi, jet 4 b-tag, m_jj, m_jjj, m_lv, m_jlv, m_bb, m_wbb, m_wwbb. For more detailed information about each feature see the original paper.


---------------------------------------------------------------------------------------------------
**Prerequisites** 
-------------------------

Python 
Anaconda Distribution - Jupyter Notebook

Google Colab

Download Python from - https://www.python.org/downloads/

**Python 3.8.8**

Install Anaconda from - https://docs.anaconda.com/anaconda/install/index.html

**conda 4.10.3**

Install pandas ref - https://pandas.pydata.org/docs/getting_started/install.html

**Pandas version - 1.2.4**

**$ conda install seaborn**

Operating System: macOS  Version: 11.5.1 Build: 20G80

Java Version: Java 1.8.0_202-b08 with Oracle Corporation Java HotSpot(TM) 64-Bit Server VM mixed mode

-----------------------------------------------

About this repo
---------------------------------

This repository contains the following files:
1) Higgs Boson Detection Using Multilayer Perceptron and Support Vector Machines.pdf - Repoort
2) AllCode-HiggsDetection-EDA-MLP-SVM-Testing.pdf - pdf file
3) HiggsDetection_NeuralComputing_Code

======================

HiggsDetection_NeuralComputing_Code
This repository contains the following files:
1) EDA_HiggsDetection.ipynb
2) MLP_HiggsDetection.ipynb
3) SVM_HiggsDetection.ipynb
4) HiggsData
5) SavedModels

File 1 - Exploratory data analysis done on Higgs Data
File 2 - Multiple models created for MLP to obtain best model
File 3 - Multiple models created for SVM to obtain best model

Folder 4 - HiggsData
Raw data, Processed data and train and test sets are present

Folder 5 - SavedModels
All saved models and for testing
Have all best models as pickle file and the file to test the models

**Running Code for TESTING**
-----------------------------------------------
For EDA
---------------------

•	Open folder -> SavedModels -> TestingBestModels -> TestingBestModels-MLP_SVM-HiggsDetection.ipynb

•	The code is already linked to the dataset, All that is needed is to ‘Run All’.

•	If being run in Google Colab - Append directory (instructions in file)

•	If running in Jupyter Notebook - uncomment the part that specifies 'To run in jupyter notebook' and run.

 
**Data**
-----------------------------------------------

Link for Data:
https://www.openml.org/search?type=data&status=active&id=4532

Also available in,
https://archive.ics.uci.edu/ml/machine-learning-databases/breast-cancer-wisconsin/wdbc.data

