close all;
clear all
%% Load Data File
load('new_data10'); 
read_data_train = data_train;
read_data_test = data_test;
% Result
%read_data_train 32036 x2049 ; 
%read_data_test  8009x2049;
% 2049 column is the class number of 1 to 10
%% Train Data by Using Data_Train and by Assuming That It Is Gaussian Distribution
% step 1: extract classes from data_train
% step 2: calculate the mean values and covariance matrix for each class using Gaussian_ML_estimate
% step 3: base on train data calculate the priori probabilities of each class. They are also parameters for beyansian classifier

% labeled:Class 1

        class1_data= read_data_train(find(read_data_train(:,2049) == 1),1:2048);
        [m1_hat, S1_hat]=Gaussian_ML_estimate(class1_data');
        p1= length(find(read_data_train(:,2049)==1))/32036;

% labeled:Class 2
       class2_data= read_data_train(find(read_data_train(:,2049) == 2),1:2048);
         [m2_hat, S2_hat]=Gaussian_ML_estimate(class2_data');
         p2= length(find(read_data_train(:,2049)==2))/32036;
% labeled:Class 3
       class3_data= read_data_train(find(read_data_train(:,2049) == 3),1:2048);
        [m3_hat, S3_hat]=Gaussian_ML_estimate(class3_data');
        p3= length(find(read_data_train(:,2049)==3))/32036;
% labeled:Class 4
      class4_data= read_data_train(find(read_data_train(:,2049) == 4),1:2048);
      [m4_hat, S4_hat]=Gaussian_ML_estimate(class4_data');
      p4= length(find(read_data_train(:,2049)==4))/32036;
% labeled:Class 5
      class5_data= read_data_train(find(read_data_train(:,2049) == 5),1:2048);
      [m5_hat, S5_hat]=Gaussian_ML_estimate(class5_data');
      p5= length(find(read_data_train(:,2049)==5))/32036;
% labeled:Class 6
      class6_data= read_data_train(find(read_data_train(:,2049) == 6),1:2048);
      [m6_hat, S6_hat]=Gaussian_ML_estimate(class6_data');
      p6= length(find(read_data_train(:,2049)==6))/32036;
% labeled:Class 7
     class7_data= read_data_train(find(read_data_train(:,2049) == 7),1:2048);
       [m7_hat, S7_hat]=Gaussian_ML_estimate(class7_data');
       p7= length(find(read_data_train(:,2049)==7))/32036;
% labeled:Class 8
     class8_data= read_data_train(find(read_data_train(:,2049) == 8),1:2048);
       [m8_hat, S8_hat]=Gaussian_ML_estimate(class8_data');
       p8= length(find(read_data_train(:,2049)==8))/32036;
% labeled:Class 9
    class9_data= read_data_train(find(read_data_train(:,2049) == 9),1:2048);
    [m9_hat, S9_hat]=Gaussian_ML_estimate(class9_data');
    p9= length(find(read_data_train(:,2049)==9))/32036;
% labeled:Class 10
   class10_data= read_data_train(find(read_data_train(:,2049) == 10),1:2048);
   [m10_hat, S10_hat]=Gaussian_ML_estimate(class10_data');
   p10= length(find(read_data_train(:,2049)==10))/32036;
   
 %m_hat and S_hat and p
 m_hat=[m1_hat m2_hat m3_hat m4_hat m5_hat m6_hat m7_hat m8_hat m9_hat m10_hat];
 S_hat=(1/10)*(S1_hat+S2_hat+S3_hat+S4_hat+S5_hat+S6_hat+S7_hat+S8_hat+S9_hat+S10_hat);
 p=[p1 p2 p3 p4 p5 p6 p7 p8 p9 p10];
 
   
%% TEST DATA by Using data_test. Use 3 distance measure Euclidean Mahalanobis and Baysian and compare three results

test=read_data_test(:,1:2048);
% Euclidean distance classifier
z_euclidean=euclidean_classifier(m_hat,test');

%  Mahalanobis distance classifier
%z_mahalanobis=mahalanobis_classifier(m_hat,S_hat,test');

%  bayes classifier and provide as input the matrices

%z_bayesian=bayes_classifier(m_hat,S_hat,p,test');
%% Error calculation

err_euclidean = (1-length(find(read_data_test(:,2049)==z_euclidean'))/8009)
%err_mahalanobis = (1-length(find(read_data_test(:,2049)==z_mahalanobis'))/8009)
%err_bayesian = (1-length(find(read_data_test(:,2049)==z_bayesian'))/8009)

% Result: error of euclidean classifier: 0.7544


