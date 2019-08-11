clear
clc
close all

load('FERET.mat')

% total number of samples in each class
EachClassNum = 7;

% number of subjects
ClassNum = length(unique(all_label));

% the first one image of each class is used for training
train_num = 1;
temp = zeros(1,EachClassNum);
temp(1:train_num) = 1;
train_ind = logical(repmat(temp,1,ClassNum));
test_ind = ~train_ind;

% obtain the training data and its label vectors
train_data = all_data(:,train_ind);
train_label = all_label(:,train_ind);

% obtain the test data and its label vectors
test_data = all_data(:,test_ind);
test_label = all_label(:,test_ind);

% total number of training and test data
train_tol = length(train_label);
test_tol = length(test_label);

% dimensionality for PCA
dim = [60 100 140 180 200];
w_val = 0.8; %parameter for fusion
accuracy = zeros(length(w_val),length(dim));
ii = 1;
for Eigen_NUM=dim
    % dimensionality reduction by PCA
    [Pro_Matrix,Mean_Image] = my_pca(train_data,Eigen_NUM);
    train_pro = Pro_Matrix'*train_data;
    test_pro = Pro_Matrix'*test_data;
    
    % unit L2 norm
    train_norm = normc(train_pro);
    test_norm = normc(test_pro);
    
    pre_label = zeros(1,test_tol);
    jj = 1;
    for w=w_val
        
        for i=1:test_tol
            % the i-th test sample
            y = test_norm(:,i);
            
            % residual computed by linear regression classification
            LRC_res = LRC(train_norm,train_label,y,ClassNum);
            
            % residual computed by nonnegative representation based classification
            NRC_res = nonRC(train_norm,train_label,y);
            
            % score level fusion
            residual = w*LRC_res+(1-w)*NRC_res;
            [~,ind] = min(residual);
            pre_label(i) = ind;
        end
        
        accuracy(jj,ii) = sum(pre_label==test_label)/test_tol
        jj = jj+1;
    end
    ii = ii+1;
end
