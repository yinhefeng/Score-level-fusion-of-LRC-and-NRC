function residual=LRC(train_data,train_label,y,ClassNum)

% Input:
% train_data: training data matrix
% train_label: label vector of train_data
% y: the test sample
% ClassNum: number of classes

% Output:
% residual: normalized residual

residual=zeros(1,ClassNum);
for j=1:ClassNum
    X=train_data(:,j==train_label); %training data of the j-th class
    trans_mat=pinv(X'*X)*X';
    beta=trans_mat*y; % estimated coefficient
    test_project=X*beta; % reconstructed test sample
    residual(j)=norm(y-test_project); % residual
end

% normalize the residual to [0,1]
res_max=max(residual);
res_min=min(residual);
residual=(residual-res_min)/(res_max-res_min);
