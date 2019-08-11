function residual=nonRC(X,trls,y)
% Input:
% X: training data matrix
% trls: label vector of X
% y: the test sample

% Output:
% residual: normalized residual

ClassNum = length(unique(trls));
[~,c] = NRC(X, y);
train_tol = length(trls);
% construct a sparse matrix to speed up
W = sparse([],[],[],train_tol,ClassNum,length(c));

% obtain the coefficient vectors for each class
for j=1:ClassNum
    ind = (j==trls);
    W(ind,j) = c(ind);
end

% compute the residual
temp = X*W-repmat(y,1,ClassNum);
residual = sqrt(sum(temp.^2));

% normalize the residual to [0,1]
res_max=max(residual);
res_min=min(residual);
residual=(residual-res_min)/(res_max-res_min);

end