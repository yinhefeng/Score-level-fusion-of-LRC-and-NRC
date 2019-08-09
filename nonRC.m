function residual=nonRC(X,trls,y)
ClassNum = length(unique(trls));
[~,c] = NRC(X, y);
train_tol = length(trls);
%构造sparse矩阵，大小为train_tol*ClassNum，最多有length(x)个非零值
W = sparse([],[],[],train_tol,ClassNum,length(c));

%得到每类对应的系数
for j=1:ClassNum
    ind = (j==trls);
    W(ind,j) = c(ind);
end

%计算测试样本和每类重构样本之间的残差
temp = X*W-repmat(y,1,ClassNum);
residual = sqrt(sum(temp.^2));

res_max=max(residual);
res_min=min(residual);
residual=(residual-res_min)/(res_max-res_min);
% %把测试样本分在最小残差对应的类别中
% [~,index]=min(residual);

end