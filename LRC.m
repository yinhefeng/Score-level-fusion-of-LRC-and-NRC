function residual=LRC(train_data,train_label,y,ClassNum)
delta=1e-3;
residual=zeros(1,ClassNum);
for j=1:ClassNum
    X=train_data(:,j==train_label);
    trans_mat=pinv(X'*X)*X';
    %         trans_mat=pinv(X'*X+delta*eye(size(X,2)))*X';
    %     trans_mat=(X'*X+delta*eye(size(X,2)))\X';
    beta=trans_mat*y;
    test_project=X*beta;
    %         beta(:,j)=(X'*X+delta*eye(size(X,2)))\X'*y;
    residual(j)=norm(y-test_project);
end

res_max=max(residual);
res_min=min(residual);
residual=(residual-res_min)/(res_max-res_min);
