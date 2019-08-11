function [Z,C] = NRC(X, Y)
%implementation of nonnegative representation based classification which is
%proposed by the following paper.
% Xu J, An W, Zhang L, et al. Sparse, collaborative, or nonnegative
% representation: Which helps pattern classification?[J].
% Pattern Recognition, 2019, 88: 679-688.

[~,n] = size(X);
m = size(Y,2);
tol = 1e-5;
maxIter = 5;
mu= 1e-1;
Z = zeros(n,m);
C = zeros(n,m);
delta = zeros(n,m);

XTX = X'*X;
XTY = X'*Y;
iter = 0;
temp_X = pinv(XTX+mu/2*eye(n));
while iter<maxIter
    iter = iter + 1;
    
    Zk = Z;
    Ck = C;
    
    % update c
    %     C = (XTX+mu/2*eye(n))\(XTY+mu/2*Z+delta/2);
    C = temp_X*(XTY+mu/2*Z+delta/2);
    
    % update z
    z_temp = C-delta/mu;
    Z = max(0,z_temp);
    
    leq1 = Z-C;
    leq2 = Z-Zk;
    leq3 = C-Ck;
    stopC1 = max(norm(leq1,'fro'),norm(leq2,'fro'));
    stopC = max(stopC1,norm(leq3,'fro'));
    %     disp(stopC)
    
    if stopC<tol || iter>=maxIter
        break;
    else
        delta = delta + mu*leq1;
    end
end