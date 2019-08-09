clear
clc
close all

load('..\orl_56x46.mat')

ClassNum = length(unique(gnd));
EachClassNum = 10;
Tr_num = 3;
% Tr_num = 1:5;
w_val = 0.2;
accuracy = zeros(length(w_val),length(Tr_num));
ii=1;
for train_num = Tr_num
    
    temp = zeros(1,EachClassNum);
    temp(1:train_num) = 1;
    train_ind = logical(repmat(temp,1,ClassNum));
    test_ind = ~train_ind;
    
    train_data = fea(:,train_ind);
    train_label = gnd(:,train_ind);
    
    test_data = fea(:,test_ind);
    test_label = gnd(:,test_ind);
    
    train_tol = length(train_label);
    test_tol = length(test_label);
    
    train_norm=normc(train_data);
    test_norm=normc(test_data);
    
    pre_label=zeros(1,test_tol);
    %     h = waitbar(0,'Please wait...');
    jj = 1;
    for w=w_val
        
        for i=1:test_tol
            y=test_norm(:,i);
            
            %             CRC_res=CRC(train_norm,ClassNum,train_label,y,P);
            LRC_res=LRC(train_norm,train_label,y,ClassNum);
            NRC_res=nonRC(train_norm,train_label,y);
            
            residual=w*LRC_res+(1-w)*NRC_res;
            [~,ind]=min(residual);
            pre_label(i)=ind;
            %         % computations take place here
            %         per = i / test_tol;
            %         waitbar(per, h ,sprintf('%2.0f%%',per*100))
        end
        %     close(h)
        
        accuracy(jj,ii)=sum(pre_label==test_label)/test_tol
        jj = jj+1;
        %     fprintf('训练样本数为：%d\n',train_num);
        %     fprintf(2,'识别率为：%3.2f%%\n\n',accuracy*100);
    end
    ii=ii+1;
end

figure
b = bar(residual,'k');
b.FaceColor = 'flat';
b.CData(ind,:) = [1 0 0];
