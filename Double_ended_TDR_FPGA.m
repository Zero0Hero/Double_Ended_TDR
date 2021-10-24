%%  双端同步模型通信
%   基于Time Delay Reservoir Computing
%   By Zhuo Liu 2021.10.23 Nanjing
%% %导入数据集
clear;clc;
load ('L_ode_28.mat'); 
L_x=X(:,1)'/max(X(:,1))/2.5+0.6; 
L_z=X(:,3)'/max(X(:,3))/1.2+0.05;                                  

P_L=270000;
%% %训练类参数 
%状态更新参数

p = 4;          
r = 2;%
n = 2.8;
h = 0.03;

NUM_of_Virtualnodes = 24;   %虚拟神经节点数(N)
N_RC=4;
NUM_of_Training = 6000;     %训练点数
Discard_steps = 100;        %前忽略点数
NUM_of_Testing = 270000;       %测试点数
Length_of_data = Discard_steps+NUM_of_Testing+NUM_of_Training; %总数据长度

%初始化
X_N = zeros(NUM_of_Testing,NUM_of_Virtualnodes*N_RC);
M=-1+(1-(-1)).*rand(1,NUM_of_Virtualnodes); %生成随机掩码
tintial = M;    %初始值

for ti=1:1:N_RC
    if ti==1
        Data_train = L_z(1:Discard_steps+NUM_of_Training) ;    %导入训练数据
        Data_input = L_x(1:Discard_steps+NUM_of_Training);
        J_Input = Data_input'*M;    %输入*掩码
    else
        Data_input = Y_testing1;
        J_Input = Data_input'*M;    %输入*掩码
    end
    X_hat=RC_t(tintial,J_Input,Discard_steps+NUM_of_Training,NUM_of_Virtualnodes,n,r,p,h) ;
    %% 单位矩阵
    II=zeros(NUM_of_Virtualnodes,NUM_of_Virtualnodes);
    for i=1:NUM_of_Virtualnodes
        for j=1:NUM_of_Virtualnodes
            if(i==j)
                II(j,i)=0.000001;
            end
        end
    end
    W(ti,:)=Data_train(1,Discard_steps+1:Discard_steps+NUM_of_Training)*X_hat(Discard_steps+1:Discard_steps+NUM_of_Training,:)*(X_hat(Discard_steps+1:Discard_steps+NUM_of_Training,:)'*X_hat(Discard_steps+1:Discard_steps+NUM_of_Training,:)+II)^(-1);
    %% 使用原始数据集带入W求得Y
    Y_testing1=W(ti,:)*X_hat';
    X_N(1:Discard_steps+NUM_of_Training,NUM_of_Virtualnodes*(ti-1)+1:NUM_of_Virtualnodes*ti)=X_hat;
end

II=0.000001 .* eye(NUM_of_Virtualnodes*N_RC);

W_N=Data_train(1,Discard_steps+1:Discard_steps+NUM_of_Training)*X_N(Discard_steps+1:Discard_steps+NUM_of_Training,:)*(X_N(Discard_steps+1:Discard_steps+NUM_of_Training,:)'*X_N(Discard_steps+1:Discard_steps+NUM_of_Training,:)+II)^(-1);
 
for ti=1:1:N_RC
    if ti==1
        Data_input = L_x(Length_of_data-P_L+1:Length_of_data);
        J_Input = Data_input'*M;    %输入*掩码
    else
        Data_input = Y_testing1;
        J_Input = Data_input'*M;    %输入*掩码
    end
    X_hat=RC_t(X_N(Discard_steps+NUM_of_Training,NUM_of_Virtualnodes*(ti-1)+1:NUM_of_Virtualnodes*ti),J_Input,NUM_of_Testing,NUM_of_Virtualnodes,n,r,p,h) ;
    Y_testing1=W(ti,:)*X_hat';
    X_N(1:NUM_of_Testing,NUM_of_Virtualnodes*(ti-1)+1:NUM_of_Virtualnodes*ti)=X_hat;
end

Y_testing1=W_N*X_N';
figure(1);
P_L=10000;
subplot(3,3,1);
plot(L_x(Length_of_data-P_L+1:Length_of_data),L_z(Length_of_data-P_L+1:Length_of_data),'g--');
ylabel('z(t)','fontsize',12);ylim([0.1,1]);%xlabel('x(t)','fontsize',12);
title('(a)','fontsize',12)
subplot(3,3,2);
plot(L_x(Length_of_data-P_L+1:Length_of_data),Y_testing1(1,270001-P_L:270000),'r--');
ylabel('G_1(t)','fontsize',12);ylim([0.1,1]);%xlabel('x(t)','fontsize',12);
title('(b)','fontsize',12)
P_L=270000;

%状态更新参数
T=1;
p = 4;          
r = 2;
n = 2.8;
h = 0.03;

NUM_of_Virtualnodes = 12;   %虚拟神经节点数(N)
N_RC=6;
Length_of_data = Discard_steps+NUM_of_Testing+NUM_of_Training; %总数据长度
OutputNodes=1;
Range_Train=Discard_steps+1:Discard_steps+NUM_of_Training;
%初始化
X_N = zeros(NUM_of_Testing,NUM_of_Virtualnodes*N_RC);
M=-1+(1-(-1)).*rand(1,NUM_of_Virtualnodes); %生成随机掩码

tintial = M;    %初始值
Data_Test=L_z(Length_of_data-NUM_of_Testing+1:Length_of_data);
flag=1;


for ti=1:1:N_RC
    if ti==1
        Data_train = L_z(1:Discard_steps+NUM_of_Training) ;    %导入训练数据
        Data_input = L_x(1:Discard_steps+NUM_of_Training);
        J_Input = Data_input'*M;   
    else
        Data_input = Y_testing2;
        J_Input = Data_input'*M; 
    end
    X_hat=RC_t(tintial,J_Input,Discard_steps+NUM_of_Training,NUM_of_Virtualnodes,n,r,p,h) ;
    %% 单位矩阵
    II=zeros(NUM_of_Virtualnodes,NUM_of_Virtualnodes);
    for i=1:NUM_of_Virtualnodes
        for j=1:NUM_of_Virtualnodes
            if(i==j)
                II(j,i)=0.000001;
            end
        end
    end
    W2(ti,:)=Data_train(1,Discard_steps+1:Discard_steps+NUM_of_Training)*X_hat(Discard_steps+1:Discard_steps+NUM_of_Training,:)*(X_hat(Discard_steps+1:Discard_steps+NUM_of_Training,:)'*X_hat(Discard_steps+1:Discard_steps+NUM_of_Training,:)+II)^(-1);
    %% 使用原始数据集带入W求得Y
    Y_testing2=W2(ti,:)*X_hat';
    X_N(1:Discard_steps+NUM_of_Training,NUM_of_Virtualnodes*(ti-1)+1:NUM_of_Virtualnodes*ti)=X_hat;
end

II=0.000001 .* eye(NUM_of_Virtualnodes*N_RC);
W_N=Data_train(1,Discard_steps+1:Discard_steps+NUM_of_Training)*X_N(Discard_steps+1:Discard_steps+NUM_of_Training,:)*(X_N(Discard_steps+1:Discard_steps+NUM_of_Training,:)'*X_N(Discard_steps+1:Discard_steps+NUM_of_Training,:)+II)^(-1);
 
for ti=1:1:N_RC
    if ti==1
        Data_input = L_x(Length_of_data-P_L+1:Length_of_data);
        J_Input = Data_input'*M; 
    else
        Data_input = Y_testing2;
        J_Input = Data_input'*M; 
    end
    X_hat=RC_t(X_N(Discard_steps+NUM_of_Training,NUM_of_Virtualnodes*(ti-1)+1:NUM_of_Virtualnodes*ti),J_Input,NUM_of_Testing,NUM_of_Virtualnodes,n,r,p,h) ;
    Y_testing2=W2(ti,:)*X_hat';
    X_N(1:NUM_of_Testing,NUM_of_Virtualnodes*(ti-1)+1:NUM_of_Virtualnodes*ti)=X_hat;
end

Y_testing2=W_N*X_N';


figure(1);
subplot(3,1,2);
Range=50001:1:53000;
plot(Range/10,Y_testing2(1,Range),'b--',Range/10,Y_testing1(1,Range),'r--',Range/10,L_z(Length_of_data-P_L+Range),'g');

ylim([0.1,0.9]);
legend('CSTDR2','CSTDR1', 'Actual signal'); 
title('(d)','fontsize',12)
ylabel('G_i(t),U_{z}(t)');

subplot(3,1,3);
NMSE4(1,:)=(Data_Test(1,1:NUM_of_Testing)-Y_testing1(1,1:NUM_of_Testing)).^2./Data_Test(1,1:NUM_of_Testing);
NMSE5(1,:)=(Data_Test(1,1:NUM_of_Testing)-Y_testing2(1,1:NUM_of_Testing)).^2./Data_Test(1,1:NUM_of_Testing);
plot(Range/10,log10(NMSE5(1,Range)),'g--',Range/10,log10(NMSE4(1,Range)),'r--')
legend('CSTDR2','CSTDR1'); ylim([-10,-1]); title('(e)','fontsize',12)


NMSE1=NMSE(Data_Test(1,1:NUM_of_Testing),Y_testing1(1,1:NUM_of_Testing));
NMSE2=NMSE(Data_Test(1,1:NUM_of_Testing),Y_testing2(1,1:NUM_of_Testing));
NMSE3=NMSE(Y_testing1(1,1:NUM_of_Testing),Y_testing2(1,1:NUM_of_Testing));
P_L=10000;
subplot(3,3,3);
plot(L_x(Length_of_data-P_L+1:Length_of_data),Y_testing2(1,270001-P_L:270000),'b--');
ylabel('G_2(t)','fontsize',12);ylim([0.1,1]);
title('(c)','fontsize',12)

NMSE=[NMSE1,NMSE2,NMSE3]
