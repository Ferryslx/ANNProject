"""
ANN实现步骤：
1.构建数据集
2.搭建神经网络
3.模型训练
4.模型测试
"""

"""
优化思路：
1.优化方法从SGD->Adam
2.学习率从0.001->0.0001
3.对数据进行标准化
4.增加网络深度，每层的神经元数量
5.调整训练的轮数
6....
"""
from sklearn.preprocessing import StandardScaler
import os
from utils.log import Logger
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader        #数据集对象(数据->Tensor->Dataset->DataLoader),数据加载器
import torch.optim as optim                                   #优化器对象
from sklearn.model_selection import train_test_split          #训练集和测试集划分
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import time
import datetime
from torchsummary import summary                               #模型结构可视化

"""搭建神经网络"""
class PhonePriceModel(nn.Module):
    def __init__(self,input_dim,output_dim):
        super().__init__()
        #优化1：增加网络深度
        self.linear1=nn.Linear(input_dim,128)
        self.linear2=nn.Linear(128,256)
        self.linear3=nn.Linear(256,512)
        self.linear4=nn.Linear(512,128)
        self.output=nn.Linear(128,output_dim)
        #日志
        logfile_name = 'ANNModel' + datetime.datetime.now().strftime('%Y%m%d')
        self.logger=Logger('../',logfile_name).get_logger()

    def forward(self,x):
        #隐藏层：加权求和+激活函数（ReLU）
        x=torch.relu(self.linear1(x))
        x=torch.relu(self.linear2(x))
        x=torch.relu(self.linear3(x))
        x=torch.relu(self.linear4(x))
        #输出层：加权求和+激活函数（softmax），但是后续损失函数用多分类交叉熵损失CrossEntropyLoss()，其中已经用了softmax
        x=self.output(x)
        return x



# 1.构建数据集
data = pd.read_csv('../data/手机价格预测.csv')

x=data.iloc[:,:-1]
y=data.iloc[:,-1]

#把特征列转成浮点型
x=x.astype(float)
#print(f'x:{x.head(5)},x.shape:{x.shape}')

x_train,x_test,y_train,y_test = train_test_split(x,y,test_size = 0.2,stratify=y)

#优化2：标准化
transfer=StandardScaler()
x_train=transfer.fit_transform(x_train)
x_test=transfer.transform(x_test)


#将数据集封装成张量->dataset->dataloader
train_dataset=TensorDataset(torch.tensor(x_train,dtype=torch.float),torch.tensor(y_train.values,dtype=torch.long))
test_dataset=TensorDataset(torch.tensor(x_test,dtype=torch.float),torch.tensor(y_test.values,dtype=torch.long))

train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
test_loader  = DataLoader(test_dataset, batch_size=8, shuffle=False)

input_dim=x_train.shape[1]  #20，充当输入特征数
output_dim=len(np.unique(y))  #4,充当输出标签数


# 2.搭建神经网络（定义类的动作放在开头了）
model=PhonePriceModel(input_dim,output_dim)
logger=model.logger
logger.info(f'正在创建模型：{model}')
#计算模型参数
summary(model,input_size=(16,input_dim))#16是每一批的大小

# 3.模型训练
logger.info('开始模型训练')
criterion=nn.CrossEntropyLoss()
#优化3：调整优化方法和学习率
#optimizer=optim.SGD(model.parameters(),lr=0.001)
optimizer=optim.Adam(model.parameters(),lr=1e-4,betas=(0.9,0.99))
#定义变量记录训练的总轮数
epochs=100
#开始每轮的训练
for epoch in range(epochs):
    # 切换模型状态（随机失活Dropout在测试集不生效，所以切换状态为训练）
    model.train()
    start = time.time()
    #定义变量，记录每次训练的损失值,批次数
    total_loss,batch_num=0.0,0
    for x,y in train_loader:
        y_pred=model(x)
        loss=criterion(y_pred,y)
        optimizer.zero_grad()
        loss.sum().backward()
        optimizer.step()

        #累计损失值
        total_loss+=loss.item()*x.size(0) #把本轮的每批次(16条)的平均损失累积起来
        batch_num+=x.size(0)
    #至此，本轮训练结束，打印训练信息
    logger.info(f'epoch:{epoch+1},loss:{total_loss/batch_num:.4f},time:{time.time()-start}s')

#保存模型(参数)
#参1：模型对象的参数（权重矩阵，偏置矩阵）  参2：保存的位置
torch.save(model.state_dict(),'../model/phone.pth')
logger.info(f'模型保存成功，保存路径是:{os.path.abspath("../model/phone.pth")}')


#后续如果需要加载模型参数，代码如下
# model=PhonePriceModel(input_dim,output_dim)
# model.load_state_dict(torch.load('../model/phone.pth'))


# 4.模型测试
#定义变量，记录预测正确的样本数
logger.info('开始模型测试')
correct=0
for x,y in test_loader:
    model.eval()  #测试模式
    y_pred=model(x)
    #根据加权求和得到类别，用argmax函数获取最大值对应的下标，就是类别
    y_pred = torch.argmax(y_pred, dim=1)    #dim=1表示逐行处理
    # 计算准确率
    correct += (y_pred == y).sum().item()

print(f'准确率Accuracy:{correct/len(test_dataset)*100}%')
logger.info(f'模型准确率为Accuracy:{correct/len(test_dataset)*100}%')