#直接调用模型
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
import numpy as np
X_train=np.random.rand(1000,1)*4-2
y_train=np.random.rand(1000)
model=LinearRegression()
model.fit(X_train,y_train)
y_pre=model.predict(X_train)
plt.plot(X_train,y_pre,color="red",label="Regression Linear")
plt.title("LinearRegression")
plt.grid()
plt.scatter(X_train,y_train)
plt.show()



#线性模型基础实现
import pandas as pd
import numpy as np

#下载数据集
data_url = "http://lib.stat.cmu.edu/datasets/boston"
raw_df = pd.read_csv(data_url, sep="\s+", skiprows=22, header=None)
data = np.hstack([raw_df.values[::2, :], raw_df.values[1::2, :2]])
target = raw_df.values[1::2, 2]

#完成训练测设集划分
train_al=0.8#占比
train_num=int(train_al*data.shape[0])
index=np.arange(data.shape[0])
np.random.shuffle(index)

train_index=index[:train_num]
test_index=index[train_num:]

X_train=[data[i] for i in train_index]
X_train=np.array(X_train)
y_train=[target[i] for i in train_index]
y_train=np.array(y_train)
X_test=[data[i] for i in test_index]
X_test=np.array(X_test)
y_test=[target[i] for i in test_index]
y_test=np.array(y_test)



def model(X,ww):
    return np.dot(X,ww)

def loss(X,y,ww):
    loss=np.sum((model(X,ww)-y)**2)/len(X)/2
    return loss
def backward(X,y,ww,alpha):
    ww=ww-alpha/len(X)*np.dot(X.transpose(),model(X,ww)-y)
    return ww

ww=np.zeros((data.shape[1])).T
echo=10000
alpha=0.00000085

for i in range(echo):
    print(f"{i}次的损失为：",loss(X_train,y_train,ww))
    ww=backward(X_train,y_train,ww,alpha)
