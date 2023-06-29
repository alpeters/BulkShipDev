#!/usr/bin/env python
# coding: utf-8

# In[1]:


import torch
import torch.nn.functional as F
import pandas as pd
import numpy as np
from sklearn.model_selection import KFold
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import cross_val_score
from skorch import NeuralNetRegressor
from skorch.callbacks import Callback
import matplotlib.pyplot as plt
from sklearn import preprocessing
from scipy import stats
import random


# In[4]:


def set_global_seed(seed: int) -> None:
    """
    Sets random seed into PyTorch, TensorFlow, Numpy and Random.
    Args:
        seed: random seed
    """
    try:
        import torch
    except ImportError:
        pass
    else:
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    try:
        import tensorflow as tf
    except ImportError:
        pass
    else:
        if parse(tf.__version__) >= Version("2.0.0"):
            tf.random.set_seed(seed)
        elif parse(tf.__version__) <= Version("1.13.2"):
            tf.set_random_seed(seed)
        else:
            tf.compat.v1.set_random_seed(seed)
    random.seed(seed)
    np.random.seed(seed)


# In[5]:


class Net1(torch.nn.Module): 
    
    def __init__(self, input_feature, n_hidden_1, n_output):
        super(Net1, self).__init__() 
        
        self.linear1 = torch.nn.Linear(input_feature, n_hidden_1)   
        self.activation1 = torch.nn.ReLU()

        
        self.predict = torch.nn.Linear(n_hidden_1, n_output)  

    def forward(self, x):  
        
        x = self.linear1(x)
        x = self.activation1(x)
        x = self.predict(x)
        return x


class Net2(torch.nn.Module): 
    
    def __init__(self, input_feature, n_hidden_1, n_hidden_2, n_output):
        super(Net2, self).__init__() 
        
        self.linear1 = torch.nn.Linear(input_feature, n_hidden_1)   
        self.activation1 = torch.nn.ReLU()
        
        self.linear2 = torch.nn.Linear(n_hidden_1, n_hidden_2) 
        self.activation2 = torch.nn.ReLU()
        
        self.predict = torch.nn.Linear(n_hidden_2, n_output)  

    def forward(self, x):  
        
        x = self.linear1(x)
        x = self.activation1(x)
        x = self.linear2(x) 
        x = self.activation2(x)
        x = self.predict(x)
        return x
        
        
class Net3(torch.nn.Module): 
    
    def __init__(self, input_feature, n_hidden_1, n_hidden_2, n_hidden_3, n_output):
        super(Net3, self).__init__() 
        
        self.linear1 = torch.nn.Linear(input_feature, n_hidden_1)   
        self.activation1 = torch.nn.ReLU()
        
        self.linear2 = torch.nn.Linear(n_hidden_1, n_hidden_2) 
        self.activation2 = torch.nn.ReLU()
        
        self.linear3 = torch.nn.Linear(n_hidden_2, n_hidden_3) 
        self.activation3 = torch.nn.ReLU()
        
        self.predict = torch.nn.Linear(n_hidden_3, n_output)  

    def forward(self, x):  
        
        x = self.linear1(x)
        x = self.activation1(x)
        x = self.linear2(x) 
        x = self.activation2(x)
        x = self.linear3(x) 
        x = self.activation3(x)
        x = self.predict(x)
        return x


# In[6]:


model = torch.nn.Sequential(
    torch.nn.Linear(16, 12),
    torch.nn.ReLU(),
    torch.nn.Linear(12, 10),
    torch.nn.ReLU(),
    torch.nn.Linear(10, 1),
    torch.nn.Sigmoid()
)

print(model)


# In[30]:


from torchviz import make_dot

model = torch.nn.Sequential()
model.add_module('W0', torch.nn.Linear(9, 1))
model.add_module('relu1', torch.nn.ReLU())
model.add_module('W1', torch.nn.Linear(1, 5))
model.add_module('relu2', torch.nn.ReLU())
model.add_module('W2', torch.nn.Linear(5, 9))
model.add_module('relu3', torch.nn.ReLU())
model.add_module('W3', torch.nn.Linear(9, 1))

x = torch.randn(1, 9)
y = model(x)

g = make_dot(y.mean(), params=dict(model.named_parameters()), show_attrs=True, show_saved=True)
g.view()


# In[30]:


train = pd.read_csv("/Users/oliver/Desktop/data/train.csv")
test = pd.read_csv("/Users/oliver/Desktop/data/test.csv")

mappings = {
  'Capesize': 0,
  'Handymax': 1,
  'Handysize': 2,
  'Panamax': 3}


train['Size.Category'] = train['Size.Category'].apply(lambda x: mappings[x])
test['Size.Category'] = test['Size.Category'].apply(lambda x: mappings[x])

train[['Dwt','Main.Engine.Power.kW','LBP..m.', 'Beam.Mld..m.', 'Draught..m.', 'MRV.Load', 'TPC', 'FC.Per.Travel.Work']] = np.log10(train[['Dwt','Main.Engine.Power.kW','LBP..m.', 'Beam.Mld..m.', 'Draught..m.', 'MRV.Load', 'TPC', 'FC.Per.Travel.Work']])
test[['Dwt','Main.Engine.Power.kW','LBP..m.', 'Beam.Mld..m.', 'Draught..m.', 'MRV.Load', 'TPC', 'FC.Per.Travel.Work']] = np.log10(test[['Dwt','Main.Engine.Power.kW','LBP..m.', 'Beam.Mld..m.', 'Draught..m.', 'MRV.Load', 'TPC', 'FC.Per.Travel.Work']])

scaler = preprocessing.StandardScaler()

train[['Age', 'Dwt','Main.Engine.Power.kW','LBP..m.', 'Beam.Mld..m.', 'Draught..m.', 'MRV.Load', 'TPC']] = scaler.fit_transform(train[['Age', 'Dwt','Main.Engine.Power.kW','LBP..m.', 'Beam.Mld..m.', 'Draught..m.', 'MRV.Load', 'TPC']])
test[['Age', 'Dwt','Main.Engine.Power.kW','LBP..m.', 'Beam.Mld..m.', 'Draught..m.', 'MRV.Load', 'TPC']] = scaler.fit_transform(test[['Age', 'Dwt','Main.Engine.Power.kW','LBP..m.', 'Beam.Mld..m.', 'Draught..m.', 'MRV.Load', 'TPC']])

x_train = train[['Age', 'Dwt','Main.Engine.Power.kW','LBP..m.', 'Beam.Mld..m.', 'Draught..m.', 'MRV.Load', 'TPC', 'Size.Category']]
y_train = train['FC.Per.Travel.Work']
x_test = test[['Age', 'Dwt','Main.Engine.Power.kW','LBP..m.', 'Beam.Mld..m.', 'Draught..m.', 'MRV.Load', 'TPC', 'Size.Category']]
y_test = test['FC.Per.Travel.Work']

X_train = np.array(x_train, dtype=np.float32)
y_train = np.array(y_train.values.reshape(-1, 1),dtype=np.float32)

X_test = np.array(x_test, dtype=np.float32)
y_test = np.array(y_test.values.reshape(-1, 1),dtype=np.float32)


# In[445]:


def create_model(neurons):
    model = Net1(input_feature=9, n_hidden_1=neurons, n_output=1) 
    return model

def create_model2(neurons1, neurons2):
    model = Net2(input_feature=9, n_hidden_1=neurons1, n_hidden_2=neurons2, n_output=1) 
    return model

def create_model3(neurons1, neurons2, neurons3):
    model =  Net3(input_feature=9, n_hidden_1=neurons1, n_hidden_2=neurons2, n_hidden_3=neurons3, n_output=1)
    return model


kf= KFold(n_splits=5,shuffle=True, random_state=0)
score_list=[]
for x in range(1,10):
    
    model1 = create_model(x)
    set_global_seed(1)
    net = NeuralNetRegressor(model1, 
                             optimizer=torch.optim.SGD,
                             predict_nonlinearity='auto',
                             device='cpu',
                             verbose=0)
    
    score = cross_val_score(net, X_train, y_train, cv=kf, scoring='neg_mean_squared_error')
    print(f'For 1 hidden layer with', x, 'neurons:')
    print('')
    print(f'Score for each fold is:', abs(score))
    print(f'Average score is:', abs(score).mean())
    print('')
    score_list.append(abs(score).mean())
    
    for y in range(1,10):
        
        
        model2 = create_model2(x, y)
        set_global_seed(1)
        net = NeuralNetRegressor(model2, 
                                 optimizer=torch.optim.SGD,
                                 predict_nonlinearity='auto',
                                 device='cpu',
                                 verbose=0)
        
        score = cross_val_score(net, X_train, y_train, cv=kf, scoring='neg_mean_squared_error')
        print(f'For 2 hidden layers with', x, 'neurons and', y, 'neurons:')
        print('')
        print(f'Score for each fold is:', abs(score))
        print(f'Average score is:', abs(score).mean())
        print('')
        score_list.append(abs(score).mean())
        
        for z in range(1,10):
            
            model3 = create_model3(x,y,z)
            set_global_seed(1)
            net = NeuralNetRegressor(model3, 
                                     optimizer=torch.optim.SGD,
                                     predict_nonlinearity='auto',
                                     device='cpu',
                                     verbose=0)
            
            score = cross_val_score(net, X_train, y_train, cv=kf, scoring='neg_mean_squared_error')
            print(f'For 3 hidden layers with', x, 'neurons and', y, 'neurons and', z, 'neurons')
            print('')
            print(f'Score for each fold is:', abs(score))
            print(f'Average score is:', abs(score).mean())
            print('')
            score_list.append(abs(score).mean())
            
small = min(score_list)
spot = score_list.index(small)
print(spot+1, 'th')
print(small)


# In[446]:


final_model = create_model3(1,5,9)

scores = []
batch_size = [1, 2, 4, 8, 16, 32, 64, 128, 256]

for i in batch_size:
    set_global_seed(1)
    net = NeuralNetRegressor(final_model, 
                             optimizer=torch.optim.SGD,
                             batch_size=i,
                             predict_nonlinearity='auto',
                             device='cpu',
                             verbose=0)
    
    score = cross_val_score(net, X_train, y_train, cv=kf, scoring='neg_mean_squared_error')
    scores.append(abs(score).mean())
    
min_mse = min(scores) 
x = batch_size[scores.index(min_mse)]
print(f'Most optimal batch size is:', x)
print(f'It has MSE of:', min_mse)


plt.title("Effect of Batch Size")
plt.xlabel("Batch Size")
plt.ylabel("Score")
plt.plot(range(len(batch_size)), scores, marker='o')
plt.xticks(range(len(batch_size)),batch_size)
plt.show()


# In[ ]:


final_model = create_model3(1,5,9)

set_global_seed(1)
net = NeuralNetRegressor(final_model, 
                         optimizer=torch.optim.SGD,
                         predict_nonlinearity='auto',
                         device='cpu',
                         verbose=0)

params = {
    'batch_size': [16, 32, 64, 128, 256, 512, 1024],
    'lr': [1, 0.1, 0.01, 0.001, 0.0001, 0.00001, 0.000001, 0.0000001],
    'max_epochs': np.arange(10, 1001, 30)
}

gs = GridSearchCV(net, params, refit=True, scoring='neg_mean_squared_error', verbose=1, cv=kf)

gs.fit(X_train, y_train)
print('Best configuration is:', gs.best_params_)
print('Best configuration has MSE of:', gs.best_score_)

y_pred = gs.predict(X_test)
slope, intercept, r_value, p_value, std_err = stats.linregress(y_test[:,0], y_pred[:,0])
r_value


# In[71]:


def create_model3(neurons1, neurons2, neurons3):
    model =  Net3(input_feature=9, n_hidden_1=neurons1, n_hidden_2=neurons2, n_hidden_3=neurons3, n_output=1)
    return model

final_model = create_model3(1,5,9)

set_global_seed(1)
net = NeuralNetRegressor(final_model, 
                         optimizer=torch.optim.SGD,
                         criterion=torch.nn.MSELoss,
                         lr=0.0001,
                         batch_size=16,
                         max_epochs=200,
                         predict_nonlinearity='auto',
                         warm_start=False,
                         device='cpu',
                         verbose=1)

net.fit(X_train, y_train)


train_loss = net.history[:, 'train_loss']
valid_loss = net.history[:, 'valid_loss']

plt.plot(train_loss, 'o-', label='training')
plt.plot(valid_loss, 'o-', label='validation')
plt.legend()
plt.show()


# In[65]:


y_pred = net.predict(X_test)
y_pred


# In[61]:


slope, intercept, r_value, p_value, std_err = stats.linregress(y_test[:,0], y_pred[:,0])
r_value


# In[336]:


# Hyper-parameters (to tune later)
num_epochs = 100
batch_size = 8
learning_rate = 0.1

train = torch.utils.data.TensorDataset(x_train, y_train)
test = torch.utils.data.TensorDataset(x_test, y_test)



# Data loader
train_loader = torch.utils.data.DataLoader(dataset=train,
                                           batch_size=batch_size, 
                                           shuffle=True)

validation_loader = torch.utils.data.DataLoader(dataset=test,
                                                batch_size=batch_size, 
                                                shuffle=False)

optimizer = torch.optim.SGD(model1.parameters(), lr=learning_rate)
loss_func = torch.nn.MSELoss()  

loss_list = []

for epoch in range(num_epochs):
    for batch_idx, (features, targets) in enumerate(train_loader):
        
        inputs, labels = (features, targets)
        
        optimizer.zero_grad()
        
        outputs = model1(inputs)
        
        loss = loss_func(outputs, labels)
        loss.backward()
        
        optimizer.step()
        
    print(loss)

