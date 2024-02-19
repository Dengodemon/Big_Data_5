#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression 
import sklearn
from sklearn import datasets

data = datasets.load_wine(as_frame = True)
predictors = data.data
target = data.target
target_names = data.target_names
print(predictors.head(9), '\n\n\Целевая переменная')
print(target.head(9))
print('Название классов:\n', target_names)


# In[2]:


from sklearn.linear_model import LinearRegression 
import sklearn
from sklearn import datasets

class_counts = data['target'].value_counts()
plt.bar(class_counts.index, class_counts.values, color=['blue', 'orange'])
plt.ylabel('Количество')
plt.title('Баланс классов')
custom_labels = ['Класс 1', 'Класс 2', 'Класс 3']
plt.xticks(class_counts.index, labels=custom_labels)
plt.show()


# In[3]:


from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(predictors, target, train_size=0.8, shuffle = True, random_state=111)
print(x_train.shape, x_test.shape, y_train.shape, y_test.shape)


# In[4]:


from sklearn.linear_model import LogisticRegression

model = LogisticRegression(max_iter = 10000, random_state = 111)
model.fit(x_train, y_train)
y_predict = model.predict(x_test)
print('Предсказанные значения: \n',y_predict)
print('Исходные значения \n',np.array(y_test))


# In[5]:


import plotly.express as px
from sklearn.metrics import confusion_matrix

plt.rcParams['figure.figsize'] = (10, 10)
fig = px.imshow(confusion_matrix(y_test, y_predict), text_auto=True)
fig.update_layout( xaxis_title = 'Target', yaxis_title = 'Prediction')


# In[6]:


from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV

param_kernel = ('linear', 'rbf', "poly", "sigmoid")
parameters = {'kernel':param_kernel}
model = SVC()
grid_search_svm = GridSearchCV(estimator=model, param_grid=parameters, cv = 6)
grid_search_svm.fit(x_train, y_train)


# In[7]:


best_model = grid_search_svm.best_estimator_
best_model.kernel


# In[8]:


import plotly.express as px
from sklearn.metrics import confusion_matrix

svm_preds = best_model.predict(x_test)
plt.rcParams['figure.figsize'] = (10, 10)
fig = px.imshow(confusion_matrix(svm_preds, y_test), text_auto=True)
fig.update_layout( xaxis_title = 'Target', yaxis_title = 'Prediction')


# In[9]:


from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV

number_of_neighbors = np.arange(3, 10)
model_KNN = KNeighborsClassifier()
params = {"n_neighbors": number_of_neighbors}
grid_search = GridSearchCV(estimator = model_KNN,param_grid = params, cv = 6)
grid_search.fit(x_train, y_train)


# In[10]:


grid_search.best_score_


# In[11]:


grid_search.best_estimator_


# In[12]:


import plotly.express as px
from sklearn.metrics import confusion_matrix

knn_preds = grid_search.predict(x_test)
plt.rcParams['figure.figsize'] = (10, 10)
fig = px.imshow(confusion_matrix(knn_preds, y_test), text_auto=True)
fig.update_layout( xaxis_title = 'Target', yaxis_title = 'Prediction')


# In[13]:


from sklearn.metrics import classification_report
print(classification_report(y_test, y_predict))
print(classification_report(svm_preds, y_test))
print(classification_report(knn_preds, y_test))

