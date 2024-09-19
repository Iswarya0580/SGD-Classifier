# SGD-Classifier
## AIM:
To write a program to predict the type of species of the Iris flower using the SGD Classifier.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
```
1)Load the Iris Dataset

2)Split Data: Separate features (x) and labels (y), then split the data into training and test sets using train_test_split().

3)Train Model: Initialize an SGDClassifier and fit it on the training data (x_train, y_train).

4)Predict and Evaluate: Use the trained model to predict the labels for the test set and calculate the accuracy score.

5)Confusion Matrix: Generate and print a confusion matrix to evaluate the model's performance.
```
## Program:
```
/*
Program to implement the prediction of iris species using SGD Classifier.
Developed by: Iswarya P
RegisterNumber: 212223230082 
*/

import pandas as pd
from sklearn.datasets import load_iris
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score,confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

iris=load_iris()

df=pd.DataFrame(data=iris.data,columns=iris.feature_names)
df['target']=iris.target

print(df.head())

x=df.drop('target',axis=1)
y=df['target']

x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=42)

sgd_clf=SGDClassifier(max_iter=1000,tol=1e-3)

sgd_clf.fit(x_train,y_train)

y_pred=sgd_clf.predict(x_test)

accuracy=accuracy_score(y_test,y_pred)
print(f"Accuracy: {accuracy:.3f}")

cm=confusion_matrix(y_test,y_pred)
print("Confusion Matrix:")
print(cm)
```

## Output:
## x and y values:
![image](https://github.com/user-attachments/assets/19159f11-d079-4148-8d07-86d22d28d12c)

## Accuracy:
![image](https://github.com/user-attachments/assets/9e931bb7-4c6a-429a-bee7-7ea034876ea6)

## Confusion Matrix:
![image](https://github.com/user-attachments/assets/030c2897-7223-477a-9981-7486c5bf69b2)


## Result:
Thus, the program to implement the prediction of the Iris species using SGD Classifier is written and verified using Python programming.
