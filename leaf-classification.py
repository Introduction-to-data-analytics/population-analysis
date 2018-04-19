import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.metrics import accuracy_score, log_loss
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC, LinearSVC, NuSVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import KFold

train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')

def encode(train, test):
    le = LabelEncoder().fit(train.species) 
    labels = le.transform(train.species)           # encode species strings
    classes = list(le.classes_)                    # save column names for submission
    test_ids = test.id                             # save test ids for submission
    
    train = train.drop(['species', 'id'], axis=1)  
    test = test.drop(['id'], axis=1)
    
    return train, labels, test, test_ids, classes

train, labels, test, test_ids, classes = encode(train, test)

classifiers = [
    KNeighborsClassifier(3),
    NuSVC(probability=True),
    DecisionTreeClassifier(),
    RandomForestClassifier(),
    AdaBoostClassifier(),
    GradientBoostingClassifier(),
    GaussianNB(),
    LinearDiscriminantAnalysis(),
    QuadraticDiscriminantAnalysis()]

kf = KFold(n_splits=10)
X = np.array(train)
y = np.array(labels)
collect = pd.DataFrame(columns=['Name','Accuracy'])
for clf in classifiers:
    count = 1
    list_accuracy = []
    name = clf.__class__.__name__
    
    print("="*30)
    print(name)
    for train_index, test_index in kf.split(X):
       # print("Flod %d#" %(count))
        #Divide data in to 2 sections(train and test section)
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
        
        clf.fit(X_train, y_train)
        predicted = clf.predict(X_test)
        
        #Accuracy
        accuracy = accuracy_score(y_test, predicted)
        list_accuracy.append(accuracy)
       # print("Accuracy = %f\n-----------------------------------------" % (accuracy))
        count += 1
    
    collect = collect.append(pd.DataFrame([[name,np.array(list_accuracy).mean()*100]],columns=['Name','Accuracy']))
    print("Average accuracy = %f\n" %(np.array(list_accuracy).mean()))