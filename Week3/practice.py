import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

df = pd.read_csv('fraud_data.csv')

X = df.iloc[:,:-1]
y = df.iloc[:,-1]

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)



def answer_four():
    from sklearn.metrics import confusion_matrix
    from sklearn.svm import SVC
    

    

    svm = SVC(C= 1e9, gamma = 1e-07).fit(X_train, y_train)
    y_svm_predictions = svm.decision_function(X_test) > -220

    confusion = confusion_matrix(y_test, y_svm_predictions)

    """
    



    
    svm = SVC(gamma=1e-07,C=1e9).fit(X_train, y_train)
    prediction = svm.decision_function(X_test) > -220
    confusion = confusion_matrix(y_test, prediction)

    """
    
    return confusion






    

print(answer_four())




