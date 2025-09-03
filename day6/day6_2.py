import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score,confusion_matrix
import pickle

data_file = r'C:\Users\13394\Desktop\项目\CV\day6\data.txt'
data = np.loadtxt(data_file)
X = data[:,:-1]
Y = data[:,-1]
X_train,X_test,Y_train,Y_test = train_test_split(X,Y,test_size=0.2,random_state=42,shuffle=True,stratify=Y)

rf_classfier = RandomForestClassifier()

rf_classfier.fit(X_train,Y_train)

Y_pred = rf_classfier.predict(X_test)

accuracy = accuracy_score(Y_test,Y_pred)

print(accuracy)
print(confusion_matrix(Y_test,Y_pred))

with open(r'C:\Users\13394\Desktop\项目\CV\day6\model','wb') as f:
    pickle.dump(rf_classfier,f)