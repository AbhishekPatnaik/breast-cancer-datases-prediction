import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

data = pd.read_csv('Datasets/breast-cancer-wisconsin.data')
data.columns=['sample_id', 'thickness', 'size', 'shape', 'adhesion', 'epithelial', 'nuclei', 'chromatin', 'nucleoli', 'mitoses', 'status']
data.head()
data['sample_id'] = data['sample_id'].astype('category').cat.codes

for i in data.columns:
    for j in data.index:
        if data[i].loc[j] == '?':
            data[i].loc[j] = 0

labels=data['status']
data=data.drop('status',axis=1)
from sklearn.preprocessing import StandardScaler
scaled_data = StandardScaler()
scaled_data = scaled_data.fit_transform(data)

print(scaled_data)

from sklearn.cross_validation import train_test_split
X_train,X_test,y_train,y_test = train_test_split(scaled_data,labels,test_size=0.33,random_state=8)


from sklearn.neighbors import KNeighborsClassifier
knn=KNeighborsClassifier(n_neighbors=42)
knn.fit(X_train,y_train)
print(knn.score(X_test,y_test))

from sklearn.ensemble import RandomForestClassifier
rf=RandomForestClassifier(n_estimators=13)
rf.fit(X_train,y_train)
print(rf.score(X_test,y_test))

from sklearn.svm import SVC
svc=SVC()
svc.fit(X_train,y_train)
print(svc.score(X_test,y_test))

from sklearn.ensemble import VotingClassifier
estimators = [('knn',knn),('rf',rf)]
eclf1=VotingClassifier(estimators=estimators,voting='soft')
eclf1.fit(X_train,y_train)
print(eclf1.score(X_test,y_test))
