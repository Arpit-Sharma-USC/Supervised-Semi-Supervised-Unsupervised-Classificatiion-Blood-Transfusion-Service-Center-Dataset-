import numpy as np
import pandas as pd
from sklearn.preprocessing import normalize
from sklearn.cluster import KMeans
from sklearn.model_selection import KFold
from sklearn import metrics
from sklearn.model_selection import GridSearchCV
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve
from sklearn.utils import shuffle
from imblearn.combine import SMOTEENN
from imblearn.over_sampling import SMOTE
from sklearn.metrics import confusion_matrix
from sklearn.metrics import auc
from sklearn.metrics import mean_squared_error
from scipy.sparse import coo_matrix, hstack
import seaborn as sns

random_state = np.random.RandomState(0)
df=pd.read_csv("/media/ghost/Games And Images/INF552/hw5/transfusion.csv")

# filter out 0s and 1s
df_0=(df.loc[df['whether'] == 0])

# pick 20% as test for class 0
X_test_0=df_0.values[:(np.int(len(df_0.values)*.20)),:4]
Y_test_0=df_0.values[:(np.int(len(df_0.values)*.20)),4:]

# pick the remaining 80% as train for class 0
X_train_0=df_0.values[(np.int(len(df_0.values)*.20)):,:4]
Y_train_0=df_0.values[(np.int(len(df_0.values)*.20)):,4:]

df_1=(df.loc[df['whether'] == 1])

# pick 20% as test for class 1
X_test_1=df_1.values[:(np.int(len(df_1.values)*.20)),:4]
Y_test_1=df_1.values[:(np.int(len(df_1.values)*.20)),4:]

# pick the remaining 80% as train for class 1
X_train_1=df_1.values[(np.int(len(df_1.values)*.20)):,:4]
Y_train_1=df_1.values[(np.int(len(df_1.values)*.20)):,4:]

# get the final train set ready
X_train_temp=np.vstack((X_train_0,X_train_1))
Y_train=np.vstack((Y_train_0,Y_train_1))

# get the final test set ready
X_test_temp=np.vstack((X_test_0,X_test_1))
Y_test=np.vstack((Y_test_0,Y_test_1))
smoter=SMOTE()
# normalize data
X_test=normalize(X_test_temp,norm='l2')

X_train=normalize(X_train_temp,norm='l2')

a=X_train
A = coo_matrix(a)
B = coo_matrix(Y_train)
temp=hstack([A,B]).toarray()
X_combined_train = shuffle(temp, random_state=1)


X_train=X_combined_train[:,:4]
Y_train=X_combined_train[:,4:]

sampling=SMOTEENN(random_state=2,kind_smote='svm')

X_train_smote, Y_train_smote = smoter.fit_sample(X_train, Y_train.ravel())

a=X_test
A = coo_matrix(a)
B = coo_matrix(Y_test)
temp=hstack([A,B]).toarray()
X_combined_test = shuffle(temp, random_state=6)


X_test=X_combined_test[:,:4]
Y_test=X_combined_test[:,4:]

X_test_smote, Y_test_smote = smoter.fit_sample(X_test, Y_test.ravel())
#
Y_test_smote=Y_test_smote.reshape(len(Y_test_smote),1)

a=X_test_smote
A = coo_matrix(a)
B = coo_matrix(Y_test_smote)
temp=hstack([A,B]).toarray()
X_combined_test = shuffle(temp, random_state=155)

# set-up the data in train and test
X_train=X_train_smote
Y_train=Y_train_smote
Y_train=Y_train.reshape(len(Y_train),1)

X_test=X_combined_test[:,:4]
Y_test=X_combined_test[:,4:]

kmeans = KMeans(n_clusters=2,n_init=20,random_state=random_state)
kmeans.fit(X_train)
print("Inertia:",kmeans.inertia_)
print("Centroids:",kmeans.cluster_centers_)
# output=kmeans.labels_
# output=output.reshape(len(output),1)
preds=kmeans.predict(X_test)
preds=preds.reshape(len(preds),1)

missclassifications = 0
classified = 0
for i in range(0,len(preds)):
    if int(preds[i, 0]) == int(Y_test[i, 0]):
        classified += 1
    else:
        missclassifications += 1

error_rate=missclassifications/(missclassifications+classified)

confusion_grid=confusion_matrix(Y_test,preds)
print(confusion_grid)
print('\nAccuracy:',1-error_rate)
sns.heatmap(confusion_grid,cmap="YlGnBu",annot=True,linewidths=.5,fmt='d')
plt.title('Confusion Matrix - Unsupervised -Test Set')
plt.show()

print("\n")
