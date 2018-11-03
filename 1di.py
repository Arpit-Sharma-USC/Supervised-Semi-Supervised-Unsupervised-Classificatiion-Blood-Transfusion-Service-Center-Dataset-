import numpy as np
import pandas as pd
from sklearn.preprocessing import normalize
from sklearn.cluster import KMeans
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
smoter=SMOTE()
X_train_smote, Y_train_smote = sampling.fit_sample(X_train, Y_train.ravel())

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
X_combined_test = shuffle(temp, random_state=5)

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
output=kmeans.labels_
output=output.reshape(len(output),1)

x_new=kmeans.transform(X_train)

index=np.arange(0,len(x_new),1)
index=index.reshape(len(index),1)
# print(index)

index_for = np.arange(0, len(x_new), 1)
col_1=x_new[:,:1].reshape(-1)
col_2=x_new[:,1:2].reshape(-1)
data = {'id': index_for, 'Margin_dist_0': col_1,'Margin_dist_1':col_2}
df = pd.DataFrame(data=data)
# now we have cluster-wise dist; now pick the 30 closest dist

df_cluster_0 = df.sort_values(by='Margin_dist_0')
df_cluster_1 = df.sort_values(by='Margin_dist_1')

# print("top 30..")

df_30_0 = pd.DataFrame()
df_30_0 = df_cluster_0.iloc[:30]


df_30_1 = pd.DataFrame()
df_30_1 = df_cluster_1.iloc[:30]
print(Y_train[1,0])

label_0=[]
counter_0=0
counter_1=0
for i in range(0,30):
    temp_id=int(df_30_0.values[i,2])
    label = int(Y_train[temp_id,0])
    label_0.append(label)
    if label==1:
        counter_1+=1
    else:
        counter_0+=1

if counter_0>=counter_1:
    label_cluster_0=0
else:
    label_cluster_0=1


label_1=[]
counter_0=0
counter_1=0
for i in range(0,30):
    temp_id=int(df_30_1.values[i,2])
    label = int(Y_train[temp_id,0])
    label_1.append(label)
    if label==1:
        counter_1+=1
    else:
        counter_0+=1

# label_cluster_0=0
# now we have cluster-wise dist; now pick the 30 closest dist
if counter_0>=counter_1:
    label_cluster_1=0
else:
    label_cluster_1=1

missclassifications=0
classified=0
for i in range(0,len(output)):
    if int(output[i,0])==int(Y_train[i,0]):
        classified+=1
    else:
        missclassifications+=1

error_rate=missclassifications/(missclassifications+classified)

confusion_grid=confusion_matrix(Y_train,output)
print(confusion_grid)
print('\nAccuracy:',1-error_rate)
sns.heatmap(confusion_grid,cmap="YlGnBu",annot=True,linewidths=.5,fmt='d')
plt.title('Confusion Matrix - Unsupervised -Train Set')
plt.show()

# temp=np.sort(temp)
# print('end')