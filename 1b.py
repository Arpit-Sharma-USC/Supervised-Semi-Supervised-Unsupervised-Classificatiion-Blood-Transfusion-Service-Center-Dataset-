import numpy as np
import pandas as pd
from sklearn.preprocessing import normalize
from sklearn.svm import LinearSVC
from sklearn.model_selection import KFold
from sklearn.model_selection import GridSearchCV
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve
from sklearn.utils import shuffle
from imblearn.combine import SMOTEENN, SMOTETomek
from imblearn.over_sampling import SMOTE
from sklearn.metrics import confusion_matrix
from sklearn.metrics import auc
from sklearn.metrics import mean_squared_error
from scipy.sparse import coo_matrix, hstack
import seaborn as sns

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
X_combined_train = shuffle(temp, random_state=10)


X_train=X_combined_train[:,:4]
Y_train=X_combined_train[:,4:]

sampling=SMOTEENN(random_state=2,kind_smote='svm')
smoter=SMOTE()
X_train_smote, Y_train_smote = sampling.fit_sample(X_train, Y_train.ravel())

a=X_test
A = coo_matrix(a)
B = coo_matrix(Y_test)
temp=hstack([A,B]).toarray()
X_combined_test = shuffle(temp, random_state=36)


X_test=X_combined_test[:,:4]
Y_test=X_combined_test[:,4:]

X_test_smote, Y_test_smote = smoter.fit_sample(X_test, Y_test.ravel())
#
Y_test_smote=Y_test_smote.reshape(len(Y_test_smote),1)

a=X_test_smote
A = coo_matrix(a)
B = coo_matrix(Y_test_smote)
temp=hstack([A,B]).toarray()
X_combined_test = shuffle(temp, random_state=65)


X_test=X_combined_test[:,:4]
Y_test=X_combined_test[:,4:]



cv=KFold(5)
Cs=np.linspace(0.0001,1,250)
tuned_parameters = [{'C': Cs}]

print("\n Linear SVC - Supervised Learning- Train set")
clf = GridSearchCV(LinearSVC(penalty='l1', dual=False,tol=0.001), tuned_parameters, cv=cv, refit=True, n_jobs=2)

print('Fitting...')

clf.fit(X_train_smote, Y_train_smote.ravel())

scores = clf.cv_results_['mean_test_score']
scores_std = clf.cv_results_['std_test_score']

print("Best SVM-penalty parameter is ", clf.best_params_)
print("Best Score with this parameter is", clf.best_score_)

print('Predicting...')
preds = clf.predict(X_train_smote)
print("Accuracy: ",1-mean_squared_error(Y_train_smote,preds))
# print(preds)

# y_pred_rf_lm = clf.predict_proba(X_test)
fpr_rf_lm, tpr_rf_lm, _ = roc_curve(Y_train_smote, preds,pos_label=1)
roc_auc = auc(fpr_rf_lm, tpr_rf_lm)
#
plt.plot(fpr_rf_lm, tpr_rf_lm, label=str(roc_auc))
plt.title('ROC curve- (Train-Set), AUC: '+str(roc_auc))
plt.xlabel('False positive rate')
plt.plot([0, 1], [0, 1], 'k--')
# plt.legend('AUC:'str(round(roc_auc)))
plt.ylabel('True positive rate')
print(roc_auc)
plt.show()
#
confusion_matrx=confusion_matrix(Y_train_smote,preds)
print(confusion_matrx)
sns.heatmap(confusion_matrx,cmap="YlGnBu",annot=True,linewidths=.5,fmt='d')
plt.title('Confusion Matrix - Train Set')
plt.show()

print("\n Linear SVC - Supervised Learning- Test set")
preds_test=clf.predict(X_test)
print("Accuracy: ",1-mean_squared_error(Y_test,preds_test))


# y_pred_rf_lm = clf.predict_proba(X_test)
fpr_rf_lm, tpr_rf_lm, _ = roc_curve(Y_test, preds_test,pos_label=1)
roc_auc = auc(fpr_rf_lm, tpr_rf_lm)
#
plt.plot(fpr_rf_lm, tpr_rf_lm, label=str(roc_auc))
plt.title('ROC curve- (Test-Set), AUC: '+str(roc_auc))
plt.xlabel('False positive rate')
plt.plot([0, 1], [0, 1], 'k--')
# plt.legend('AUC:'str(round(roc_auc)))
plt.ylabel('True positive rate')
print(roc_auc)
plt.show()
#
confusion_matrx=confusion_matrix(Y_test,preds_test)
print(confusion_matrx)
sns.heatmap(confusion_matrx,cmap="YlGnBu",annot=True,linewidths=.5,fmt='d')
plt.title('Confusion Matrix - Test Set')
plt.show()

