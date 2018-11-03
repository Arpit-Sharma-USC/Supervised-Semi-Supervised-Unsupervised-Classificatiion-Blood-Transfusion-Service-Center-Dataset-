import numpy as np
import pandas as pd
from sklearn.preprocessing import normalize
from sklearn.svm import LinearSVC
from sklearn.model_selection import KFold
from sklearn import metrics
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


x_0=int(len(X_train_0)*0.50)

X_train_0_labelled=X_train_0[x_0:,:]
Y_train_0_labelled=Y_train_0[x_0:,:]

X_train_0_unlabelled=X_train_0[:x_0,:]
Y_train_0_unlabelled=Y_train_0[:x_0,:]


# pick 20% as test for class 1
X_test_1=df_1.values[:(np.int(len(df_1.values)*.20)),:4]
Y_test_1=df_1.values[:(np.int(len(df_1.values)*.20)),4:]

# pick the remaining 80% as train for class 1
X_train_1=df_1.values[(np.int(len(df_1.values)*.20)):,:4]
Y_train_1=df_1.values[(np.int(len(df_1.values)*.20)):,4:]


x_1=int(len(X_train_1)*0.50)

X_train_1_labelled=X_train_1[x_1:,:]
Y_train_1_labelled=Y_train_1[x_1:,:]

X_train_1_unlabelled=X_train_1[:x_1,:]
Y_train_1_unlabelled=Y_train_1[:x_1,:]


# get the final train set ready
X_train_temp=np.vstack((X_train_0_labelled,X_train_1_labelled))
Y_train_labelled=np.vstack((Y_train_0_labelled,Y_train_1_labelled))


X_train_temp_unlabelled=np.vstack((X_train_0_unlabelled,X_train_1_unlabelled))
Y_train_unlabelled=np.vstack((Y_train_0_unlabelled,Y_train_1_unlabelled))


X_train_unlabelled=normalize(X_train_temp_unlabelled,norm='l2')


a=X_train_unlabelled
A = coo_matrix(a)
B = coo_matrix(Y_train_unlabelled)
temp=hstack([A,B]).toarray()
X_combined_train_unlabelled = shuffle(temp, random_state=10)


X_train_unlabelled=X_combined_train_unlabelled[:,:4]
Y_train_unlabelled=X_combined_train_unlabelled[:,4:]

sampling=SMOTEENN(random_state=2,kind_smote='svm')
smoter=SMOTE()
X_train_unlabelled_smote, Y_train_unlabelled_smote = sampling.fit_sample(X_train_unlabelled, Y_train_unlabelled.ravel())



# get the final test set ready
X_test_temp=np.vstack((X_test_0,X_test_1))
Y_test=np.vstack((Y_test_0,Y_test_1))

# normalize data
X_test=normalize(X_test_temp,norm='l2')

X_train_labelled=normalize(X_train_temp,norm='l2')

a=X_train_labelled
A = coo_matrix(a)
B = coo_matrix(Y_train_labelled)
temp=hstack([A,B]).toarray()
X_combined_train_labelled = shuffle(temp, random_state=10)


X_train_labelled=X_combined_train_labelled[:,:4]
Y_train_labelled=X_combined_train_labelled[:,4:]

sampling=SMOTEENN(random_state=2,kind_smote='svm')
smoter=SMOTE()
X_train_labelled_smote, Y_train_labelled_smote = sampling.fit_sample(X_train_labelled, Y_train_labelled.ravel())

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


global_model=0
cv=KFold(5)
Cs=np.linspace(0.0001,10,50)
tuned_parameters = [{'C': Cs}]
stop=len(X_train_unlabelled_smote)
for i in range(1, stop):


    print("\n Linear SVC - Semi-Supervised Learning- Train set-Labelled iteration:",i)
    clf = GridSearchCV(LinearSVC(penalty='l1', dual=False,tol=0.001), tuned_parameters, cv=cv, refit=True, n_jobs=2)

    print('Fitting...')

    clf.fit(X_train_labelled_smote, Y_train_labelled_smote.ravel())

    scores = clf.cv_results_['mean_test_score']
    scores_std = clf.cv_results_['std_test_score']

    print("Best SVM-penalty parameter is ", clf.best_params_)
    print("Best Score with this parameter is", clf.best_score_)
    global_model = clf
    #
    # # from here
    #

    X_margin = clf.decision_function(X_train_unlabelled_smote)
    X_margin = np.abs(X_margin)

    index_for = np.arange(0, len(X_margin), 1)

    data = {'id': index_for, 'Margin_dist': X_margin}
    df = pd.DataFrame(data=data)
    df = df.sort_values(by='Margin_dist')

    # print("top 1..")

    df_10 = pd.DataFrame()
    df_10 = df.iloc[:1]
    # marg = np.int(df_10.values[0, 0])
    # print(marg)
    check_arr = X_train_labelled_smote
    check_Y_arr = Y_train_labelled_smote

    to_use_indices = []
    temp_X = X_train_unlabelled_smote
    temp_Y = Y_train_unlabelled_smote
    for j in range(0, 1):
        temp = np.int(df_10.values[j, 1])
        to_use_indices.append(temp)
        check_arr = np.vstack((check_arr, np.array(X_train_unlabelled_smote[temp].reshape(1, 4))))
        # print(check_Y_arr.shape)
        # print(Y_train_unlabelled_smote[temp].reshape(1, 1).shape)
        print("index:",temp)
        my_label = clf.predict(X_train_unlabelled_smote[temp:temp+1,:])
        Y_train_unlabelled_smote[temp]=my_label
        check_Y_arr = np.vstack((check_Y_arr.reshape(len(check_Y_arr),1), (Y_train_unlabelled_smote[temp].reshape(1, 1))))

    # print(check_arr.shape)
    temp_X = np.delete(temp_X, to_use_indices, axis=0)
    temp_Y = np.delete(temp_Y, to_use_indices, axis=0)
    # print(temp_X)
    # print(to_use_indices)

    # overwrite X_train and Y_train
    X_train_labelled_smote = check_arr
    Y_train_labelled_smote = check_Y_arr

    # remove from Rest
    Y_train_unlabelled_smote = temp_Y
    X_train_unlabelled_smote = temp_X
    print(len(X_train_unlabelled_smote))
    if i == stop-1:
        # print("\n Linear SVC - Semi-Supervised Learning- Test set :")
        # preds_test = clf.predict(X_train_labelled_smote)
        # print("Accuracy: ", 1 - mean_squared_error(Y_train_labelled_smote, preds_test))
        # # y_pred_rf_lm = clf.predict_proba(X_test)
        # fpr_rf_lm, tpr_rf_lm, _ = roc_curve(Y_train_labelled_smote, preds_test,pos_label=1)
        # roc_auc = auc(fpr_rf_lm, tpr_rf_lm)
        # #
        # plt.plot(fpr_rf_lm, tpr_rf_lm, label=str(roc_auc))
        # plt.title('ROC curve- (Test-Set), AUC: '+str(roc_auc))
        # plt.xlabel('False positive rate')
        # plt.plot([0, 1], [0, 1], 'k--')
        # # plt.legend('AUC:'str(round(roc_auc)))
        # plt.ylabel('True positive rate')
        # print(roc_auc)
        # plt.show()
        # #
        # confusion_matrx=confusion_matrix(Y_train_labelled_smote,preds_test)
        # print(confusion_matrx)
        # sns.heatmap(confusion_matrx,cmap="YlGnBu",annot=True,linewidths=.5,fmt='d')
        # plt.title('Confusion Matrix - Test Set')
        # plt.show()
        #
        print("\n Linear SVC - Supervised Learning- Test set")
        preds_test = clf.predict(X_test)
        print("Accuracy: ", 1 - mean_squared_error(Y_test, preds_test))

        # y_pred_rf_lm = clf.predict_proba(X_test)
        fpr_rf_lm, tpr_rf_lm, _ = roc_curve(Y_test, preds_test, pos_label=1)
        roc_auc = auc(fpr_rf_lm, tpr_rf_lm)
        #
        plt.plot(fpr_rf_lm, tpr_rf_lm, label=str(roc_auc))
        plt.title('ROC curve- (Test-Set), AUC: ' + str(roc_auc))
        plt.xlabel('False positive rate')
        plt.plot([0, 1], [0, 1], 'k--')
        # plt.legend('AUC:'str(round(roc_auc)))
        plt.ylabel('True positive rate')
        print(roc_auc)
        plt.show()
        #
        confusion_matrx = confusion_matrix(Y_test, preds_test)
        print(confusion_matrx)
        sns.heatmap(confusion_matrx, cmap="YlGnBu", annot=True, linewidths=.5, fmt='d')
        plt.title('Confusion Matrix - Test Set')
        plt.show()

print('end')

#
