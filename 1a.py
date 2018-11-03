import numpy as np
import pandas as pd

df=pd.read_csv("/media/ghost/Games And Images/INF552/hw5/transfusion.csv")

# filter out 0s and 1s
df_0=(df.loc[df['whether'] == 0])

# pick 20% as test for class 0
X_test_0=df_0.values[:(np.int(len(df_0.values)*.20)),:4]
Y_test_0=df_0.values[:(np.int(len(df_0.values)*.20)),4:]

# pick the remaining 80% as train for class 0
X_train_0=df_0.values[(np.int(len(df_0.values)*.20)):,:4]
Y_train_0=df_0.values[:(np.int(len(df_0.values)*.20)),4:]

df_1=(df.loc[df['whether'] == 1])

# pick 20% as test for class 1
X_test_1=df_1.values[:(np.int(len(df_1.values)*.20)),:4]
Y_test_1=df_1.values[:(np.int(len(df_1.values)*.20)),4:]

# pick the remaining 80% as train for class 1
X_train_1=df_1.values[(np.int(len(df_1.values)*.20)):,:4]
Y_train_1=df_1.values[:(np.int(len(df_1.values)*.20)),4:]

# get the final train set ready
X_train=np.vstack((X_train_0,X_train_1))
Y_train=np.vstack((Y_train_0,Y_train_1))

# get the final test set ready
X_test=np.vstack((X_test_0,X_test_1))
Y_test=np.vstack((Y_test_0,Y_test_1))

