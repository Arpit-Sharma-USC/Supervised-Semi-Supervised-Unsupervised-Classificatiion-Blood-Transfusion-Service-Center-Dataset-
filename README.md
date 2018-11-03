# Supervised, Semi-Supervised and Unsupervised Classificatiion of Blood Transfusion Service Center Dataset 

# Dataset
 https://archive. ics.uci.edu/ml/datasets/Blood+Transfusion+Service+Center
 
 # Approach
 
•	Supervised:
    
    o	L1-penalized SVM to classify the data, with 5 fold cross validation for choose the penalty parameter.

•	Semi-Supervised:

    o	Self Training- 50% of the positive class along with 50% of the negative class in the training set as labeled data and the rest as unlabelled data. 

    o	Then finding the unlabeled data point that is the closest to the decision boundary of the SVM. I then let the SVM label it (ignoring its true label), and added it to the labeled data, and retrained the SVM. I continued this process until all unlabeled data was used. I then tested the ﬁnal SVM on the test data and reported the accuracy, AUC, ROC, and confusion matrix for the test set. 

•	Unsupervised:

    o  I ran k-means algorithm on the whole training set. Ignored the labels of the data, and assumed k = 2. 

    o	I then computed the centers of the two clusters and found the closest 30 data points to each center. I then read the true labels of those 30 data points and took a majority poll within them. The majority poll becomes the label predicted by k-means for the members of each cluster. Then I compared the labels provided by kmeans with the true labels of the training data and reported accuracy and the confusion matrix.

# Comparison of all Approaches

I observed the Supervised learning is the best method that can be used in this scenario where the sample of data is very less.
Due to scarcity of the data the semi-supervised and unsupervised learning approaches do not perform well at all.

Ideally semi-supervised performs well on the train set when compared to the unsupervised learning. I believe the true powers of the unsupervised learning are realized when we have a large amount of data to deal with, on contrary to the problem-dataset.

For the given data-set Supervised learning method is the winner hands-down.


For Detailed observations and accuracy graphs refer to 'Results.pdf'
