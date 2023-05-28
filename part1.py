from sklearn.datasets import make_blobs
from sklearn.model_selection import train_test_split
import numpy as np
import matplotlib.pyplot as plt



inputs, targets = make_blobs(n_samples = 1000, centers = [(0,0), (5,5)], n_features = 2, cluster_std = 1)
X_train, X_test, y_train, y_test = train_test_split(inputs, targets, test_size=0.33, random_state=60)


plt.scatter(X_train[:,0], X_train[:,1])
plt.title('Linearly separable data')
plt.xlabel('X1')
plt.ylabel('X2')
plt.show()

from sklearn import svm

clf = svm.SVC(kernel='linear')
clf = svm.SVC(kernel='rbf')
clf = svm.SVC(kernel='poly')


clf = clf.fit(X_train, y_train)

predictions = clf.predict(X_test)
from sklearn.metrics import plot_confusion_matrix

predictions = clf.predict(X_test)


matrix = plot_confusion_matrix(clf, X_test, y_test,
                                 cmap=plt.cm.Blues,
                                 normalize='true')
plt.title('Confusion matrix for our classifier')
    
plt.show()


support_vectors = clf.support_vectors_
plt.scatter(X_train[:,0], X_train[:,1])
plt.scatter(support_vectors[:,0], support_vectors[:,1], color='red')
plt.title('Linearly separable data with support vectors')
plt.xlabel('X1')
plt.ylabel('X2')
plt.show()

from mlxtend.plotting import plot_decision_regions

plot_decision_regions(X_test, y_test, clf=clf, legend=2)
plt.show()