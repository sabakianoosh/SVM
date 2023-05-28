# Imports
from sklearn.datasets import make_blobs
from sklearn.model_selection import train_test_split
import numpy as np
import matplotlib.pyplot as plt
from sklearn import svm
from sklearn.metrics import plot_confusion_matrix
from mlxtend.plotting import plot_decision_regions
import os, cv2


# Configuration options
train_dir1 = 'E:\\6\AI\\PJ4\\P3\\2'
train_dir2 = 'E:\\6\\AI\\PJ4\\P3\\7'

x_trainNtest1 = []
x_trainNtest2 = []
for path in os.listdir(train_dir1):
  if os.path.isfile(os.path.join(train_dir1,path)):
    x_trainNtest1.append(cv2.cvtColor(cv2.imread(f"{train_dir1}/{path}"), cv2.COLOR_RGB2GRAY))

for path in os.listdir(train_dir2):
  if os.path.isfile(os.path.join(train_dir2,path)):
    x_trainNtest2.append(cv2.cvtColor(cv2.imread(f"{train_dir2}/{path}"), cv2.COLOR_RGB2GRAY))

y_trainNtest1 = [0]*len(x_trainNtest1)
y_trainNtest2 = [1]*len(x_trainNtest2)

x_trainNtest1 = np.array(x_trainNtest1)
x_trainNtest2 = np.array(x_trainNtest2)
y_trainNtest1 = np.array(y_trainNtest1)
y_trainNtest2 = np.array(y_trainNtest2)

X = np.concatenate((x_trainNtest1,x_trainNtest2))
Y = np.concatenate((y_trainNtest1,y_trainNtest2))
print(X.shape)
X = X.reshape(-1 ,16*16) / 255

X_train, X_test, y_train, y_test = train_test_split( X, Y, test_size=0.33, random_state=42)
# Save and load temporarily
# np.save('./data.npy', (X_train, X_test, y_train, y_test))
# X_train, X_test, y_train, y_test = np.load('./data.npy', allow_pickle=True)

# Generate scatter plot for training data 
plt.scatter(X_train[:,0], X_train[:,1])
plt.title('Linearly separable data')
plt.xlabel('X1')
plt.ylabel('X2')
plt.show()

# Initialize SVM classifier
clf = svm.SVC(kernel='linear')
# clf = svm.SVC(kernel='rbf')
# clf = svm.SVC(kernel='poly')

# Fit data
clf = clf.fit(X_train, y_train)

# Predict the test set
predictions = clf.predict(X_test)

# Generate confusion matrix
matrix = plot_confusion_matrix(clf, X_test, y_test,
                                 cmap=plt.cm.Blues,
                                 normalize='true')
plt.title('Confusion matrix for our classifier')
plt.show()

# Get support vectors
support_vectors = clf.support_vectors_

# Visualize support vectors
plt.scatter(X_train[:,0], X_train[:,1])
plt.scatter(support_vectors[:,0], support_vectors[:,1], color='red')
plt.title('Linearly separable data with support vectors')
plt.xlabel('X1')
plt.ylabel('X2')
plt.show()

# Plot decision boundary
from sklearn.decomposition import PCA
from mlxtend.plotting import plot_decision_regions

pca = PCA(n_components = 2)
X_train2 = pca.fit_transform(X_train)
clf.fit(X_train2, y_train)
plot_decision_regions(X_train2, y_train, clf=clf, legend=2)
plt.show()
