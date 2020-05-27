import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn import datasets, svm
import numpy as np

def plot_scattered(X, y, title):
    """Pots special scatter plot"""

    fig, axes = plt.subplots(nrows=1, ncols=1)
    axes.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.Set1, edgecolor='k')
    axes.set_xlabel('Sepal length')
    axes.set_ylabel('Sepal width')
    axes.set_xticks(())
    axes.set_yticks(())
    axes.set_title(title)

    return fig, axes

def plot_svm(X, y, clf, title):
    """Plots SVM results"""

    # Plot two class data with SVM result
    fig, axes = plot_scattered(X, y, title)

    # plot the decision function
    xlim = axes.get_xlim()
    ylim = axes.get_ylim()

    # create grid to evaluate model
    xx = np.linspace(xlim[0], xlim[1], 30)
    yy = np.linspace(ylim[0], ylim[1], 30)
    YY, XX = np.meshgrid(yy, xx)
    xy = np.vstack([XX.ravel(), YY.ravel()]).T
    Z = clf.decision_function(xy).reshape(XX.shape)

    # plot decision boundary and margins
    axes.contour(XX, YY, Z, colors='k', levels=[-1, 0, 1], alpha=0.5,
               linestyles=['--', '-', '--'])
    # plot support vectors
    axes.scatter(clf.support_vectors_[:, 0], clf.support_vectors_[:, 1], s=100,
               linewidth=1, facecolors='none', edgecolors='k')

    return fig, axes

################################################################################

# Import Iris data set
# Select two dimensions that give convenient data
iris = datasets.load_iris()
X = iris.data[:, [1, 3]]
y = iris.target

################################################################################

# Plot raw data
plot_scattered(X, y, 'Raw data')

################################################################################

# Convert to two two classes
newy = []
for t in y:
    if t==0: newy.append(0)
    else: newy.append(1)

# Plot two class data
#plot_scattered(X, newy, 'Separable two-class problem')

################################################################################

# Linear SVM
clf = svm.SVC(kernel='linear', C=1000)
clf.fit(X, newy)
plot_svm(X, newy, clf, 'Separable two-class problem - Linear SVM')

# Poly SVM
#clf = svm.SVC(kernel='poly')
#clf.fit(X, y)
#plot_svm(X, y, clf, 'Two-class problem - Polynomial SVM')
#
## RBF SVM
#clf = svm.SVC(kernel='rbf')
#clf.fit(X, y)
#plot_svm(X, y, clf, 'Two-class problem - RBF SVM')

################################################################################

# Predict some points
xtest = np.array([
    [2.9, 1.7],
    [4.3, 2.],
    [3.15, 0.95],
    [3.05, 0.6],
    [3.6, 0.3],
    [4.4, 1.]
])
ytest = clf.predict(xtest)

fig, axes = plot_svm(X, newy, clf, 'Separable two-class problem - Linear SVM')
axes.scatter(xtest[:, 0], xtest[:, 1], marker='x', c=ytest, cmap=plt.cm.Set1, edgecolor='k')

################################################################################

# Convert targest to two-class non-separable situation
newy = []
for t in y:
    if t==2: newy.append(0)
    else: newy.append(1)

# Plot two class data
#plot_scattered(X, newy, 'Non-separable two-class problem')

################################################################################

# Linear SVM
clf = svm.SVC(kernel='linear', C=1000)
clf.fit(X, newy)
#plot_svm(X, newy, clf, 'Non-separable two-class problem - Linear SVM')

## Poly SVM
#clf = svm.SVC(kernel='poly')
#clf.fit(X, newy)
#plot_svm(X, newy, clf, 'Non-separable two-class problem - Polynomial SVM')
#
## RBF SVM
#clf = svm.SVC(kernel='rbf')
#clf.fit(X, newy)
#plot_svm(X, newy, clf, 'Non-separable two-class problem - RBF SVM')

################################################################################

plt.show()
