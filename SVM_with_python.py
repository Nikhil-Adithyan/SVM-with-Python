# Importing packages

import pandas as pd # data processing
import numpy as np # working with arrays
import itertools # specialized tools
import seaborn as sb # visualization
import matplotlib.pyplot as plt # visualization
from sklearn.svm import SVC # SVM model algorithm
from sklearn.model_selection import train_test_split # splitting the data
from sklearn.datasets.samples_generator import make_circles # sample data
from mpl_toolkits import mplot3d # 3D plot
from termcolor import colored as cl # text customization

sb.set_style('whitegrid')
plt.rcParams['figure.figsize'] = (20, 10)

# LINEARLY SEPERABLE DATA

df = sb.load_dataset('iris')
sb.pairplot(df, hue = 'species', palette = 'bright')
plt.savefig('pairplot.png')

# Processing & visualizing data

df = df[(df['species'] != 'virginica')]
df = df.drop(['sepal_length','sepal_width'], axis = 1)

for i in df.species.values:
    if i == 'setosa':
        df.replace(i, 0, inplace = True)
    elif i == 'versicolor':
        df.replace(i, 1, inplace = True)

X = df.iloc[:, 0:2]
y = df['species']

sb.scatterplot(X.iloc[:, 0], X.iloc[:, 1], hue = y, s = 200, palette = 'spring')
plt.legend(loc = 'upper left', fontsize = 14)
plt.title('PETAL LENGTH / PETAL WIDTH', fontsize = 17)
plt.xlabel('Petal length', fontsize = 14)
plt.ylabel('Petal width', fontsize = 14)
plt.xticks(fontsize = 12)
plt.yticks(fontsize = 12)

plt.savefig('scatter.png')
plt.show()

# Dependent & Independent variable

X_var = np.asarray(X)
y_var = np.asarray(y)

print(cl('X_var samples : ', attrs = ['bold']), X_var[:5])
print(cl('y_var samples : ', attrs = ['bold']), y_var[:5])

# splitting the data

X_train, X_test, y_train, y_test = train_test_split(X_var, y_var, test_size = 0.3, random_state = 4)

print(cl('X_train samples : ', attrs = ['bold']), X_train[:5])
print(cl('X_test samples : ', attrs = ['bold']), X_test[:5])
print(cl('y_train samples : ', attrs = ['bold']), y_train[:5])
print(cl('y_test samples : ', attrs = ['bold']), y_test[:5])

# Modelelling

model = SVC(kernel = 'linear', C = 1E10)
model.fit(X_train, y_train)
yhat = model.predict(X_test)
support_vectors = model.support_vectors_

print(cl('yhat samples : ', attrs = ['bold']), yhat[:10])
print(cl('Support vectors : ', attrs = ['bold']), support_vectors)

# Visualizing the model

ax = plt.gca()
sb.scatterplot(X_var[:, 0], X_var[:, 1], hue = y, s = 200, palette = 'spring', legend = False)
plt.title('LINEARLY SEPERABLE SVM MODEL', fontsize = 18)
plt.xlabel('Petal Length', fontsize = 14)
plt.ylabel('Petal Width', fontsize = 14)
plt.xticks(fontsize = 12)
plt.yticks(fontsize = 12)
xlim = ax.get_xlim()
ylim = ax.get_ylim()

xx = np.linspace(xlim[0], xlim[1], 30)
yy = np.linspace(ylim[0], ylim[1], 30)
YY, XX = np.meshgrid(yy, xx)
xy = np.vstack([XX.ravel(), YY.ravel()]).T
Z = model.decision_function(xy).reshape(XX.shape)

ax.contour(XX, YY, Z, colors = 'k', levels = [-1, 0, 1], alpha = 0.5,
           linestyles = ['--', '-', '--'])

ax.scatter(model.support_vectors_[:, 0], model.support_vectors_[:, 1], s = 200,
           linewidth = 1, facecolors = 'none', edgecolors = 'b')

plt.savefig('svm_linear.png')
plt.show()

# NON-LINEARLY SEPERABLE DATA

X, y = make_circles(200, factor = 0.1, noise = 0.1)

print(cl('X samples : ', attrs = ['bold']), X[:10])
print(cl('y samples : ', attrs = ['bold']), y[:10])

# Visualizing data

sb.scatterplot(X[:, 0], X[:, 1], s = 200, hue = y, edgecolor = 'b', linewidth = 2, palette = 'spring', alpha = 0.6)
plt.xticks(fontsize = 12)
plt.yticks(fontsize = 12)

plt.savefig('non_scatter.png')
plt.show()

# Splitting the data

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 4)

print(cl('X_train samples : ', attrs = ['bold']), X_train[:5])
print(cl('X_test samples : ', attrs = ['bold']), X_test[:5])
print(cl('y_train samples : ', attrs = ['bold']), y_train[:5])
print(cl('y_test samples : ', attrs = ['bold']), y_test[:5])

# Modelling

model = SVC(kernel = 'rbf')
model.fit(X_train, y_train)
yhat = model.predict(X_test)
support_vectors = model.support_vectors_

print(cl('yhat samples : ', attrs = ['bold']), yhat[:10])
print(cl('Support vectors : ', attrs = ['bold']), support_vectors)

# Visualizing the model

ax = plt.gca()
sb.scatterplot(X[:, 0], X[:, 1], hue = y, s = 200, palette = 'spring', legend = False, edgecolor = 'white')
plt.title('NON-LINEARLY SEPERABLE SVM MODEL', fontsize = 18)
plt.xticks(fontsize = 12)
plt.yticks(fontsize = 12)
xlim = ax.get_xlim()
ylim = ax.get_ylim()

xx = np.linspace(xlim[0], xlim[1], 30)
yy = np.linspace(ylim[0], ylim[1], 30)
YY, XX = np.meshgrid(yy, xx)
xy = np.vstack([XX.ravel(), YY.ravel()]).T
Z = model.decision_function(xy).reshape(XX.shape)

ax.contour(XX, YY, Z, colors = 'k', levels = [-1, 0, 1], alpha = 0.5,
           linestyles = ['--', '-', '--'])

ax.scatter(model.support_vectors_[:, 0], model.support_vectors_[:, 1], s = 200,
           linewidth = 1, facecolors = 'none', edgecolors = 'b')

plt.savefig('svm_non_linear.png')
plt.show()

# 3d visualization

r = np.exp(-(X ** 2).sum(1))

ax = plt.subplot(projection = '3d')
ax.scatter3D(X[:, 0], X[:, 1], r, c = y, s = 200, cmap = 'spring', edgecolor = 'b', alpha = 0.5)
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('r')

plt.savefig('non_linear_3dplot.png')
plt.show()