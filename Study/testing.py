from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier

# Загружаем данные
iris = load_iris()
X, y = iris.data, iris.target

# Обучаем дерево
clf = DecisionTreeClassifier(max_depth=3, criterion='gini')
clf.fit(X, y)

import matplotlib.pyplot as plt
from sklearn.tree import plot_tree

plt.figure(figsize=(12, 8))
plot_tree(clf, 
          feature_names=iris.feature_names,  
          class_names=iris.target_names,
          filled=True)
plt.title("Решающее дерево (Gini)")
plt.show()