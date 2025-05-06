import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import precision_score, confusion_matrix

df = pd.read_csv(r'C:\Users\sonak\OneDrive\Рабочий стол\Универ 2 курс\4 семестр\ИИиММО (экзамен)\processed_titanic.csv')


X = df.drop(columns=['PassengerId', 'Transported'])
y = df['Transported']


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=50)


clf = DecisionTreeClassifier(criterion='gini', max_depth=4, random_state=50)  # Ограничиваем максимальную глубину дерева
clf.fit(X_train, y_train)


y_pred = clf.predict(X_test)


precision = precision_score(y_test, y_pred)
print(f'Precision: {precision:.2f}')


cm = confusion_matrix(y_test, y_pred)
print('Confusion Matrix:')
print(cm)

# Дерево
plt.figure(figsize=(16, 14))
plot_tree(clf, filled=True,
          feature_names=X.columns,
          class_names=['Not Transported', 'Transported'],
          rounded=True,
          fontsize=8,
          max_depth=4,  # глубина
          proportion=False,
          node_ids=True,
          label='all')
plt.title('Decision Tree Visualization', fontsize=16)
plt.tight_layout()
plt.show()

#  матрицы ошибок
plt.figure(figsize=(7, 5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Not Transported', 'Transported'], yticklabels=['Not Transported', 'Transported'])
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.show()
