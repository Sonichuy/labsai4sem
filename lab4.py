import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import f1_score, recall_score, precision_score
from sklearn.model_selection import cross_val_score
import matplotlib.pyplot as plt


data = pd.read_csv(r'C:\Users\sonak\OneDrive\Рабочий стол\Универ 2 курс\4 семестр\ИИиММО (экзамен)\processed_titanic.csv')
X = data.drop('Transported', axis=1)
y = data['Transported']


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=50)


rf = RandomForestClassifier(random_state=50)
gb = GradientBoostingClassifier(random_state=50)

rf.fit(X_train, y_train)
gb.fit(X_train, y_train)

rf_pred = rf.predict(X_test)
gb_pred = gb.predict(X_test)

# Расчет метрик
metrics = {
    'Random Forest': {
        'F1': f1_score(y_test, rf_pred), #гармоническое среднее между точностью (precision) и полнотой (recall)
        'Recall': recall_score(y_test, rf_pred), #какую долю всех реальных положительных случаев модель смогла правильно обнаружить
        'Precision': precision_score(y_test, rf_pred) #какую долю предсказанных положительных случаев действительно являются положительными. Формула
    },
    'Gradient Boosting': {
        'F1': f1_score(y_test, gb_pred),
        'Recall': recall_score(y_test, gb_pred),
        'Precision': precision_score(y_test, gb_pred)
    }
}


rf_cv = cross_val_score(rf, X, y, cv=5, scoring='f1').mean()
gb_cv = cross_val_score(gb, X, y, cv=5, scoring='f1').mean()

fig, ax = plt.subplots(figsize=(10, 6))
metrics_df = pd.DataFrame(metrics).T
metrics_df.plot(kind='bar', ax=ax)
plt.title('Сравнение метрик классификации')
plt.ylabel('Значение метрики')
plt.xticks(rotation=0)
plt.show()

print(f"Случайный лес (F1 при кросс-валидации): {rf_cv:.4f}")
print(f"Градиентный бустинг (F1 при кросс-валидации): {gb_cv:.4f}\n")
print(pd.DataFrame(metrics))
