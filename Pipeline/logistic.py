import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

# Загрузка данных
train_data = pd.read_csv('C:/Users/sahav/Desktop/Data Science/Pipeline/train.csv')
# Создание нового признака AgeBand
def age_band(age):
    if pd.isnull(age):
        return 'Unknown'
    elif age <= 12:
        return '0'
    elif age <= 19:
        return '1'
    elif age <= 35:
        return '2'
    elif age <= 50:
        return '3'
    elif age <= 65:
        return '4'
    else:
        return '5'

train_data['AgeBand'] = train_data['Age'].apply(age_band)

# Замена пропусков для числовых и категориальных признаков
numerical_features = ['RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck']
categorical_features = ['HomePlanet', 'CryoSleep', 'Destination', 'VIP', 'Cabin', 'AgeBand']

# Преобразуем все категориальные признаки в строки
train_data[categorical_features] = train_data[categorical_features].astype(str)

# Создаем процессоры для обработки данных
numeric_transformer = Pipeline([
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

categorical_transformer = Pipeline([
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('encoder', OneHotEncoder(handle_unknown='ignore'))
])

# Комбинируем процессоры в ColumnTransformer
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numerical_features),
        ('cat', categorical_transformer, categorical_features)
    ])

# Разделение данных на признаки и целевую переменную
y = train_data['Transported']
X = train_data.drop(columns=['Transported', 'Name', 'PassengerId'])

# Построение модели с обработкой данных
pipeline = Pipeline([
    ('preprocessor', preprocessor),
    ('classifier', LogisticRegression(max_iter=1000))
])

# Разделение на обучающую и тестовую выборки
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Обучение модели
pipeline.fit(X_train, y_train)

# Предсказания и оценка качества
predictions = pipeline.predict(X_test)
accuracy = accuracy_score(y_test, predictions)

# Вывод результатов в файл
with open('logistic_regression_results.txt', 'w') as f:
    f.write(f'Accuracy: {accuracy:.4f}\n')
    f.write(classification_report(y_test, predictions))

# Матрица ошибок
conf_matrix = confusion_matrix(y_test, predictions)
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues')
plt.title('Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.savefig('logistic_regression_confusion_matrix.png')
plt.show()

# График точности
plt.figure(figsize=(6, 4))
plt.bar(['Accuracy'], [accuracy], color='green')
plt.ylim(0, 1)
plt.title('Model Accuracy')
plt.ylabel('Accuracy')
plt.savefig('Logistic_regression_accuracy.png')
plt.show()
