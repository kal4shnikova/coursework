
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, confusion_matrix
from sklearn.impute import SimpleImputer

# Загрузка данных
df = pd.read_excel("data/Данные_для_курсовои_Классическое_МО.xlsx")
df = df.drop(columns=["Unnamed: 0"])

# Целевая переменная
y = df["CC50, mM"] > df["CC50, mM"].median()
X = df.drop(columns=["IC50, mM", "CC50, mM", "SI",
                     "IC50 > median", "CC50 > median", "SI > median", "SI > 8"], errors='ignore')

# Удаление признаков с большим числом пропусков
threshold = 0.3
missing_ratio = X.isnull().mean()
X = X.loc[:, missing_ratio < threshold]

# Заполнение NaN медианой
imputer = SimpleImputer(strategy="median")
X = pd.DataFrame(imputer.fit_transform(X), columns=X.columns)

# Разделение
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Модель
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Предсказание
y_pred = model.predict(X_test)
y_proba = model.predict_proba(X_test)[:, 1]

# Метрики
acc = accuracy_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
auc = roc_auc_score(y_test, y_proba)
conf_mat = confusion_matrix(y_test, y_pred)

print(f"Accuracy: {acc:.2f}")
print(f"F1-score: {f1:.2f}")
print(f"ROC-AUC: {auc:.2f}")
print("Confusion matrix:")
print(conf_mat)
