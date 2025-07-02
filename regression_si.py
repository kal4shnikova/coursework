
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.impute import SimpleImputer

# Загрузка данных
df = pd.read_excel("data/Данные_для_курсовои_Классическое_МО.xlsx")
df = df.drop(columns=["Unnamed: 0"])

# Целевая переменная
y = df["SI"]
X = df.drop(columns=["IC50, mM", "CC50, mM", "SI",
                     "IC50 > median", "CC50 > median", "SI > median", "SI > 8"], errors='ignore')

# Удаление признаков с большим числом пропусков
threshold = 0.3
missing_ratio = X.isnull().mean()
X = X.loc[:, missing_ratio < threshold]

# Заполнение NaN медианой
imputer = SimpleImputer(strategy="median")
X = pd.DataFrame(imputer.fit_transform(X), columns=X.columns)

# Разделение на train/test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Обучение модели
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Предсказание и метрики
y_pred = model.predict(X_test)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"RMSE: {rmse:.2f}")
print(f"MAE: {mae:.2f}")
print(f"R²: {r2:.3f}")
