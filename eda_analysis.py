
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Загрузка данных
df = pd.read_excel("data/Данные_для_курсовои_Классическое_МО.xlsx")

# Удаление лишнего столбца
df = df.drop(columns=["Unnamed: 0"])

# Проверка пропущенных значений
missing = df.isnull().sum()
print("Пропущенные значения:\n", missing[missing > 0])

# Описательная статистика
print("\nСтатистики целевых переменных:")
print(df[["IC50, mM", "CC50, mM", "SI"]].describe())

# Визуализация распределений
plt.figure(figsize=(15, 4))
for i, column in enumerate(["IC50, mM", "CC50, mM", "SI"]):
    plt.subplot(1, 3, i + 1)
    sns.histplot(df[column], bins=30, kde=True)
    plt.title(f"Распределение {column}")
plt.tight_layout()
plt.show()

# Расчет медиан
median_ic50 = df["IC50, mM"].median()
median_cc50 = df["CC50, mM"].median()
median_si = df["SI"].median()

print(f"Медиана IC50: {median_ic50:.2f}")
print(f"Медиана CC50: {median_cc50:.2f}")
print(f"Медиана SI: {median_si:.2f}")

# Создание бинарных признаков
df["IC50 > median"] = df["IC50, mM"] > median_ic50
df["CC50 > median"] = df["CC50, mM"] > median_cc50
df["SI > median"] = df["SI"] > median_si
df["SI > 8"] = df["SI"] > 8

# Корреляционный анализ
correlation_matrix = df.corr(numeric_only=True)[["IC50, mM", "CC50, mM", "SI"]].sort_values(by="SI", ascending=False)
print("\nКорреляции с целевыми переменными:")
print(correlation_matrix.head(10))
