
# Курсовая работа: Моделирование эффективности лекарственных соединений

## 📌 Описание проекта
Цель: Построение моделей для оценки эффективности и токсичности химических соединений. Используются регрессионные и классификационные методы для предсказания IC50, CC50 и SI.

## 📁 Структура проекта

```
├── data/
│   └── Данные_для_курсовои_Классическое_МО.xlsx
├── eda/
│   └── eda_analysis.py
├── models/
│   ├── regression_ic50.py
│   ├── regression_cc50.py
│   ├── regression_si.py
│   ├── classification_ic50_median.py
│   ├── classification_cc50_median.py
│   ├── classification_si_median.py
│   └── classification_si_above8.py
├── report/
│   └── Курсовая_работа_отчет.docx
└── README.md
```

## 📊 Построенные модели

### Регрессия:
- IC50: RMSE = 440.56, R² = 0.418
- CC50: RMSE = 459.65, R² = 0.592
- SI: RMSE = 1353.90, R² = 0.087

### Классификация:
- IC50 > медианы: Accuracy = 0.75, F1 = 0.75, AUC = 0.78
- CC50 > медианы: Accuracy = 0.81, F1 = 0.81, AUC = 0.88
- SI > медианы: Accuracy = 0.68, F1 = 0.66, AUC = 0.72
- SI > 8: Accuracy = 0.70, F1 = 0.49, AUC = 0.72

## ✅ Используемые технологии
- Python, pandas, scikit-learn
- Random Forest
- Оценка моделей: RMSE, MAE, R², Accuracy, F1, ROC-AUC

## 📎 Отчет
Финальный отчет находится в папке `report/`.
