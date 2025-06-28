import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    roc_curve, auc, precision_recall_curve, classification_report, 
    roc_auc_score, confusion_matrix, ConfusionMatrixDisplay, 
    accuracy_score, precision_score, recall_score, f1_score
)
import mlflow
import mlflow.sklearn
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

# Настройка отображения
pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)
pd.set_option('display.max_colwidth', None)
pd.set_option('display.float_format', lambda x: '%.3f' % x)

# Загрузка данных
print("Загрузка данных...")
df = pd.read_csv('Anketa.csv', on_bad_lines='warn', sep=';')

# Переименование колонок
column_mapping = {
    'Календарный_год': 'year',
    'Регион': 'Region',
    'Период': 'period',
    'poselenie': 'locality_type',
    'Населенный_пункт': 'locality',
    'zapis': 'time_of_register',
    'ozhid': 'wait_time',
    'Дата_проведения_опроса_в_формате_дд_мм_гггг': 'date_of_survey',
    'Номер_респондента': 'respondent_number',
    '16': 'region_medical_care_availability',
    '17': 'doctor_domestic_visit_problem',
    '18': 'diag_services_available',
    '19': 'disabled_facilitites_available',
    'Возраст': 'age'
}

df = df.rename(columns=column_mapping)
columns_to_remove = ['Id', 'Удовлетворенность_доступностью_лекарств', 'health_selfestimation', 
                     'children_number', '1st_child_health', '2nd_child_health', '3d_child_health', 
                     '4th_child_health', '5th_child_health']
df = df.drop(columns=columns_to_remove, errors='ignore')

# Обработка данных
print("Обработка данных...")
columns = [
    'time_of_register', 'wait_time', 'near_cab', 'comfort',
    'attitude', 'explain', 'expect', 'loyalty', 'gen_sat',
    'diag_services_available', 'disabled_facilitites_available',
    'region_medical_care_availability', 'problem_solved'
]

# Удаление строк с "Затрудняюсь ответить"
masks = {col: df[col] == 'Затрудняюсь ответить' for col in columns}
mask_df = pd.DataFrame(masks)
mask_df['count'] = mask_df.sum(axis=1)
rows_to_drop = mask_df[mask_df['count'].isin([10, 11, 12, 13])].index
df = df.drop(rows_to_drop)

# Подготовка признаков
print("Подготовка признаков...")
# Здесь нужно добавить код генерации признаков из feature_generation.py
# Для примера возьмем базовые признаки
categorical_columns = ['time_of_register', 'wait_time', 'near_cab', 'comfort', 'attitude', 
                      'explain', 'expect', 'loyalty', 'gen_sat', 'diag_services_available', 
                      'disabled_facilitites_available', 'region_medical_care_availability']

# Кодирование категориальных признаков
le = LabelEncoder()
for col in categorical_columns:
    if col in df.columns:
        df[col] = le.fit_transform(df[col].astype(str))

# Целевая переменная
y = (df['problem_solved'] == 'Да').astype(int)

# Выбор числовых признаков
numeric_columns = df.select_dtypes(include=[np.number]).columns.tolist()
numeric_columns = [col for col in numeric_columns if col not in ['problem_solved', 'respondent_number']]

X = df[numeric_columns].copy()

# Обработка пропущенных значений
imputer = SimpleImputer(strategy='mean')
X = pd.DataFrame(imputer.fit_transform(X), columns=X.columns)

# Добавление рандомной фичи для отбора признаков
np.random.seed(42)
X['random'] = np.random.randn(len(X))

# Разделение на обучающую и тестовую выборки
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Параметры модели
params = {
    'C': 100,
    'max_iter': 1000,
    'penalty': 'l2',
    'solver': 'lbfgs',
    'random_state': 42
}

# Обучение модели для определения важности признаков
print("Определение важности признаков...")
model_with_random = LogisticRegression(**params)
model_with_random.fit(X_train, y_train)

# Получение важности признаков
feature_importance_with_random = pd.DataFrame({
    'feature': X_train.columns,
    'importance': np.abs(model_with_random.coef_[0])
})
feature_importance_with_random = feature_importance_with_random.sort_values('importance', ascending=False)

# Удаление рандомной фичи
feature_importance_with_random = feature_importance_with_random[feature_importance_with_random['feature'] != 'random']

# Выбор топ-36 признаков
top_36_features = feature_importance_with_random.head(36)['feature'].tolist()
print(f"\nТоп-36 наиболее важных признаков:")
for i, feature in enumerate(top_36_features, 1):
    importance = feature_importance_with_random[feature_importance_with_random['feature'] == feature]['importance'].iloc[0]
    print(f"{i:2d}. {feature:<40} (важность: {importance:.6f})")

# Создание датасета с топ-36 признаками
X_top36 = X[top_36_features]
X_train_top36 = X_train[top_36_features]
X_test_top36 = X_test[top_36_features]

# Настройка MLflow
mlflow.set_experiment("Medical_Survey_Top36_Features")

# Обучение модели с топ-36 признаками
print("\nОбучение модели с топ-36 признаками...")
with mlflow.start_run(run_name="logistic_regression_top36"):
    # Логирование параметров
    mlflow.log_params(params)
    mlflow.log_param("n_features", 36)
    mlflow.log_param("selected_features", str(top_36_features))
    
    # Обучение модели
    model = LogisticRegression(**params)
    model.fit(X_train_top36, y_train)
    
    # Предсказания
    y_pred = model.predict(X_test_top36)
    y_pred_proba = model.predict_proba(X_test_top36)[:, 1]
    
    # Вычисление метрик
    metrics = {
        'accuracy': accuracy_score(y_test, y_pred),
        'precision': precision_score(y_test, y_pred),
        'recall': recall_score(y_test, y_pred),
        'f1': f1_score(y_test, y_pred),
        'roc_auc': roc_auc_score(y_test, y_pred_proba)
    }
    
    # Логирование метрик
    mlflow.log_metrics(metrics)
    
    # Логирование модели
    mlflow.sklearn.log_model(model, "logistic_regression_top36", 
                           input_example=X_test_top36.head(1))
    
    # Вывод результатов
    print("\nРезультаты модели с топ-36 признаками:")
    print(f"Accuracy: {metrics['accuracy']:.3f}")
    print(f"Precision: {metrics['precision']:.3f}")
    print(f"Recall: {metrics['recall']:.3f}")
    print(f"F1-score: {metrics['f1']:.3f}")
    print(f"ROC-AUC: {metrics['roc_auc']:.3f}")
    
    # Создание графиков
    
    # 1. ROC-кривая
    plt.figure(figsize=(15, 5))
    
    plt.subplot(1, 3, 1)
    fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
    roc_auc = auc(fpr, tpr)
    
    plt.plot(fpr, tpr, color='darkorange', lw=2, 
             label=f'ROC curve (AUC = {roc_auc:.3f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.legend(loc="lower right")
    plt.grid(True)
    
    # 2. Precision-Recall кривая
    plt.subplot(1, 3, 2)
    precision, recall, _ = precision_recall_curve(y_test, y_pred_proba)
    pr_auc = auc(recall, precision)
    
    plt.plot(recall, precision, color='blue', lw=2, 
             label=f'PR curve (AUC = {pr_auc:.3f})')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve')
    plt.legend()
    plt.grid(True)
    
    # 3. Гистограмма разделяющей способности
    plt.subplot(1, 3, 3)
    
    # Разделение предсказаний по классам
    y_pred_proba_class0 = y_pred_proba[y_test == 0]
    y_pred_proba_class1 = y_pred_proba[y_test == 1]
    
    plt.hist(y_pred_proba_class0, bins=50, alpha=0.7, label='Class 0 (Problem Not Solved)', 
             color='red', density=True)
    plt.hist(y_pred_proba_class1, bins=50, alpha=0.7, label='Class 1 (Problem Solved)', 
             color='green', density=True)
    plt.xlabel('Predicted Probability')
    plt.ylabel('Density')
    plt.title('Model Separability')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig('model_performance_plots.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Логирование графиков
    mlflow.log_artifact('model_performance_plots.png', "plots")
    
    # Матрица ошибок
    plt.figure(figsize=(8, 6))
    cm = confusion_matrix(y_test, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['Not Solved', 'Solved'])
    disp.plot(cmap='Blues', values_format='d')
    plt.title('Confusion Matrix')
    plt.savefig('confusion_matrix.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    mlflow.log_artifact('confusion_matrix.png', "plots")
    
    # Дополнительный анализ: важность признаков в топ-36
    feature_importance_top36 = pd.DataFrame({
        'feature': top_36_features,
        'importance': np.abs(model.coef_[0])
    }).sort_values('importance', ascending=False)
    
    plt.figure(figsize=(12, 8))
    plt.barh(range(len(feature_importance_top36)), feature_importance_top36['importance'])
    plt.yticks(range(len(feature_importance_top36)), feature_importance_top36['feature'])
    plt.xlabel('Feature Importance')
    plt.title('Top-36 Features Importance')
    plt.gca().invert_yaxis()
    plt.tight_layout()
    plt.savefig('feature_importance_top36.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    mlflow.log_artifact('feature_importance_top36.png', "plots")
    
    # Сохранение списка признаков
    feature_importance_top36.to_csv('top36_features_importance.csv', index=False)
    mlflow.log_artifact('top36_features_importance.csv', "data")
    
    print(f"\nМодель успешно обучена и залогирована в MLflow!")
    print(f"ROC-AUC: {roc_auc:.3f}")
    print(f"PR-AUC: {pr_auc:.3f}")
    print(f"Количество признаков: {len(top_36_features)}")

print("\nАнализ завершен! Все результаты залогированы в MLflow.") 