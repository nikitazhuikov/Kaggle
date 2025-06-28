import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier, ExtraTreesClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import mlflow
import mlflow.sklearn
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

# Попытка импорта дополнительных библиотек
try:
    from xgboost import XGBClassifier
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False
    print("XGBoost не установлен")

try:
    from catboost import CatBoostClassifier
    CATBOOST_AVAILABLE = True
except ImportError:
    CATBOOST_AVAILABLE = False
    print("CatBoost не установлен")

try:
    from lightgbm import LGBMClassifier
    LIGHTGBM_AVAILABLE = True
except ImportError:
    LIGHTGBM_AVAILABLE = False
    print("LightGBM не установлен")

# Настройка отображения
pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)
pd.set_option('display.float_format', lambda x: '%.3f' % x)

# Загрузка данных
print("Загрузка данных...")
df = pd.read_csv('Anketa.csv', on_bad_lines='warn', sep=';')

# Переименование колонок
column_mapping = {
    'Календарный_год': 'year', 'Регион': 'Region', 'Период': 'period',
    'poselenie': 'locality_type', 'Населенный_пункт': 'locality',
    'zapis': 'time_of_register', 'ozhid': 'wait_time',
    'Дата_проведения_опроса_в_формате_дд_мм_гггг': 'date_of_survey',
    'Номер_респондента': 'respondent_number', '16': 'region_medical_care_availability',
    '17': 'doctor_domestic_visit_problem', '18': 'diag_services_available',
    '19': 'disabled_facilitites_available', 'Возраст': 'age'
}

df = df.rename(columns=column_mapping)
columns_to_remove = ['Id', 'Удовлетворенность_доступностью_лекарств', 'health_selfestimation', 
                     'children_number', '1st_child_health', '2nd_child_health', '3d_child_health', 
                     '4th_child_health', '5th_child_health']
df = df.drop(columns=columns_to_remove, errors='ignore')

# Обработка данных
print("Обработка данных...")
columns = ['time_of_register', 'wait_time', 'near_cab', 'comfort', 'attitude', 'explain', 
           'expect', 'loyalty', 'gen_sat', 'diag_services_available', 
           'disabled_facilitites_available', 'region_medical_care_availability', 'problem_solved']

# Удаление строк с "Затрудняюсь ответить"
masks = {col: df[col] == 'Затрудняюсь ответить' for col in columns}
mask_df = pd.DataFrame(masks)
mask_df['count'] = mask_df.sum(axis=1)
rows_to_drop = mask_df[mask_df['count'].isin([10, 11, 12, 13])].index
df = df.drop(rows_to_drop)

# Подготовка признаков
print("Подготовка признаков...")
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

# Параметры модели для определения важности
params = {
    'C': 100, 'max_iter': 1000, 'penalty': 'l2', 'solver': 'lbfgs', 'random_state': 42
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

# Параметры для различных моделей
model_params = {
    'Logistic Regression': {'C': 100, 'max_iter': 1000, 'penalty': 'l2', 'solver': 'lbfgs', 'random_state': 42},
    'Random Forest': {'n_estimators': 100, 'max_depth': 10, 'min_samples_split': 5, 'min_samples_leaf': 2, 'random_state': 42, 'n_jobs': -1},
    'AdaBoost': {'n_estimators': 100, 'learning_rate': 0.1, 'random_state': 42},
    'Gradient Boosting': {'n_estimators': 100, 'max_depth': 6, 'learning_rate': 0.1, 'random_state': 42},
    'Extra Trees': {'n_estimators': 100, 'max_depth': 10, 'min_samples_split': 5, 'min_samples_leaf': 2, 'random_state': 42, 'n_jobs': -1}
}

# Создание словаря моделей
models = {
    'KNN': KNeighborsClassifier(n_neighbors=5),
    'Logistic Regression': LogisticRegression(**model_params['Logistic Regression']),
    'Random Forest': RandomForestClassifier(**model_params['Random Forest']),
    'AdaBoost': AdaBoostClassifier(**model_params['AdaBoost']),
    'Gradient Boosting': GradientBoostingClassifier(**model_params['Gradient Boosting']),
    'Extra Trees': ExtraTreesClassifier(**model_params['Extra Trees'])
}

# Добавление моделей, если библиотеки доступны
if XGBOOST_AVAILABLE:
    models['XGBoost'] = XGBClassifier(n_estimators=100, max_depth=6, learning_rate=0.1, random_state=42, n_jobs=-1, verbosity=0)

if CATBOOST_AVAILABLE:
    models['CatBoost'] = CatBoostClassifier(iterations=100, depth=6, learning_rate=0.1, random_state=42, verbose=False)

if LIGHTGBM_AVAILABLE:
    models['LightGBM'] = LGBMClassifier(n_estimators=100, max_depth=6, learning_rate=0.1, random_state=42, n_jobs=-1, verbose=-1)

# Настройка MLflow
mlflow.set_experiment("Medical_Survey_Multiple_Models")

# Результаты для сводной таблицы
results = []

print("\nОбучение и оценка моделей...")
for model_name, model in tqdm(models.items(), desc="Обучение моделей"):
    print(f"\nОбучение {model_name}...")
    
    with mlflow.start_run(run_name=f"{model_name.lower().replace(' ', '_')}"):
        try:
            # Логирование параметров
            if model_name in model_params:
                mlflow.log_params(model_params[model_name])
            mlflow.log_param("n_features", 36)
            mlflow.log_param("model_name", model_name)
            
            # Обучение модели
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
            mlflow.sklearn.log_model(model, f"{model_name.lower().replace(' ', '_')}", 
                                   input_example=X_test_top36.head(1))
            
            # Сохранение результатов
            results.append({
                'Model': model_name,
                'Accuracy': metrics['accuracy'],
                'Precision': metrics['precision'],
                'Recall': metrics['recall'],
                'F1-Score': metrics['f1'],
                'ROC-AUC': metrics['roc_auc']
            })
            
            print(f"  Accuracy: {metrics['accuracy']:.3f}")
            print(f"  Precision: {metrics['precision']:.3f}")
            print(f"  Recall: {metrics['recall']:.3f}")
            print(f"  F1-score: {metrics['f1']:.3f}")
            print(f"  ROC-AUC: {metrics['roc_auc']:.3f}")
            
        except Exception as e:
            print(f"Ошибка при обучении {model_name}: {str(e)}")
            results.append({
                'Model': model_name,
                'Accuracy': np.nan,
                'Precision': np.nan,
                'Recall': np.nan,
                'F1-Score': np.nan,
                'ROC-AUC': np.nan
            })

# Создание сводной таблицы
print("\n" + "="*80)
print("СВОДНАЯ ТАБЛИЦА РЕЗУЛЬТАТОВ МОДЕЛЕЙ")
print("="*80)

results_df = pd.DataFrame(results)
results_df = results_df.sort_values('ROC-AUC', ascending=False)
print(results_df.round(3).to_string(index=False))

# Сохранение результатов
results_df.to_csv('model_comparison_results.csv', index=False)
print(f"\nРезультаты сохранены в 'model_comparison_results.csv'")

# Создание графиков
print("\nСоздание графиков сравнения...")

# График сравнения метрик
fig, axes = plt.subplots(2, 2, figsize=(15, 12))

# ROC-AUC
axes[0, 0].barh(results_df['Model'], results_df['ROC-AUC'])
axes[0, 0].set_title('ROC-AUC Comparison')
axes[0, 0].set_xlabel('ROC-AUC Score')

# F1-Score
axes[0, 1].barh(results_df['Model'], results_df['F1-Score'])
axes[0, 1].set_title('F1-Score Comparison')
axes[0, 1].set_xlabel('F1-Score')

# Accuracy
axes[1, 0].barh(results_df['Model'], results_df['Accuracy'])
axes[1, 0].set_title('Accuracy Comparison')
axes[1, 0].set_xlabel('Accuracy')

# Precision vs Recall
axes[1, 1].scatter(results_df['Precision'], results_df['Recall'], s=100)
for i, model in enumerate(results_df['Model']):
    axes[1, 1].annotate(model, (results_df['Precision'].iloc[i], results_df['Recall'].iloc[i]))
axes[1, 1].set_xlabel('Precision')
axes[1, 1].set_ylabel('Recall')
axes[1, 1].set_title('Precision vs Recall')
axes[1, 1].grid(True)

plt.tight_layout()
plt.savefig('model_comparison_plots.png', dpi=300, bbox_inches='tight')
plt.show()

# Тепловая карта метрик
plt.figure(figsize=(10, 6))
metrics_matrix = results_df.set_index('Model')[['Accuracy', 'Precision', 'Recall', 'F1-Score', 'ROC-AUC']]
sns.heatmap(metrics_matrix, annot=True, cmap='YlOrRd', fmt='.3f', cbar_kws={'label': 'Score'})
plt.title('Model Performance Heatmap')
plt.tight_layout()
plt.savefig('model_performance_heatmap.png', dpi=300, bbox_inches='tight')
plt.show()

# Логирование результатов в MLflow
with mlflow.start_run(run_name="model_comparison_summary"):
    mlflow.log_artifact('model_comparison_results.csv', "results")
    mlflow.log_artifact('model_comparison_plots.png', "plots")
    mlflow.log_artifact('model_performance_heatmap.png', "plots")
    
    # Логирование лучшей модели
    best_model = results_df.iloc[0]
    mlflow.log_metric("best_roc_auc", best_model['ROC-AUC'])
    mlflow.log_metric("best_f1_score", best_model['F1-Score'])
    mlflow.log_param("best_model", best_model['Model'])

print(f"\nАнализ завершен!")
print(f"Лучшая модель по ROC-AUC: {best_model['Model']} ({best_model['ROC-AUC']:.3f})")
print(f"Лучшая модель по F1-Score: {results_df.loc[results_df['F1-Score'].idxmax(), 'Model']} ({results_df['F1-Score'].max():.3f})") 