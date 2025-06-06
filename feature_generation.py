import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, PolynomialFeatures
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import chi2_contingency

def analyze_difficult_answers(df):
    # Список колонок для анализа
    columns = [
        'time_of_register', 'wait_time', 'near_cab', 'comfort',
        'attitude', 'explain', 'expect', 'loyalty', 'gen_sat',
        'diag_services_available', 'disabled_facilitites_available',
        'region_medical_care_availabilit', 'problem_solved'
    ]
    
    # Создаем матрицу для хранения результатов
    association_matrix = pd.DataFrame(index=columns, columns=columns)
    cramer_matrix = pd.DataFrame(index=columns, columns=columns)
    
    # Функция для вычисления коэффициента Крамера
    def cramers_v(confusion_matrix):
        chi2 = chi2_contingency(confusion_matrix)[0]
        n = confusion_matrix.sum().sum()
        min_dim = min(confusion_matrix.shape) - 1
        return np.sqrt(chi2 / (n * min_dim))
    
    for col1 in columns:
        for col2 in columns:
            # Создаем таблицу сопряженности для "Затрудняюсь ответить"
            mask1 = df[col1] == 'Затрудняюсь ответить'
            mask2 = df[col2] == 'Затрудняюсь ответить'
            
            # Создаем таблицу сопряженности
            contingency = pd.crosstab(mask1, mask2)
            
            # Вычисляем коэффициент Крамера
            cramer_v = cramers_v(contingency)
            cramer_matrix.loc[col1, col2] = cramer_v
            
            # Вычисляем процент совпадений
            total = len(df)
            both_difficult = ((mask1) & (mask2)).sum()
            association_matrix.loc[col1, col2] = (both_difficult / total) * 100
    
    # Создаем тепловую карту для коэффициента Крамера
    plt.figure(figsize=(12, 10))
    sns.heatmap(cramer_matrix.astype(float), 
                annot=True, 
                cmap='YlOrRd', 
                fmt='.2f',
                square=True)
    plt.title('Коэффициент Крамера между ответами "Затрудняюсь ответить"')
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig('difficult_answers_cramer.png')
    plt.close()
    
    # Создаем тепловую карту для процента совпадений
    plt.figure(figsize=(12, 10))
    sns.heatmap(association_matrix.astype(float), 
                annot=True, 
                cmap='YlOrRd', 
                fmt='.1f',
                square=True)
    plt.title('Процент совпадений ответов "Затрудняюсь ответить" (%)')
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig('difficult_answers_association.png')
    plt.close()
    
    # Выводим статистику по каждому столбцу
    print("\nСтатистика по ответам 'Затрудняюсь ответить':")
    for col in columns:
        difficult_count = (df[col] == 'Затрудняюсь ответить').sum()
        total_count = len(df)
        percentage = (difficult_count / total_count) * 100
        print(f"{col}: {difficult_count} ({percentage:.1f}%)")
    
    return cramer_matrix, association_matrix

def generate_features(df):
    # 1. Временные признаки
    df['survey_date'] = pd.to_datetime(df['date_of_survey'])
    df['survey_month'] = df['survey_date'].dt.month
    df['survey_quarter'] = df['survey_date'].dt.quarter
    df['survey_day_of_week'] = df['survey_date'].dt.dayofweek
    
    # 2. Признаки на основе возраста
    df['age_group'] = pd.cut(df['age'], 
                            bins=[0, 18, 30, 45, 60, 100],
                            labels=['0-18', '19-30', '31-45', '46-60', '60+'])
    
    # 3. Признаки удовлетворенности
    satisfaction_columns = ['time_of_register', 'wait_time', 'near_cab', 'comfort',
                          'attitude', 'explain', 'expect']
    
    df['satisfaction_std'] = df[satisfaction_columns].std(axis=1)
    df['max_ratings_count'] = (df[satisfaction_columns] == 5).sum(axis=1)
    df['min_ratings_count'] = (df[satisfaction_columns] == 1).sum(axis=1)
    
    # 4. Признаки на основе частоты посещений
    visit_mapping = {
        'менее 1 мес. назад': 0.5,
        'от 1 до 3 мес. назад': 2,
        'от 3 до 4 мес. назад': 3.5,
        'от 4 до 6 мес. назад': 5,
        'от 6 до 12 мес. назад': 9,
        'более 12 мес. назад': 18
    }
    df['last_visit_months'] = df['last_visit'].map(visit_mapping)
    df['visit_frequency'] = 1 / (df['last_visit_months'] + 0.1)
    
    # 5. Бинарные признаки
    df['is_urban'] = df['poselenie'].map({'Город': 1, 'Село': 0})
    df['is_male'] = (df['gender'] == 'Мужской').astype(int)
    df['org_type'] = df['org_type'].fillna('Преимущественно в государственных')
    df['is_state_org'] = df['org_type'].str.contains('государственных').astype(int)
    
    # 6. Признаки доступности услуг
    df['has_home_visit_problem'] = (df['doctor_domestic_visit_problem'] == 'Да').astype(int)
    df['diag_available'] = (df['diag_services_available'] == 'Да').astype(int)
    df['disabled_facilities'] = (df['disabled_facilitites_available'] == 'Да').astype(int)
    
    # 7. Композитные признаки
    df['accessibility_index'] = (df['has_home_visit_problem'] + 
                               df['diag_available'] + 
                               df['disabled_facilities']) / 3
    
    df['service_quality_index'] = (df['time_of_register'] + 
                                 df['wait_time'] + 
                                 df['near_cab'] + 
                                 df['comfort']) / 4
    
    df['staff_interaction_index'] = (df['attitude'] + 
                                   df['explain'] + 
                                   df['expect']) / 3
    
    # 8. Признаки на основе региона
    region_counts = df['Регион'].value_counts()
    df['region_respondent_count'] = df['Регион'].map(region_counts)
    
    region_satisfaction = df.groupby('Регион')['avg_satisfaction'].mean()
    df['region_avg_satisfaction'] = df['Регион'].map(region_satisfaction)
    
    region_age = df.groupby('Регион')['age'].mean()
    df['region_avg_age'] = df['Регион'].map(region_age)
    
    # 9. Кодирование региона
    le = LabelEncoder()
    df['region_label'] = le.fit_transform(df['Регион'])
    
    region_dummies = pd.get_dummies(df['Регион'], prefix='region')
    top_n_regions = 20
    top_regions = df['Регион'].value_counts().nlargest(top_n_regions).index
    region_dummies = region_dummies[['region_' + reg for reg in top_regions]]
    df = pd.concat([df, region_dummies], axis=1)
    
    # 10. Региональные кластеры
    region_groups = {
        'Центральный': ['Москва', 'Московская область', 'Тверская область', 'Ярославская область'],
        'Северо-Западный': ['Санкт-Петербург', 'Ленинградская область', 'Мурманская область'],
        'Южный': ['Краснодарский край', 'Ростовская область', 'Волгоградская область'],
        'Приволжский': ['Нижегородская область', 'Самарская область', 'Татарстан'],
        'Уральский': ['Свердловская область', 'Челябинская область', 'Пермский край'],
        'Сибирский': ['Новосибирская область', 'Красноярский край', 'Иркутская область'],
        'Дальневосточный': ['Приморский край', 'Хабаровский край', 'Сахалинская область']
    }
    
    df['federal_district'] = 'Другое'
    for district, regions in region_groups.items():
        df.loc[df['Регион'].isin(regions), 'federal_district'] = district
    
    district_dummies = pd.get_dummies(df['federal_district'], prefix='district')
    df = pd.concat([df, district_dummies], axis=1)
    
    # 11. Региональные метрики
    region_urban = df.groupby('Регион')['is_urban'].mean()
    df['region_urban_ratio'] = df['Регион'].map(region_urban)
    
    region_male = df.groupby('Регион')['is_male'].mean()
    df['region_male_ratio'] = df['Регион'].map(region_male)
    
    # 12. Полиномиальные признаки
    numeric_features = [
        'age',
        'avg_satisfaction',
        'satisfaction_std',
        'last_visit_months',
        'visit_frequency',
        'service_quality_index',
        'staff_interaction_index',
        'accessibility_index',
        'region_avg_satisfaction',
        'region_avg_age'
    ]
    
    # Заполняем пропуски в числовых признаках
    for feature in numeric_features:
        if feature in df.columns:
            # Для возраста используем медиану
            if feature == 'age':
                df[feature] = df[feature].fillna(df[feature].median())
            # Для остальных признаков используем среднее
            else:
                df[feature] = df[feature].fillna(df[feature].mean())
    
    # Создаем полиномиальные признаки
    poly = PolynomialFeatures(degree=2, include_bias=False)
    poly_features = poly.fit_transform(df[numeric_features])
    
    feature_names = poly.get_feature_names_out(numeric_features)
    poly_df = pd.DataFrame(poly_features, columns=feature_names)
    poly_df = poly_df.drop(columns=numeric_features)
    df = pd.concat([df, poly_df], axis=1)
    
    # 13. Взаимодействия между важными признаками
    df['satisfaction_age_interaction'] = df['avg_satisfaction'] * df['age']
    df['satisfaction_urban_interaction'] = df['avg_satisfaction'] * df['is_urban']
    df['satisfaction_gender_interaction'] = df['avg_satisfaction'] * df['is_male']
    df['service_access_interaction'] = df['service_quality_index'] * df['accessibility_index']
    df['visit_satisfaction_interaction'] = df['visit_frequency'] * df['avg_satisfaction']
    
    # 14. Нормализованные полиномиальные признаки
    for col in df.columns:
        if '^2' in col or 'interaction' in col:
            df[f'{col}_normalized'] = (df[col] - df[col].mean()) / df[col].std()
    
    # 15. Инвертированные бинарные признаки
    df['problem_not_solved'] = df['problem_solved'].replace([0, 1], [1, 0]).fillna(-1).astype(int)
    
    return df

# Пример использования:
# df = pd.read_csv('data.csv')
# df = generate_features(df) 