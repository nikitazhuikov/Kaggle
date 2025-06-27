# Решения для избежания FutureWarning: Downcasting behavior in `replace` is deprecated

import pandas as pd
import numpy as np

# Исходные данные
data = {
    'problem_solved': ['Да', 'Нет', 'Затрудняюсь ответить', 'Да', 'Нет']
}
df = pd.DataFrame(data)

print("Исходные данные:")
print(df['problem_solved'].value_counts())
print()

# ПРОБЛЕМНЫЙ КОД (вызывает FutureWarning):
# df['problem_not_solved'] = df['problem_solved'].replace(['Да', 'Нет', 'Затрудняюсь ответить'], [0, 1, 1])

# РЕШЕНИЕ 1: Явное указание типа данных
print("Решение 1: Явное указание типа данных")
df['problem_not_solved_1'] = df['problem_solved'].replace(['Да', 'Нет', 'Затрудняюсь ответить'], [0, 1, 1]).astype(int)
print(df[['problem_solved', 'problem_not_solved_1']])
print()

# РЕШЕНИЕ 2: Использование map() вместо replace()
print("Решение 2: Использование map()")
mapping = {'Да': 0, 'Нет': 1, 'Затрудняюсь ответить': 1}
df['problem_not_solved_2'] = df['problem_solved'].map(mapping)
print(df[['problem_solved', 'problem_not_solved_2']])
print()

# РЕШЕНИЕ 3: Использование numpy.where()
print("Решение 3: Использование numpy.where()")
df['problem_not_solved_3'] = np.where(
    df['problem_solved'] == 'Да', 0,
    np.where(df['problem_solved'].isin(['Нет', 'Затрудняюсь ответить']), 1, np.nan)
)
print(df[['problem_solved', 'problem_not_solved_3']])
print()

# РЕШЕНИЕ 4: Использование apply() с функцией
print("Решение 4: Использование apply() с функцией")
def convert_problem_solved(value):
    if value == 'Да':
        return 0
    elif value in ['Нет', 'Затрудняюсь ответить']:
        return 1
    else:
        return np.nan

df['problem_not_solved_4'] = df['problem_solved'].apply(convert_problem_solved)
print(df[['problem_solved', 'problem_not_solved_4']])
print()

# РЕШЕНИЕ 5: Использование pd.Categorical (для категориальных данных)
print("Решение 5: Использование pd.Categorical")
df['problem_not_solved_5'] = pd.Categorical(
    df['problem_solved'].replace(['Да', 'Нет', 'Затрудняюсь ответить'], [0, 1, 1]),
    categories=[0, 1],
    ordered=True
).codes
print(df[['problem_solved', 'problem_not_solved_5']])
print()

# РЕШЕНИЕ 6: Использование replace с параметром downcast=None
print("Решение 6: Использование replace с downcast=None")
df['problem_not_solved_6'] = df['problem_solved'].replace(
    ['Да', 'Нет', 'Затрудняюсь ответить'], 
    [0, 1, 1], 
    downcast=None
).astype(int)
print(df[['problem_solved', 'problem_not_solved_6']])
print()

# Проверка типов данных
print("Типы данных всех решений:")
for i in range(1, 7):
    col_name = f'problem_not_solved_{i}'
    print(f"{col_name}: {df[col_name].dtype}")

print("\nВсе решения дают одинаковый результат:")
print(df[['problem_solved'] + [f'problem_not_solved_{i}' for i in range(1, 7)]]) 