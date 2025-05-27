import pytest
import pandas as pd
import numpy as np
from src.data_processing import load_data, preprocess_data, save_processed_data

def test_preprocess_data():
    # Создаем тестовые данные
    df = pd.DataFrame({
        'A': [1, 2, 2, np.nan],
        'B': [4, 5, 5, 6]
    })
    
    # Обрабатываем данные
    processed_df = preprocess_data(df)
    
    # Проверяем результаты
    assert len(processed_df) == 3  # Дубликаты удалены
    assert processed_df['A'].isna().sum() == 0  # Нет пропущенных значений
    assert processed_df['B'].nunique() == 3  # Уникальные значения сохранены 