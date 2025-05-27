import pandas as pd
import numpy as np
from pathlib import Path

def load_data(file_path: str) -> pd.DataFrame:
    """
    Загрузка данных из CSV файла
    
    Args:
        file_path (str): Путь к файлу данных
        
    Returns:
        pd.DataFrame: Загруженные данные
    """
    return pd.read_csv(file_path)

def preprocess_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Базовая предобработка данных
    
    Args:
        df (pd.DataFrame): Исходные данные
        
    Returns:
        pd.DataFrame: Обработанные данные
    """
    # Удаление дубликатов
    df = df.drop_duplicates()
    
    # Обработка пропущенных значений
    df = df.fillna(df.mean())
    
    return df

def save_processed_data(df: pd.DataFrame, output_path: str):
    """
    Сохранение обработанных данных
    
    Args:
        df (pd.DataFrame): Данные для сохранения
        output_path (str): Путь для сохранения
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False) 