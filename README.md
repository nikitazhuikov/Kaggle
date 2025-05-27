# Kaggle Projects

Этот репозиторий содержит проекты с Kaggle, включая анализ данных и машинное обучение.

## Структура проекта

- `data/` - директория для хранения наборов данных
- `notebooks/` - Jupyter notebooks для анализа и визуализации
- `src/` - исходный код для обработки данных и моделей
- `tests/` - модульные тесты

## Установка

1. Создайте виртуальное окружение:
```bash
python -m venv venv
```

2. Активируйте виртуальное окружение:
```bash
# Windows
.\venv\Scripts\activate
# Linux/Mac
source venv/bin/activate
```

3. Установите зависимости:
```bash
pip install -r requirements.txt
```

## Проекты

### 1. Анализ цен на жилье
- Набор данных: Housing.csv
- Notebook: EDA_house_pricing.ipynb
- Основной анализ: Kaggle_housing.ipynb

### 2. PySpark анализ
- Notebook: PySpark (1).ipynb

## Использование

1. Запустите Jupyter Notebook:
```bash
jupyter notebook
```

2. Откройте нужный notebook из директории `notebooks/`

## Тестирование

Для запуска тестов используйте:
```bash
pytest tests/
```