import pandas as pd

# Read the CSV file
df = pd.read_csv('Anketa.csv')

# Create a dictionary for column renaming
column_mapping = {
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
    'Возраст': 'Age'
}

# Rename columns
df = df.rename(columns=column_mapping)

# Remove specified columns
columns_to_remove = ['Id', 'Удовлетворенность_доступностью_лекарств']
df = df.drop(columns=columns_to_remove, errors='ignore')

# Save the result to a new file
df.to_csv('Anketa_renamed.csv', index=False)

print("Column renaming and removal completed successfully!") 