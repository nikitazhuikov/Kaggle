import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Read the data
df = pd.read_csv('data.csv')

# Filter data for 2023 and months 2-5
df_2023 = df[(df['Календарный_год'] == 2023) & (df['Период'].isin([2, 3, 4, 5]))]

# Get the first 10 regions
top_10_regions = df_2023['Регион'].value_counts().head(10).index

# Create a figure with 4 subplots
fig, axes = plt.subplots(2, 2, figsize=(20, 15))
fig.suptitle('Распределение респондентов по полу в топ-10 регионах (2023)', fontsize=16)

# Flatten axes for easier iteration
axes = axes.flatten()

# Create subplots for each month
for idx, month in enumerate([2, 3, 4, 5]):
    # Filter data for current month
    month_data = df_2023[df_2023['Период'] == month]
    
    # Create pivot table for the plot
    pivot_data = pd.pivot_table(
        month_data[month_data['Регион'].isin(top_10_regions)],
        values='Календарный_год',  # Using any column as values since we just need counts
        index='Регион',
        columns='gender',
        aggfunc='count'
    )
    
    # Plot
    pivot_data.plot(kind='bar', ax=axes[idx], stacked=True)
    axes[idx].set_title(f'Месяц {month}')
    axes[idx].set_xlabel('Регион')
    axes[idx].set_ylabel('Количество респондентов')
    axes[idx].tick_params(axis='x', rotation=45)
    axes[idx].legend(title='Пол')
    axes[idx].grid(True, alpha=0.3)

# Adjust layout
plt.tight_layout()
plt.savefig('gender_by_region.png', bbox_inches='tight', dpi=300)
plt.close() 