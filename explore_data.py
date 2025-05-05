import pandas as pd

# Load the dataset
df = pd.read_csv('/home/ubuntu/data_analysis_project/Titanic-Dataset.csv')

# Display basic information
print('--- Dataset Head ---')
print(df.head())
print('\n--- Dataset Info ---')
df.info()
print('\n--- Dataset Description ---')
print(df.describe(include='all'))
print('\n--- Missing Values ---')
print(df.isnull().sum())

