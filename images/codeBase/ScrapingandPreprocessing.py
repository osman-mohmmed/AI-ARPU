# %%
import requests
from bs4 import BeautifulSoup
from tabulate import tabulate
from io import StringIO
import pandas as pd
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from scipy import stats
import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import sys
# sys.path.append(r'\code')
from LinearRegDiagnostic import LinearRegDiagnostic

# %%
'''DATA-SCRAPING
    Uploaded data to GitHup Repository
    Scraped the data from raw'''

# Raw URL of the Markdown file
url ='https://raw.githubusercontent.com/osman-mohmmed/aibas/refs/heads/main/data/telecom_arpu_data.md'

# Step 1: Fetch the Markdown content
response = requests.get(url)
response.raise_for_status()
md_content = response.text

# Step 2: Manually process the Markdown table
# Split the content into lines
lines = md_content.splitlines()

# Find the start of the table (look for |---|)
table_start = -1
for i, line in enumerate(lines):
    if '|---' in line:
        table_start = i - 1  # Header is the line before this
        break

if table_start == -1:
    print("No table found in the Markdown file.")
else:
    # Extract the table lines
    table_lines = lines[table_start:]
    table_text = "\n".join(table_lines)

    # Use pandas to parse the table
    df = pd.read_csv(StringIO(table_text), sep="|").iloc[:, 1:-1]  # Trim empty columns
  
df.describe()


# %%
''' Preprocessing:
    1-DATA-CLEANING 
        -Null values handelling :
            -replace 'nan' aith np.nan.
            -droped columns with 10k nulls & droped rows with null values.'''



df = df.applymap(lambda x: x.strip() if isinstance(x, str) else x)
df.columns = df.columns.str.strip()
#delete noisy chacters ':----'
df.drop(df[df['mobile_number'] == '----------------:'].index,inplace= True)

# Remove null values with a threshold

df = df.replace(['NaN', 'nan', 'null', '', None], np.nan, regex=True)
null_counts = df.isnull().sum()
threshold = 10000
df = df.loc[:, df.isnull().sum() <= threshold]
print(null_counts)

#drop rows with null values
df = df.dropna()
df.describe()


# %%
'''Data type '''
#drop date columns
df.drop(['last_date_of_month_9','last_date_of_month_8','last_date_of_month_7','last_date_of_month_6'], axis=1, inplace = True )
df.drop(['date_of_last_rech_6','date_of_last_rech_7','date_of_last_rech_8','date_of_last_rech_9'], axis = 1, inplace=True)

#drop id columns
df.drop(['mobile_number','circle_id'], axis=1, inplace=True) 

#convert to numeric
df = df.apply(pd.to_numeric)

df.dtypes


# %%
'''Data visualisation:
    boxplot of lable column'''
plt.figure(figsize=(8,8))
sns.boxplot( y = 'arpu_9', data = df) 

plt.savefig('tmp/learningBase/DP_boxplot_before.png')

# %%
#detect and remove outliers with zscores
zs = stats.zscore(df['arpu_9'])
zs_ind = abs(zs) < 2
df_filtered = df[zs_ind]
plt.figure(figsize=(8,8))
sns.boxplot( y = 'arpu_9', data = df_filtered) 
plt.savefig('tmp/learningBase/DP_boxplot_after.png')

# %%
'''Modeling
    1| OLS Model'''

# Define the split ratio
split_ratio = 0.8

# Split index
split_index = int(len(df_filtered) * split_ratio)

# Split data
train_df = df_filtered[:split_index]
test_df = df_filtered[split_index:]

#
lable = df_filtered['arpu_9']
features = df_filtered.drop('arpu_9', axis=1)

X_train = train_df[features.columns] 
X_test = test_df[features.columns] 
y_train = train_df['arpu_9']
y_test = test_df['arpu_9']

activation = test_df.iloc[[7]]

train_df.to_csv("tmp/learningBase/train/training_data.csv", index = False)
test_df.to_csv("tmp/learningBase/validation/test_data.csv",index = False)
activation.to_csv("tmp/activationBase/activation_data.csv",index = False)

df_filtered.to_csv("tmp/learningBase/joint_data_collection.csv")
