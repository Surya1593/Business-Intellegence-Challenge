# -*- coding: utf-8 -*-


import pandas as pd
import matplotlib.pyplot as plt
import warnings
import seaborn as sns
from operator import attrgetter
import matplotlib.colors as mcolors
import numpy as np

# Loading data here and geting 2 tables and 1 merged dataframe
def load_data(path1, path2):

  A = pd.read_csv(path1, sep = ",")
  B = pd.read_csv(path2, sep = ",")
  dataframe = pd.merge(A, B, on=['Conv_ID'])
  dataframe = dataframe.set_index('Conv_Date')
  dataframe.index = pd.to_datetime(dataframe.index)
  print("Initial data from 2 tables merged by Conv_ID:")
  print(dataframe)
  return (dataframe, A, B)


# Function that returns a number of customers with its distribution
def no_of_customers (dataframe):

  n_orders = dataframe.groupby(['User_ID'])['Conv_ID'].nunique()
  mult_orders_perc = np.sum(n_orders > 1) / dataframe['User_ID'].nunique()
  print("\n")
  print(f'{100 * mult_orders_perc:.2f}% of customers ordered more than once\n')

  ax = sns.distplot(n_orders, kde=False, hist=True)
  ax.set(title='\nDistribution of number of orders per customer',xlabel='# of orders',ylabel='# of customers')
  

  return (n_orders,mult_orders_perc, ax)


# Cohort analysis by months. Gives us a data about customers conversion fraction by month.
def cohort_analysis(n_orders,mult_orders_perc, ax, A, B):

  df = pd.merge(A, B, on=['Conv_ID'])
  df = df[['User_ID', 'Conv_ID', 'Conv_Date']]
  df['Conv_Date'] = pd.to_datetime(df['Conv_Date'], errors='coerce')
  df['order_month'] = df['Conv_Date'].dt.to_period('M')
  df['cohort'] = df.groupby('User_ID')['Conv_Date'] \
                  .transform('min') \
                  .dt.to_period('M') 
  df_cohort = df.groupby(['cohort', 'order_month']) \
                .agg(n_customers=('User_ID', 'nunique')) \
                .reset_index(drop=False)
  df_cohort['period_number'] = (df_cohort.order_month - df_cohort.cohort).apply(attrgetter('n'))
  cohort_pivot = df_cohort.pivot_table(index = 'cohort',
                                      columns = 'period_number',
                                      values = 'n_customers')
  cohort_size = cohort_pivot.iloc[:,0]
  retention_matrix = cohort_pivot.divide(cohort_size, axis = 0)


  with sns.axes_style("white"):
      fig, ax = plt.subplots(1, 2, figsize=(12, 8), sharey=True, gridspec_kw={'width_ratios': [1, 11]})
      
      # retention matrix
      sns.heatmap(retention_matrix, 
                  mask=retention_matrix.isnull(), 
                  annot=True, 
                  fmt='.0%', 
                  cmap='RdYlGn', 
                  ax=ax[1])
      ax[1].set_title('\nMonthly Cohorts: User Retention', fontsize=16)
      ax[1].set(xlabel='# of periods',
                ylabel='')
      

      # cohort size
      cohort_size_df = pd.DataFrame(cohort_size).rename(columns={0: 'cohort_size'})
      white_cmap = mcolors.ListedColormap(['white'])
      sns.heatmap(cohort_size_df, 
                  annot=True, 
                  cbar=False, 
                  fmt='g', 
                  cmap=white_cmap, 
                  ax=ax[0])
      
    

      fig.tight_layout()


path1 = "table_A_conversions.csv" # table 1 name 
path2 = "table_B_attribution.csv" # table 2 name 
dataframe, A, B = load_data(path1, path2) # load and clean data and return a dataframe


# Resampling by months and weeks and their line charts
data1 = dataframe['Revenue'].resample('MS').mean()
data1.plot(title="\nRevenue over time period (months)",figsize=(15, 6))

plt.show()


data2 = dataframe['Revenue'].resample('W').mean()
data2.plot(title="\nRevenue over time period (weeks)",figsize=(15, 6))

plt.show()



# Customers segmentation by most popular channel
print("\nNumber of all values by the Channel")
print(dataframe.groupby(['Channel']).count())

print("\nNumber of Customers by the Channel")
data3 = dataframe.groupby('Channel')['User_ID'].nunique()
print(data3)

# Call cohort analysis and number of customers analysis via functions
n_orders,mult_orders_perc,ax = no_of_customers(dataframe)
cohort_analysis(n_orders,mult_orders_perc, ax, A, B)

