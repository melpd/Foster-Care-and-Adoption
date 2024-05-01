import pandas as pd

xls= pd.ExcelFile("C:/Users/mpdes/Downloads/state-afcars-data-2013-2022.xlsx")
# Read specific sheets
served_df = pd.read_excel(xls, 'Served')
in_care_df = pd.read_excel(xls, 'In Care on September 30th')
entered_df = pd.read_excel(xls, 'Entered')
exited_df = pd.read_excel(xls, 'Exited')
waiting_df = pd.read_excel(xls, 'Waiting for Adoption')
terminated_df = pd.read_excel(xls, 'Parental Rights Terminated')
adopted_df = pd.read_excel(xls, 'Adopted')

#Data Cleaning

# Clean and format dataframes
def clean_dataframe(df):
    df = df.iloc[7:].reset_index(drop=True) # Remove top section with titles
    df.columns.values[0] = 'States' # Rename the first column to 'States'
    years = range(2013, 2023)
    df.columns = ['States'] + list(years) # Set columns as years 2013-2022
    df.set_index('States', inplace=True)
    df.fillna(0, inplace=True)
    for col in df.columns:
        df[col] = df[col].astype(int)
    return df
# Clean dataframes by removing empty spaces
served_df = clean_dataframe(served_df)
in_care_df = clean_dataframe(in_care_df)
entered_df = clean_dataframe(entered_df)
waiting_df = clean_dataframe(waiting_df)
terminated_df = clean_dataframe(terminated_df)
adopted_df = clean_dataframe(adopted_df)
exited_df = clean_dataframe(exited_df)

served_df = served_df.iloc[:-1]
in_care_df = in_care_df.iloc[:-1]
entered_df = entered_df.iloc[:-1]
waiting_df = waiting_df.iloc[:-1]
terminated_df = terminated_df.iloc[:-1]
adopted_df = adopted_df.iloc[:-1]
exited_df = exited_df.iloc[:-1]

# Exclude rows corresponding to "Puerto Rico*" and "Total"
served_df_excluding_pr_and_total = served_df[~served_df.index.str.contains('Puerto Rico|Total')]
in_care_df_excluding_pr_and_total = in_care_df[~in_care_df.index.str.contains('Puerto Rico|Total')]
entered_df_excluding_pr_and_total = entered_df[~entered_df.index.str.contains('Puerto Rico|Total')]
waiting_df_excluding_pr_and_total = waiting_df[~waiting_df.index.str.contains('Puerto Rico|Total')]
terminated_df_excluding_pr_and_total = terminated_df[~terminated_df.index.str.contains('Puerto Rico|Total')]
adopted_df_excluding_pr_and_total = adopted_df[~adopted_df.index.str.contains('Puerto Rico|Total')]
exited_df_excluding_pr_and_total = exited_df[~exited_df.index.str.contains('Puerto Rico|Total')]

# Exploratory Data Analysis
total_served_per_year = served_df_excluding_pr_and_total.sum(axis=0)
total_served_per_year_df = total_served_per_year.to_frame().T

total_in_care_per_year = in_care_df_excluding_pr_and_total.sum(axis=0)
total_in_care_per_year_df = total_in_care_per_year.to_frame().T

total_entered_per_year = entered_df_excluding_pr_and_total.sum(axis=0)
total_entered_per_year_df = total_entered_per_year.to_frame().T

total_waiting_per_year = waiting_df_excluding_pr_and_total.sum(axis=0)
total_waiting_per_year_df = total_waiting_per_year.to_frame().T

total_terminated_per_year = terminated_df_excluding_pr_and_total.sum(axis=0)
total_terminated_per_year_df = total_terminated_per_year.to_frame().T

total_adopted_per_year = adopted_df_excluding_pr_and_total.sum(axis=0)
total_adopted_per_year_df = total_adopted_per_year.to_frame().T

total_exited_per_year = exited_df_excluding_pr_and_total.sum(axis=0)
total_exited_per_year_df = total_exited_per_year.to_frame().T

import matplotlib.pyplot as plt

# Plot the total number of children served over the years
plt.figure(figsize=(10, 6))
plt.plot(total_served_per_year_df.columns.astype(int), total_served_per_year_df.values.flatten(), marker='o', linestyle='-')
plt.title('Total Number of Children in Foster Care per Year (Excluding Puerto Rico)')
plt.xlabel('Year')
plt.ylabel('Total Number of Children Served by the Foster Care System')
plt.grid(True)
plt.xticks(total_served_per_year_df.columns.astype(int), rotation=45)
plt.tight_layout()
plt.show()

#Plot the number of children in-care at the end of each year
plt.figure(figsize=(10, 6))
plt.plot(total_in_care_per_year_df.columns.astype(int), total_in_care_per_year_df.values.flatten(), marker='o', linestyle='-')
plt.title('Total Number of Children In-Care per Year (Excluding Puerto Rico)')
plt.xlabel('Year')
plt.ylabel('Total Number of Children Still In Care at the end of the Year')
plt.grid(True)
plt.xticks(total_in_care_per_year_df.columns.astype(int), rotation=45)
plt.tight_layout()
plt.show()

#Plot of the number of children that are waiting to be adopted per year
plt.figure(figsize=(10, 6))
plt.plot(total_waiting_per_year_df.columns.astype(int), total_waiting_per_year_df.values.flatten(), marker='o', linestyle='-')
plt.title('Total Number of Children that are Waiting to be Adopted Each Year (Excluding Puerto Rico)')
plt.xlabel('Year')
plt.ylabel('Total Number of Children Waiting to be Adopted')
plt.grid(True)
plt.xticks(total_waiting_per_year_df.columns.astype(int), rotation=45)
plt.tight_layout()
plt.show()

#Plot of the number of children that entered the foster care system per year
plt.figure(figsize=(10, 6))
plt.plot(total_entered_per_year_df.columns.astype(int), total_entered_per_year_df.values.flatten(), marker='o', linestyle='-')
plt.title('Total Number of Children that Entered the Foster Care System Each Year (Excluding Puerto Rico)')
plt.xlabel('Year')
plt.ylabel('Total Number of Children Entering Foster Care')
plt.grid(True)
plt.xticks(total_entered_per_year_df.columns.astype(int), rotation=45)
plt.tight_layout()
plt.show()

#Plot of the number of parental rights terminated per year
plt.figure(figsize=(10, 6))
plt.plot(total_terminated_per_year_df.columns.astype(int), total_terminated_per_year_df.values.flatten(), marker='o', linestyle='-')
plt.title('Total Number of Parental Rights Terminated Each Year (Excluding Puerto Rico)')
plt.xlabel('Year')
plt.ylabel('Total Number of Parental Rights Terminated')
plt.grid(True)
plt.xticks(total_terminated_per_year_df.columns.astype(int), rotation=45)
plt.tight_layout()
plt.show()

# Plot the total number of children adopted over the years
plt.figure(figsize=(10, 6))
plt.plot(total_adopted_per_year_df.columns.astype(int), total_adopted_per_year_df.values.flatten(), marker='o', linestyle='-')
plt.title('Total Number of Children Adopted per Year (Excluding Puerto Rico)')
plt.xlabel('Year')
plt.ylabel('Total Number of Children Adopted per Year')
plt.grid(True)
plt.xticks(total_adopted_per_year_df.columns.astype(int), rotation=45)
plt.tight_layout()
plt.show()

# Plot the total number of children exiting over the years
plt.figure(figsize=(10, 6))
plt.plot(total_exited_per_year_df.columns.astype(int), total_exited_per_year_df.values.flatten(), marker='o', linestyle='-')
plt.title('Total Number of Children Exiting Foster Care per Year (Excluding Puerto Rico)')
plt.xlabel('Year')
plt.ylabel('Total Number of Children Exiting Foster Care')
plt.grid(True)
plt.xticks(total_exited_per_year_df.columns.astype(int), rotation=45)
plt.tight_layout()
plt.show()

total_per_year_list = [total_exited_per_year_df ,total_entered_per_year_df, total_adopted_per_year_df, total_in_care_per_year_df, total_terminated_per_year_df, total_served_per_year_df, total_waiting_per_year_df]
labels = ['Exited','Entered', 'Adopted', 'In-Care', 'Terminated', 'Served', 'Waiting']
combined_df = pd.concat(total_per_year_list)
# Plot the total number of children served per year for each dataframe
plt.figure(figsize=(10, 6))
for df, label in zip(total_per_year_list, labels):
    plt.plot(df.columns.astype(int), df.values.flatten(), marker='o', linestyle='-', label=label)
plt.title('Total Number of Children Served per Year')
plt.xlabel('Year')
plt.ylabel('Total Number of Children Served')
plt.grid(True)
plt.legend()
plt.xticks(df.columns.astype(int), rotation=45)
plt.tight_layout()
plt.show()

#Heatmap
import geopandas as gpd

# Load the shapefile of the United States
usa = gpd.read_file("C:/Users/mpdes/Downloads/cb_2022_us_all_500k/cb_2022_us_state_500k/cb_2022_us_state_500k.shp")
# Filter out US territories
usa_states = usa[~usa['NAME'].isin(['United States Virgin Islands', 'Guam', 'Puerto Rico', 'Commonwealth of the Northern Mariana Islands', 'American Samoa'])]
# Merge the filtered shapefile with adoption data
usa_adoption = usa_states.merge(adopted_df_excluding_pr_and_total, how='left', left_on='NAME', right_index=True)
"""
# Plot the map using the number of children adopted in 2022
fig, ax = plt.subplots(1, 1, figsize=(42, 30))
usa_adoption.plot(column=2022, cmap='YlGnBu', linewidth=0.8, ax=ax, edgecolor='0.8', legend=True)
plt.title('Number of Children Adopted in the United States (2022)')
plt.axis('off')
plt.show()

usa_served = usa_states.merge(served_df_excluding_pr_and_total, how='left', left_on='NAME', right_index=True)
fig, ax = plt.subplots(1, 1, figsize=(42, 30))
usa_served.plot(column=2022, cmap='YlGnBu', linewidth=0.8, ax=ax, edgecolor='0.8', legend=True)
plt.title('Number of Children Served by the Foster Care System in the United States (2022)')
plt.axis('off')
plt.show()

usa_in_care = usa_states.merge(in_care_df_excluding_pr_and_total, how='left', left_on='NAME', right_index=True)
fig, ax = plt.subplots(1, 1, figsize=(42, 30))
usa_adoption.plot(column=2022, cmap='YlGnBu', linewidth=0.8, ax=ax, edgecolor='0.8', legend=True)
plt.title('Number of Children In-Care in the United States (2022)')
plt.axis('off')
plt.show()

usa_entered = usa_states.merge(entered_df_excluding_pr_and_total, how='left', left_on='NAME', right_index=True)
fig, ax = plt.subplots(1, 1, figsize=(42, 30))
usa_entered.plot(column=2022, cmap='YlGnBu', linewidth=0.8, ax=ax, edgecolor='0.8', legend=True)
plt.title('Number of Children that Entered the Foster Care System in the United States (2022)')
plt.axis('off')
plt.show()

usa_waiting = usa_states.merge(waiting_df_excluding_pr_and_total, how='left', left_on='NAME', right_index=True)
fig, ax = plt.subplots(1, 1, figsize=(42, 30))
usa_adoption.plot(column=2022, cmap='YlGnBu', linewidth=0.8, ax=ax, edgecolor='0.8', legend=True)
plt.title('Number of Children Waiting to be Adopted in the United States (2022)')
plt.axis('off')
plt.show()

usa_terminated = usa_states.merge(terminated_df_excluding_pr_and_total, how='left', left_on='NAME', right_index=True)
fig, ax = plt.subplots(1, 1, figsize=(42, 30))
usa_terminated.plot(column=2022, cmap='YlGnBu', linewidth=0.8, ax=ax, edgecolor='0.8', legend=True)
plt.title('Number of Children With Parental Rights Terminated in the United States (2022)')
plt.axis('off')
plt.show()

usa_exited = usa_states.merge(exited_df_excluding_pr_and_total, how='left', left_on='NAME', right_index=True)
fig, ax = plt.subplots(1, 1, figsize=(42, 30))
usa_exited.plot(column=2022, cmap='YlGnBu', linewidth=0.8, ax=ax, edgecolor='0.8', legend=True)
plt.title('Number of Children Exiting Foster Care in the United States (2022)')
plt.axis('off')
plt.show()
"""
from statsmodels.tsa.arima.model import ARIMA

# Check if the data is stationary
from statsmodels.tsa.stattools import adfuller
#Determine if the data is stationary
# Iterate over each column (year) in the DataFrame
for year in adopted_df_excluding_pr_and_total.columns:
    # Perform ADF test for stationarity on the data for the current year
    result = adfuller(adopted_df_excluding_pr_and_total[year].dropna())
    print(f'Year: {year}')
    print('ADF Statistic:', result[0])
    print('p-value:', result[1])
    print('Critical Values:')
    for key, value in result[4].items():
        print(f'   {key}: {value}')
    print('\n')

import itertools
import matplotlib.pyplot as plt
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.arima.model import ARIMA
"""
# Function to determine the optimal (p, d, q) parameters for ARIMA using AIC
def find_best_arima_parameters(train_data):
    best_aic = float('inf')
    best_order = None
    
    # Define the range of values for p and q
    p_range = range(0, 5)  # Adjust the range as needed
    q_range = range(0, 5)  # Adjust the range as needed
    
    # Iterate over all possible combinations of p and q
    for p, q in itertools.product(p_range, q_range):
        # Skip (0, 0) order as it's equivalent to a simple moving average model
        if p == 0 and q == 0:
            continue
        
        try:
            # Fit ARIMA model
            model = ARIMA(train_data, order=(p, 0, q))  # Assuming d=0 (data is stationary)
            fit_model = model.fit()
            
            # Check if current model has lower AIC than the best so far
            if fit_model.aic < best_aic:
                best_aic = fit_model.aic
                best_order = (p, 0, q)  # Assuming d=0
        except:
            continue
    
    return best_order

# Plot ACF and PACF to determine p and q
def plot_acf_pacf(train_data):
    fig, ax = plt.subplots(2, 1, figsize=(12, 8))
    plot_acf(train_data, ax=ax[0])
    plot_pacf(train_data, ax=ax[1])
    plt.show()

# Split data into training and testing sets
train_df, test_df = train_test_split(adopted_df_excluding_pr_and_total, test_size=0.2, random_state=42)

# Loop over each state
for state, train_data in train_df.iterrows():
    # Plot ACF and PACF
    plot_acf_pacf(train_data)
    
    # Determine optimal (p, q) parameters using AIC
    p, q = find_best_arima_parameters(train_data)
    
    print(f'Best (p, q) for {state}: ({p}, 0, {q})')  # Assuming d=0 for stationary data

"""
import pandas as pd
from statsmodels.tsa.arima.model import ARIMA

# Loop over each state and fit ARIMA model
for state, data in adopted_df_excluding_pr_and_total.iterrows():
    train_data = data.dropna()
    # Convert index to DatetimeIndex with yearly frequency
    start_year = 2013 
    end_year = 2022
    years = range(start_year, end_year + 1)
    train_data.index = pd.to_datetime([f'{year}-01-01' for year in years])
    model = ARIMA(train_data, order=(1, 0, 1))
    fit_model = model.fit()
    # Make forecasts
    forecast = fit_model.forecast(steps=3)
    adopted_df_excluding_pr_and_total.loc[state, 2023] = forecast.iloc[0]
    adopted_df_excluding_pr_and_total.loc[state, 2024] = forecast.iloc[1]
    adopted_df_excluding_pr_and_total.loc[state, 2025] = forecast.iloc[2]

#print(adopted_df_excluding_pr_and_total.head())

for state, data in in_care_df_excluding_pr_and_total.iterrows():
    train_data = data.dropna()
    # Convert index to DatetimeIndex with yearly frequency
    start_year = 2013 
    end_year = 2022
    years = range(start_year, end_year + 1)
    train_data.index = pd.to_datetime([f'{year}-01-01' for year in years])
    model = ARIMA(train_data, order=(1, 0, 1))
    fit_model = model.fit()
    # Make forecasts
    forecast = fit_model.forecast(steps=3)
    in_care_df_excluding_pr_and_total.loc[state, 2023] = forecast.iloc[0]
    in_care_df_excluding_pr_and_total.loc[state, 2024] = forecast.iloc[1]
    in_care_df_excluding_pr_and_total.loc[state, 2025] = forecast.iloc[2]
#print(in_care_df_excluding_pr_and_total.head())

for state, data in entered_df_excluding_pr_and_total.iterrows():
    train_data = data.dropna()
    # Convert index to DatetimeIndex with yearly frequency
    start_year = 2013 
    end_year = 2022
    years = range(start_year, end_year + 1)
    train_data.index = pd.to_datetime([f'{year}-01-01' for year in years])
    model = ARIMA(train_data, order=(1, 0, 1))
    fit_model = model.fit()
    # Make forecasts
    forecast = fit_model.forecast(steps=3)
    entered_df_excluding_pr_and_total.loc[state, 2023] = forecast.iloc[0]
    entered_df_excluding_pr_and_total.loc[state, 2024] = forecast.iloc[1]
    entered_df_excluding_pr_and_total.loc[state, 2025] = forecast.iloc[2]
#print(entered_df_excluding_pr_and_total.head())

for state, data in waiting_df_excluding_pr_and_total.iterrows():
    train_data = data.dropna()
    # Convert index to DatetimeIndex with yearly frequency
    start_year = 2013 
    end_year = 2022
    years = range(start_year, end_year + 1)
    train_data.index = pd.to_datetime([f'{year}-01-01' for year in years])
    model = ARIMA(train_data, order=(1, 0, 1))
    fit_model = model.fit()
    # Make forecasts
    forecast = fit_model.forecast(steps=3)
    waiting_df_excluding_pr_and_total.loc[state, 2023] = forecast.iloc[0]
    waiting_df_excluding_pr_and_total.loc[state, 2024] = forecast.iloc[1]
    waiting_df_excluding_pr_and_total.loc[state, 2025] = forecast.iloc[2]
#print(waiting_df_excluding_pr_and_total.head())

for state, data in exited_df_excluding_pr_and_total.iterrows():
    train_data = data.dropna()
    # Convert index to DatetimeIndex with yearly frequency
    start_year = 2013 
    end_year = 2022
    years = range(start_year, end_year + 1)
    train_data.index = pd.to_datetime([f'{year}-01-01' for year in years])
    model = ARIMA(train_data, order=(1, 0, 1))
    fit_model = model.fit()
    # Make forecasts
    forecast = fit_model.forecast(steps=3)
    exited_df_excluding_pr_and_total.loc[state, 2023] = forecast.iloc[0]
    exited_df_excluding_pr_and_total.loc[state, 2024] = forecast.iloc[1]
    exited_df_excluding_pr_and_total.loc[state, 2025] = forecast.iloc[2]


for state, data in terminated_df_excluding_pr_and_total.iterrows():
    train_data = data.dropna()
    # Convert index to DatetimeIndex with yearly frequency
    start_year = 2013 
    end_year = 2022
    years = range(start_year, end_year + 1)
    train_data.index = pd.to_datetime([f'{year}-01-01' for year in years])
    model = ARIMA(train_data, order=(1, 0, 1))
    fit_model = model.fit()
    # Make forecasts
    forecast = fit_model.forecast(steps=3)
    terminated_df_excluding_pr_and_total.loc[state, 2023] = forecast.iloc[0]
    terminated_df_excluding_pr_and_total.loc[state, 2024] = forecast.iloc[1]
    terminated_df_excluding_pr_and_total.loc[state, 2025] = forecast.iloc[2]
#print(terminated_df_excluding_pr_and_total.head())

for state, data in served_df_excluding_pr_and_total.iterrows():
    train_data = data.dropna()
    # Convert index to DatetimeIndex with yearly frequency
    start_year = 2013 
    end_year = 2022
    years = range(start_year, end_year + 1)
    train_data.index = pd.to_datetime([f'{year}-01-01' for year in years])
    model = ARIMA(train_data, order=(1, 0, 1))
    fit_model = model.fit()
    # Make forecasts
    forecast = fit_model.forecast(steps=3)
    served_df_excluding_pr_and_total.loc[state, 2023] = forecast.iloc[0]
    served_df_excluding_pr_and_total.loc[state, 2024] = forecast.iloc[1]
    served_df_excluding_pr_and_total.loc[state, 2025] = forecast.iloc[2]
print(served_df_excluding_pr_and_total.head())

#Heatmaps of Updated Predictions
# Plot the total number of children served over the years
total_served_per_year = served_df_excluding_pr_and_total.sum(axis=0)
total_served_per_year_df = total_served_per_year.to_frame().T
plt.figure(figsize=(10, 6))
plt.plot(total_served_per_year_df.columns.astype(int), total_served_per_year_df.values.flatten(), marker='o', linestyle='-')
plt.title('Total Number of Children in Foster Care per Year (Excluding Puerto Rico)')
plt.xlabel('Year')
plt.ylabel('Total Number of Children Served by the Foster Care System')
plt.grid(True)
plt.xticks(total_served_per_year_df.columns.astype(int), rotation=45)
plt.tight_layout()
plt.show()

#Plot the number of children in-care at the end of each year
total_in_care_per_year = in_care_df_excluding_pr_and_total.sum(axis=0)
total_in_care_per_year_df = total_in_care_per_year.to_frame().T
plt.figure(figsize=(10, 6))
plt.plot(total_in_care_per_year_df.columns.astype(int), total_in_care_per_year_df.values.flatten(), marker='o', linestyle='-')
plt.title('Total Number of Children In-Care per Year (Excluding Puerto Rico)')
plt.xlabel('Year')
plt.ylabel('Total Number of Children Still In Care at the end of the Year')
plt.grid(True)
plt.xticks(total_in_care_per_year_df.columns.astype(int), rotation=45)
plt.tight_layout()
plt.show()

#Plot of the number of children that are waiting to be adopted per year
total_waiting_per_year = waiting_df_excluding_pr_and_total.sum(axis=0)
total_waiting_per_year_df = total_waiting_per_year.to_frame().T
plt.figure(figsize=(10, 6))
plt.plot(total_waiting_per_year_df.columns.astype(int), total_waiting_per_year_df.values.flatten(), marker='o', linestyle='-')
plt.title('Total Number of Children that are Waiting to be Adopted Each Year (Excluding Puerto Rico)')
plt.xlabel('Year')
plt.ylabel('Total Number of Children Waiting to be Adopted')
plt.grid(True)
plt.xticks(total_waiting_per_year_df.columns.astype(int), rotation=45)
plt.tight_layout()
plt.show()

#Plot of the number of children that entered the foster care system per year
total_entered_per_year = entered_df_excluding_pr_and_total.sum(axis=0)
total_entered_per_year_df = total_entered_per_year.to_frame().T
plt.figure(figsize=(10, 6))
plt.plot(total_entered_per_year_df.columns.astype(int), total_entered_per_year_df.values.flatten(), marker='o', linestyle='-')
plt.title('Total Number of Children that Entered the Foster Care System Each Year (Excluding Puerto Rico)')
plt.xlabel('Year')
plt.ylabel('Total Number of Children Entering Foster Care')
plt.grid(True)
plt.xticks(total_entered_per_year_df.columns.astype(int), rotation=45)
plt.tight_layout()
plt.show()

#Plot of the number of parental rights terminated per year
total_terminated_per_year = terminated_df_excluding_pr_and_total.sum(axis=0)
total_terminated_per_year_df = total_terminated_per_year.to_frame().T
plt.figure(figsize=(10, 6))
plt.plot(total_terminated_per_year_df.columns.astype(int), total_terminated_per_year_df.values.flatten(), marker='o', linestyle='-')
plt.title('Total Number of Parental Rights Terminated Each Year (Excluding Puerto Rico)')
plt.xlabel('Year')
plt.ylabel('Total Number of Parental Rights Terminated')
plt.grid(True)
plt.xticks(total_terminated_per_year_df.columns.astype(int), rotation=45)
plt.tight_layout()
plt.show()

# Plot the total number of children adopted over the years
total_adopted_per_year = adopted_df_excluding_pr_and_total.sum(axis=0)
total_adopted_per_year_df = total_adopted_per_year.to_frame().T
plt.figure(figsize=(10, 6))
plt.plot(total_adopted_per_year_df.columns.astype(int), total_adopted_per_year_df.values.flatten(), marker='o', linestyle='-')
plt.title('Total Number of Children Adopted per Year (Excluding Puerto Rico)')
plt.xlabel('Year')
plt.ylabel('Total Number of Children Adopted per Year')
plt.grid(True)
plt.xticks(total_adopted_per_year_df.columns.astype(int), rotation=45)
plt.tight_layout()
plt.show()

# Plot the total number of children exiting over the years
total_exited_per_year = exited_df_excluding_pr_and_total.sum(axis=0)
total_exited_per_year_df = total_exited_per_year.to_frame().T
plt.figure(figsize=(10, 6))
plt.plot(total_exited_per_year_df.columns.astype(int), total_exited_per_year_df.values.flatten(), marker='o', linestyle='-')
plt.title('Total Number of Children Exiting Foster Care per Year (Excluding Puerto Rico)')
plt.xlabel('Year')
plt.ylabel('Total Number of Children Exiting Foster Care')
plt.grid(True)
plt.xticks(total_exited_per_year_df.columns.astype(int), rotation=45)
plt.tight_layout()
plt.show()

total_per_year_list = [total_exited_per_year_df ,total_entered_per_year_df, total_adopted_per_year_df, total_in_care_per_year_df, total_terminated_per_year_df, total_served_per_year_df, total_waiting_per_year_df]
labels = ['Exited','Entered', 'Adopted', 'In-Care', 'Terminated', 'Served', 'Waiting']
# Plot the total number of children served per year for each dataframe
plt.figure(figsize=(10, 6))
for df, label in zip(total_per_year_list, labels):
    plt.plot(df.columns.astype(int), df.values.flatten(), marker='o', linestyle='-', label=label)
plt.title('Total Number of Children Served per Year')
plt.xlabel('Year')
plt.ylabel('Total Number of Children Served')
plt.grid(True)
plt.legend()
plt.xticks(df.columns.astype(int), rotation=45)
plt.tight_layout()
plt.show()

#Heatmap
import geopandas as gpd

# Load the shapefile of the United States
usa = gpd.read_file("C:/Users/mpdes/Downloads/cb_2022_us_all_500k/cb_2022_us_state_500k/cb_2022_us_state_500k.shp")
"""
# Filter out US territories
usa_states = usa[~usa['NAME'].isin(['United States Virgin Islands', 'Guam', 'Puerto Rico', 'Commonwealth of the Northern Mariana Islands', 'American Samoa'])]

# Merge the filtered shapefile with adoption data
usa_adoption = usa_states.merge(adopted_df_excluding_pr_and_total, how='left', left_on='NAME', right_index=True)

# Plot the map using the number of children adopted in 2025
fig, ax = plt.subplots(1, 1, figsize=(42, 30))
usa_adoption.plot(column=2025, cmap='YlGnBu', linewidth=0.8, ax=ax, edgecolor='0.8', legend=True)
plt.title('Number of Children Adopted in the United States (2025)')
plt.axis('off')
plt.show()

usa_served = usa_states.merge(served_df_excluding_pr_and_total, how='left', left_on='NAME', right_index=True)
fig, ax = plt.subplots(1, 1, figsize=(42, 30))
usa_served.plot(column=2025, cmap='YlGnBu', linewidth=0.8, ax=ax, edgecolor='0.8', legend=True)
plt.title('Number of Children Served by the Foster Care System in the United States (2025)')
plt.axis('off')
plt.show()

usa_in_care = usa_states.merge(in_care_df_excluding_pr_and_total, how='left', left_on='NAME', right_index=True)
fig, ax = plt.subplots(1, 1, figsize=(42, 30))
usa_in_care.plot(column=2025, cmap='YlGnBu', linewidth=0.8, ax=ax, edgecolor='0.8', legend=True)
plt.title('Number of Children Currently In-Care in the United States (2025)')
plt.axis('off')
plt.show()

usa_entered = usa_states.merge(entered_df_excluding_pr_and_total, how='left', left_on='NAME', right_index=True)
fig, ax = plt.subplots(1, 1, figsize=(42, 30))
usa_entered.plot(column=2025, cmap='YlGnBu', linewidth=0.8, ax=ax, edgecolor='0.8', legend=True)
plt.title('Number of Children that Entered the Foster Care System in the United States (2025)')
plt.axis('off')
plt.show()

usa_waiting = usa_states.merge(waiting_df_excluding_pr_and_total, how='left', left_on='NAME', right_index=True)
fig, ax = plt.subplots(1, 1, figsize=(42, 30))
usa_adoption.plot(column=2025, cmap='YlGnBu', linewidth=0.8, ax=ax, edgecolor='0.8', legend=True)
plt.title('Number of Children Waiting to be Adopted in the United States (2025)')
plt.axis('off')
plt.show()

usa_terminated = usa_states.merge(terminated_df_excluding_pr_and_total, how='left', left_on='NAME', right_index=True)
fig, ax = plt.subplots(1, 1, figsize=(42, 30))
usa_terminated.plot(column=2025, cmap='YlGnBu', linewidth=0.8, ax=ax, edgecolor='0.8', legend=True)
plt.title('Number of Children With Parental Rights Terminated in the United States (2025)')
plt.axis('off')
plt.show()

usa_exited = usa_states.merge(exited_df_excluding_pr_and_total, how='left', left_on='NAME', right_index=True)
fig, ax = plt.subplots(1, 1, figsize=(42, 30))
usa_exited.plot(column=2025, cmap='YlGnBu', linewidth=0.8, ax=ax, edgecolor='0.8', legend=True)
plt.title('Number of Children Exiting Foster Care in the United States (2025)')
plt.axis('off')
plt.show()
"""
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

dfs = [adopted_df_excluding_pr_and_total, served_df_excluding_pr_and_total, waiting_df_excluding_pr_and_total, terminated_df_excluding_pr_and_total, entered_df_excluding_pr_and_total, exited_df_excluding_pr_and_total, in_care_df_excluding_pr_and_total]
#Normalize Each Data Frame
scaler = StandardScaler()
normalized_dfs = [scaler.fit_transform(df) for df in dfs]
# Cluster Each Normalized Data Frame
kmeans = KMeans(n_clusters=4)
cluster_results = [kmeans.fit_predict(df) for df in normalized_dfs]
# Combine Cluster Memberships
combined_clusters = pd.concat([pd.Series(cluster) for cluster in cluster_results], axis=1)
# Apply Clustering to Combined Representations
final_kmeans = KMeans(n_clusters=4)
final_clusters = final_kmeans.fit_predict(combined_clusters)
usa_map = usa.merge(pd.DataFrame(final_clusters, columns=['Cluster']), left_index=True, right_index=True)

# Plot the heatmap
fig, ax = plt.subplots(1, 1, figsize=(10, 8))
usa_map.plot(column='Cluster', cmap='viridis', linewidth=0.8, ax=ax, edgecolor='0.8', legend=True)
ax.set_title('Clustered States based on Foster Care System Trends')
plt.show()

# Get cluster centers
cluster_centers = final_kmeans.cluster_centers_

# Convert to DataFrame for easier analysis
cluster_centers_df = pd.DataFrame(cluster_centers, columns=combined_clusters.columns)

# Print cluster centers (average feature values)
print("Cluster Centers:")
print(cluster_centers_df)

# Group states by cluster and analyze characteristics
for cluster_label in range(final_kmeans.n_clusters):
    states_in_cluster = combined_clusters.index[final_clusters == cluster_label]
    print(f"\nCluster {cluster_label}:")
    print(f"States: {states_in_cluster}")
    # Optionally, print summary statistics or visualize characteristics of states within each cluster
    # For example:
    # print("Summary statistics:")
    # print(combined_clusters.loc[states_in_cluster].describe())
