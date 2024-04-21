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

served_df = served_df.iloc[:-1]
in_care_df = in_care_df.iloc[:-1]
entered_df = entered_df.iloc[:-1]
waiting_df = waiting_df.iloc[:-1]
terminated_df = terminated_df.iloc[:-1]
adopted_df = adopted_df.iloc[:-1]

# Exploratory Data Analysis
# Exclude rows corresponding to "Puerto Rico*" and "Total"
served_df_excluding_pr_and_total = served_df[~served_df.index.str.contains('Puerto Rico|Total')]
total_served_per_year = served_df_excluding_pr_and_total.sum(axis=0)
total_served_per_year_df = total_served_per_year.to_frame().T
#print("Total Served per Year (Excluding Puerto Rico and Total):")
#print(total_served_per_year_df)

in_care_df_excluding_pr_and_total = in_care_df[~in_care_df.index.str.contains('Puerto Rico|Total')]
total_in_care_per_year = in_care_df_excluding_pr_and_total.sum(axis=0)
total_in_care_per_year_df = total_in_care_per_year.to_frame().T
#print(total_in_care_per_year_df)

entered_df_excluding_pr_and_total = entered_df[~entered_df.index.str.contains('Puerto Rico|Total')]
total_entered_per_year = entered_df_excluding_pr_and_total.sum(axis=0)
total_entered_per_year_df = total_entered_per_year.to_frame().T
#print(total_entered_per_year_df)

waiting_df_excluding_pr_and_total = waiting_df[~waiting_df.index.str.contains('Puerto Rico|Total')]
total_waiting_per_year = waiting_df_excluding_pr_and_total.sum(axis=0)
total_waiting_per_year_df = total_waiting_per_year.to_frame().T
#print(total_waiting_per_year_df)

terminated_df_excluding_pr_and_total = terminated_df[~terminated_df.index.str.contains('Puerto Rico|Total')]
total_terminated_per_year = terminated_df_excluding_pr_and_total.sum(axis=0)
total_terminated_per_year_df = total_terminated_per_year.to_frame().T
#print(total_terminated_per_year_df)

adopted_df_excluding_pr_and_total = adopted_df[~adopted_df.index.str.contains('Puerto Rico|Total')]
total_adopted_per_year = adopted_df_excluding_pr_and_total.sum(axis=0)
total_adopted_per_year_df = total_adopted_per_year.to_frame().T
#print(total_adopted_per_year_df)
"""
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
#plt.show()

#Plot the number of children in-care at the end of each year
plt.figure(figsize=(10, 6))
plt.plot(total_in_care_per_year_df.columns.astype(int), total_in_care_per_year_df.values.flatten(), marker='o', linestyle='-')
plt.title('Total Number of Children In-Care per Year (Excluding Puerto Rico)')
plt.xlabel('Year')
plt.ylabel('Total Number of Children Still In Care at the end of the Year')
plt.grid(True)
plt.xticks(total_in_care_per_year_df.columns.astype(int), rotation=45)
plt.tight_layout()
#plt.show()

#Plot of the number of children that are waiting to be adopted per year
plt.figure(figsize=(10, 6))
plt.plot(total_waiting_per_year_df.columns.astype(int), total_waiting_per_year_df.values.flatten(), marker='o', linestyle='-')
plt.title('Total Number of Children that are Waiting to be Adopted Each Year (Excluding Puerto Rico)')
plt.xlabel('Year')
plt.ylabel('Total Number of Children Waiting to be Adopted')
plt.grid(True)
plt.xticks(total_waiting_per_year_df.columns.astype(int), rotation=45)
plt.tight_layout()
#plt.show()

#Plot of the number of children that entered the foster care system per year
plt.figure(figsize=(10, 6))
plt.plot(total_entered_per_year_df.columns.astype(int), total_entered_per_year_df.values.flatten(), marker='o', linestyle='-')
plt.title('Total Number of Children that Entered the Foster Care System Each Year (Excluding Puerto Rico)')
plt.xlabel('Year')
plt.ylabel('Total Number of Children Entering Foster Care')
plt.grid(True)
plt.xticks(total_entered_per_year_df.columns.astype(int), rotation=45)
plt.tight_layout()
#plt.show()

#Plot of the number of parental rights terminated per year
plt.figure(figsize=(10, 6))
plt.plot(total_terminated_per_year_df.columns.astype(int), total_terminated_per_year_df.values.flatten(), marker='o', linestyle='-')
plt.title('Total Number of Parental Rights Terminated Each Year (Excluding Puerto Rico)')
plt.xlabel('Year')
plt.ylabel('Total Number of Parental Rights Terminated')
plt.grid(True)
plt.xticks(total_terminated_per_year_df.columns.astype(int), rotation=45)
plt.tight_layout()
#plt.show()

# Plot the total number of children adopted over the years
plt.figure(figsize=(10, 6))
plt.plot(total_adopted_per_year_df.columns.astype(int), total_adopted_per_year_df.values.flatten(), marker='o', linestyle='-')
plt.title('Total Number of Children Adopted per Year (Excluding Puerto Rico)')
plt.xlabel('Year')
plt.ylabel('Total Number of Children Adopted per Year')
plt.grid(True)
plt.xticks(total_adopted_per_year_df.columns.astype(int), rotation=45)
plt.tight_layout()
#plt.show()
total_per_year_list = [total_entered_per_year_df, total_adopted_per_year_df, total_in_care_per_year_df, total_terminated_per_year_df, total_served_per_year_df, total_waiting_per_year_df]
labels = ['Served', 'In Care', 'Entered', 'Waiting for Adoption', 'Terminated', 'Adopted']
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
#plt.show()

#Heatmap
import geopandas as gpd

# Load the shapefile of the United States
usa = gpd.read_file("C:/Users/mpdes/Downloads/cb_2022_us_all_500k/cb_2022_us_state_500k/cb_2022_us_state_500k.shp")

# Filter out US territories
usa_states = usa[~usa['NAME'].isin(['United States Virgin Islands', 'Guam', 'Puerto Rico', 'Commonwealth of the Northern Mariana Islands', 'American Samoa'])]

# Merge the filtered shapefile with adoption data
usa_adoption = usa_states.merge(adopted_df_excluding_pr_and_total, how='left', left_on='NAME', right_index=True)

# Plot the map using the number of children adopted in 2022
fig, ax = plt.subplots(1, 1, figsize=(42, 30))
usa_adoption.plot(column=2022, cmap='YlGnBu', linewidth=0.8, ax=ax, edgecolor='0.8', legend=True)
plt.title('Number of Children Adopted in the United States (2022)')
plt.axis('off')
plt.show()
"""
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_absolute_error
train_data = adopted_df_excluding_pr_and_total.iloc[:-2]
test_data = adopted_df_excluding_pr_and_total.iloc[-2:]

# Example: Train the ARIMA model for each state
for state in adopted_df_excluding_pr_and_total.columns:
    # Train the ARIMA model
    model = ARIMA(train_data[state], order=(1, 1, 1))
    model_fit = model.fit()

    # Make predictions on the test data
    predictions = model_fit.forecast(steps=2)

    # Evaluate the model
    mae = mean_absolute_error(test_data[state], predictions)
    print(f'Mean Absolute Error for {state}: {mae}')

    # Forecast future values
    future_forecast = model_fit.forecast(steps=2)
    print(f'Forecast for {state} for next {2} years: {future_forecast}')
"""
# Forecasting
# Forecasting with Non-Seasonal ARIMA
from statsmodels.tsa.arima.model import ARIMA
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error

# Splitting the data into features (X) and target variable (y)
X = adopted_df_excluding_pr_and_total.drop(columns=[2022])  # Features (exclude the last year as target variable)
y = adopted_df_excluding_pr_and_total[2022]  # Target variable (number of adoptions in 2022)
# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
# Training a non-seasonal ARIMA model
model = ARIMA(y_train, order=(1, 1, 1))
model_fit = model.fit()
#Making predictions on the test set
y_pred = model_fit.predict(start=len(X_train), end=len(X_train) + len(X_test) - 1)
#Evaluating the model
mae = mean_absolute_error(y_test, y_pred)
print(f'Mean Absolute Error: {mae}')
# Example: Forecasting future values
future_forecast = model_fit.forecast(steps=2)  # Forecasting 2 years into the future
print(future_forecast)"""
