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

import matplotlib.pyplot as plt
"""
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
plt.show()"""
total_per_year_list = [total_entered_per_year_df, total_adopted_per_year_df, total_in_care_per_year_df, total_terminated_per_year_df, total_served_per_year_df, total_waiting_per_year_df]
labels = ['Served', 'In Care', 'Entered', 'Waiting for Adoption', 'Terminated', 'Adopted']
combined_df = pd.concat(total_per_year_list)
# Plot the total number of children served per year for each dataframe
plt.figure(figsize=(10, 6))
for df, label in total_per_year_list:
    if df.index.name:  # Check if index name is not empty
        plt.plot(df.columns.astype(int), df.values.flatten(), label=label)
plt.title('Total Number of Children Served per Year')
plt.xlabel('Year')
plt.ylabel('Total Number of Children Served')
plt.grid(True)
plt.legend()
plt.xticks(df.columns.astype(int), rotation=45)
plt.tight_layout()
plt.show()

