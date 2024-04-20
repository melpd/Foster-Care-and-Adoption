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

#TO DO:
#- Data Cleaning
#   -Remove top section with titles
# remove the first 5 rows
# Remove top section with titles
def clean_dataframe(df):
    df = df.iloc[6:,].reset_index(drop=True)
    # Rename the first column to 'States'
    df = df.rename(columns={df.columns[0]: 'States'})
    return df
# Clean dataframes by removing empty spaces
served_df = clean_dataframe(served_df)
in_care_df = clean_dataframe(in_care_df)
entered_df = clean_dataframe(entered_df)
waiting_df = clean_dataframe(waiting_df)
terminated_df = clean_dataframe(terminated_df)
adopted_df = clean_dataframe(adopted_df)

#set the columns for each data frame
def set_years(df):
    years = range(2013, 2023)
    df.columns = ['States'] + list(years)
# Set columns as years for all dataframes
set_years(served_df)
set_years(in_care_df)
set_years(entered_df)
set_years(waiting_df)
set_years(terminated_df)
set_years(adopted_df)

# Reformat first Row
def reformat_row(df):
    for i in range(1,11):
        year_str = df.iloc[0, i]  # Accessing the first column
        year = int(year_str.split()[-1])  # Extracting the year and converting to integer
        df.iloc[0, i] = year
reformat_row(served_df)
reformat_row(in_care_df)
reformat_row(entered_df)
reformat_row(waiting_df)
reformat_row(terminated_df)
reformat_row(adopted_df)
print(served_df.head(5))
print(served_df.columns)