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

#- Data Cleaning

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

served_df = served_df.drop(served_df.index[-1])
in_care_df = in_care_df.drop(in_care_df.index[-1])
entered_df = entered_df.drop(entered_df.index[-1])
waiting_df = waiting_df.drop(waiting_df.index[-1])
terminated_df = terminated_df.drop(terminated_df[-1])
adopted_df = adopted_df.drop(adopted_df.index[-1])
print(served_df.columns)
print(served_df.dtypes)
print(served_df.head(5))
print(served_df.tail(10))
