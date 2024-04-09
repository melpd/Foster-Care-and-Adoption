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
print(adopted_df.head(20))

#TO DO:
#- Data Cleaning
#   -Remove top section with titles
# remove the first 5 rows
served_df = served_df.iloc[6:,]
served_df.reset_index(drop=True, inplace=False)
print(served_df.head(20))
print(served_df.iloc[8,1])
#   -Convert years from string to integer