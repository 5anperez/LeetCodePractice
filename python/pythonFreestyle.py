# import pandas as pd
# import matplotlib.pyplot as plt
# import seaborn as sns

# # Load the CSV file
# file_path = 'Bold21_Data_set-orders_export_1V2-Bold21_data_set-orders_export_1-Marcelo_Leal_Martinez.csv'  # Replace with your file path
# data = pd.read_csv(file_path)

# # Convert 'Created at' to datetime with the specific format
# data['Year'] = pd.to_datetime(data['Created at'], format='%Y-%m-%d %H:%M:%S %z', errors='coerce', utc=True).dt.year

# # Drop rows where year could not be extracted
# data = data.dropna(subset=['Year'])

# # Convert year to integer
# data['Year'] = data['Year'].astype(int)

# # # TESTING: I want to see this new object w/ columns as indeces (rows)
# # temp = data.groupby(['Year', 'Lineitem name']).sum()
# # print(temp)

# # # TESTING: Convert back into regular enum (ie 0, 1, ..., n) indeces (rows)
# # temp = data.groupby(['Year', 'Lineitem name']).sum().reset_index()

# # Grouping by year and product, and counting occurrences: First, the DataFrame is grouped by the 
# # 'Year' and 'Lineitem name' columns. Second, the size (count) of each group is calculated. 
# # Third, the index is reset to turn the grouped data back into a regular DataFrame and rename the count column.
# product_counts_by_year = data.groupby(['Year', 'Lineitem name']).size().reset_index(name='Count')

# # Finding the most sold product for each year
# most_sold_products = product_counts_by_year.sort_values(['Year', 'Count'], ascending=[True, False]).drop_duplicates('Year')

# print(most_sold_products)









# # Load the CSV file
# file_path = 'sales_memos.csv'
# data = pd.read_csv(file_path)

# # Counting occurrences of each company
# collaboration_counts = data['Company'].value_counts()

# # Finding the company with the most collaborations
# most_collaborated_company = collaboration_counts.idxmax()
# most_collaborations = collaboration_counts.max()

# # Calculating the percentage above others relative to total collaborations
# total_collaborations = collaboration_counts.sum()
# percentage_above_others = ((most_collaborations - collaboration_counts.drop(most_collaborated_company).max()) / total_collaborations) * 100

# print(most_collaborated_company)
# print(most_collaborations)
# print(percentage_above_others)




# # Load the CSV file
# file_path = 'sales_memos.csv'
# data = pd.read_csv(file_path)

# # Filter for accepted collaborations
# data_accepted = data[data['Status'] == 'ACCEPTED']
# data_total = data['Company'].value_counts()
# print(data.shape[0])
# print(data_total.sum())


# # Counting occurrences of each company in accepted collaborations
# # Since company names make up the column, the names will 
# # now be the row indeces and the num of collabs will be the actual rows
# collaboration_counts = data_accepted['Company'].value_counts()
# print(collaboration_counts.sum())

# # TESTING: Check out the new object ie the new df here
# # print(collaboration_counts)


# # Finding the company with the most collaborations and 
# # get its name and number of collabs
# most_collaborated_company = collaboration_counts.idxmax()
# most_collaborations = collaboration_counts.max()

# # Calculating the percentage above others
# second_most_collaborations = collaboration_counts.drop(most_collaborated_company).max()
# percentage_above_others = ((most_collaborations - second_most_collaborations) / second_most_collaborations) * 100

# print(f"Most Collaborated Company: {most_collaborated_company}")
# print(f"Number of Collaborations: {most_collaborations}")
# print(f"Percentage Above Others: {percentage_above_others:.2f}%")











# # Load the CSV file
# file_path = 'caja-dia-a-dia-no-Pii.csv'
# data = pd.read_csv(file_path)

# debit_amount_column = 'Monto DEBE'
# transfer_type_column = 'Tipo Comp DEBE'

# # Filtering out rows without debit amount
# debit_data = data[~data[debit_amount_column].isna()]

# # Calculating the average debit amount
# average_debit_amount = debit_data[debit_amount_column].mean()

# print("AVG:", average_debit_amount)

# # Filtering records with debit amount above the average
# above_average_debits = debit_data[debit_data[debit_amount_column] > average_debit_amount]

# # Extracting the records by transfer type
# records_by_transfer_type = above_average_debits.groupby(transfer_type_column).size().reset_index(name='Count')

# print(records_by_transfer_type)




















# # Read the CSV file with the specified encoding
# df = pd.read_csv("hotel_room.csv", encoding="ISO-8859-1")

# # Display the first 5 rows of the DataFrame
# print(df.head())

# # Display the data types of each column
# print(df.info())



# # Prepare data for pie chart
# labels = category_summary['Category']
# sizes = category_summary['Percentage']
# mean_review_counts = category_summary['Mean_Review_Count'].round(2)
# min_review_counts = category_summary['Min_Review_Count']
# max_review_counts = category_summary['Max_Review_Count']

# # Create pie chart
# plt.figure(figsize=(10, 8))
# plt.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=140)
# plt.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.

# # Add legend with mean, min, and max review counts
# legend_labels = [f'{label} - Mean: {mean}, Min: {min_count}, Max: {max_count}' for label, mean, min_count, max_count in zip(labels, mean_review_counts, min_review_counts, max_review_counts)]
# plt.legend(legend_labels, title="Review Counts", loc="best")

# plt.title("Distribution of Room Prices by Category")
# plt.show()









'''
CLEAN UP OF QUOTES AND COMMAS HERE
'''


# # Read the CSV file with the specified encoding
# filename = 'hotel_room.csv'
# data = pd.read_csv(filename, encoding='ISO-8859-1')

# # Filter out non-numeric rows from 'Room Price'
# data['Room Price (in BDT or any other currency)'] = data['Room Price (in BDT or any other currency)'].str.replace('\"', '').str.replace(',', '')
# data['Room Price (in BDT or any other currency)'] = pd.to_numeric(data['Room Price (in BDT or any other currency)'], errors='coerce')
# data = data.dropna(subset=['Room Price (in BDT or any other currency)'])

# # Filter out non-numeric rows from 'Room Price'
# data['review_count'] = data['review_count'].str.replace('\"', '').str.replace(',', '')
# data['review_count'] = pd.to_numeric(data['review_count'], errors='coerce')
# data = data.dropna(subset=['review_count'])
# print(data['review_count'])
# print(data['Room Price (in BDT or any other currency)'])

# # Calculate mean and standard deviation of Room Prices
# mean_price = data['Room Price (in BDT or any other currency)'].mean()
# std_price = data['Room Price (in BDT or any other currency)'].std()

# # Define the categories
# category1 = data[data['Room Price (in BDT or any other currency)'] <= mean_price - std_price]
# category2 = data[(data['Room Price (in BDT or any other currency)'] > mean_price - std_price) & (data['Room Price (in BDT or any other currency)'] <= mean_price + std_price)]
# category3 = data[data['Room Price (in BDT or any other currency)'] > mean_price + std_price]

# print(category2)

# #Calculate mean, max, and min Review Count for each category
# mean_review1 = category1['review_count'].mean()
# max_review1 = category1['review_count'].max()
# min_review1 = category1['review_count'].min()

# mean_review2 = category2['review_count'].mean()
# max_review2 = category2['review_count'].max()
# min_review2 = category2['review_count'].min()

# mean_review3 = category3['review_count'].mean()
# max_review3 = category3['review_count'].max()
# min_review3 = category3['review_count'].min()

# # Create a pie chart
# # Data for pie chart
# sizes = [len(category1), len(category2), len(category3)]
# labels = [f'Category 1: Mean Review {mean_review1}, Max {max_review1}, Min {min_review1}', 
#           f'Category 2: Mean Review {mean_review2}, Max {max_review2}, Min {min_review2}', 
#           f'Category 3: Mean Review {mean_review3}, Max {max_review3}, Min {min_review3}']

# # Plotting the pie chart
# plt.figure(figsize=(10, 6))
# plt.pie(sizes, labels=labels, autopct='%1.1f%%')
# plt.title('Distribution of Room Prices')
# plt.show()






'''
FUNCTION TO CLEAN UP QUOTES AND COMMAS HERE
'''

# # Read the CSV file with the specified encoding
# filename = './CSVs/hotel_room.csv'
# room_price = 'Room Price (in BDT or any other currency)'
# review_count = 'review_count'

# # Load the dataset
# data = pd.read_csv(filename, encoding='ISO-8859-1')

# # Function to clean and convert column values
# def clean_and_convert(column):
#     # Remove quotes and replace commas
#     data[column] = data[column].str.replace('"', '').str.replace(',', '')
#     # Convert to numeric, handling non-numeric values
#     data[column] = pd.to_numeric(data[column], errors='coerce')

# # Clean 'Room Price' and 'review_count' columns
# clean_and_convert(room_price)
# clean_and_convert(review_count)

# # Get all properties (rows) w/ more than 1000 reviews
# fd = data[data[review_count] > 1000]

# # Get the average room price out of these in fd
# avg = fd[room_price].mean

# # Display the cleaned data
# print(avg)





# # Read the CSV file into a Pandas DataFrame
# df = pd.read_csv("./CSVs/simulated_customers.csv")

# # Create a histogram of the "age" column in the DataFrame
# df['age'].hist()

# # Display the histogram
# plt.show()


# # Create a histogram of the age distribution
# plt.figure(figsize=(10, 6))
# plt.hist(df['age'].dropna(), bins=10, color='skyblue', edgecolor='black')
# plt.title('Age Distribution of Simulated Customers')
# plt.xlabel('Age')
# plt.ylabel('Frequency')
# plt.grid(axis='y', alpha=0.75)

# # Show the histogram
# plt.show()






# Load the dataset
# salaries_data = pd.read_csv('./CSVs/ds_salaries.csv')

# # Analyzing how salaries change across professions and locations
# salary_by_profession_location = salaries_data.groupby(['job_title', 'company_location'])['salary_in_usd'].describe()

# # Finding the location that pays the most for ML Engineers
# ml_engineer_salaries = salaries_data[salaries_data['job_title'] == 'Machine Learning Engineer']
# highest_paying_location = ml_engineer_salaries.groupby('company_location')['salary_in_usd'].mean().idxmax()

# print(salary_by_profession_location)
# print("Location that pays the most for ML Engineers:", highest_paying_location)



# # Filter for ML Engineer positions
# ml_engineer_salaries = salaries_data[salaries_data['job_title'] == 'Machine Learning Engineer']

# # Group by location and calculate the average salary
# average_salary_by_location = ml_engineer_salaries.groupby('company_location')['salary_in_usd'].mean()

# # Convert the GroupBy object to a DataFrame for better readability
# average_salary_df = average_salary_by_location.reset_index()

# print(average_salary_df)




















# # Read the CSV file into a Pandas DataFrame
# df = pd.read_csv("EMERGENCIA_2023_YAKU.xlsx_M.P.H-MATUCANA-Sheet1.csv")

# # Find the row with the maximum value in the GALONES column
# max_row = df.loc[df['GALONES'].astype(float).idxmax()]

# # Extract the plate and gallons from the row
# max_plate = max_row['PLACA']
# max_gallons = max_row['GALONES']

# # Print the plate and gallons
# print(f"The plate with the most gallons is {max_plate} with {max_gallons} gallons.")


'''
NaN clean up here
'''


# # Read the CSV file into a Pandas DataFrame
# data = pd.read_csv("EMERGENCIA_2023_YAKU.xlsx_M.P.H-MATUCANA-Sheet1.csv")

# # Convert the 'GALONES' column to numeric after cleaning up commas
# data['GALONES'] = data['GALONES'].str.replace(',', '').astype(float)

# # Find the plate with the maximum gallons
# max_gallons_plate = data.loc[data['GALONES'].idxmax(), 'PLACA']

# # Investigate it: this revealed NaN
# print(max_gallons_plate)

# # Investigate the issue by checking for any NaN values in the 'PLACA' and 'GALONES' columns
# # These two lines count the number of NaN rows. They revealed 22 placa nans and 21 galones nans
# nan_in_placa = data['PLACA'].isna().sum()
# nan_in_gallons = data['GALONES'].isna().sum()

# # Now I know that both are riddled with nans
# print(nan_in_placa, nan_in_gallons, data['GALONES'].max())

# # Filter out rows with NaN in 'PLACA' 
# data = data.dropna(subset=['PLACA'])

# # Now find the plate with the maximum gallons in the filtered data
# max_gallons_plate_filtered = data.loc[data['GALONES'].idxmax(), 'PLACA']
# max_gallons = data['GALONES'].max()  
# print(f"{max_gallons_plate_filtered}, {max_gallons:,}")


















# # Read the CSV file into a Pandas DataFrame
# df = pd.read_csv("Bold21_Data_set-orders_export_1V2-Bold21_data_set-orders_export_1-Marcelo_Leal_Martinez.csv")

# # Count the number of times each distinct value appears in the 'Name' column
# name_counts = df['Name'].value_counts()

# # Sort the counts in descending order
# sorted_name_counts = name_counts.sort_values(ascending=False)

# # Display the counts
# print(sorted_name_counts)


# fulfilled = sorted_name_counts['Fulfillment Status'] == 'fulfilled'
# fulfilledCount = fulfilled.value_counts()

# print(fulfilledCount)



# # Read the CSV file into a Pandas DataFrame
# df = pd.read_csv("Bold21_Data_set-orders_export_1V2-Bold21_data_set-orders_export_1-Marcelo_Leal_Martinez.csv")

# # Filter for rows where 'Fulfillment Status' is 'fulfilled'
# fulfilled_df = df[df['Fulfillment Status'] == 'fulfilled']

# # Count the number of times each distinct value appears in the 'Name' column of the filtered DataFrame
# name_counts = fulfilled_df['Name'].value_counts()

# # Sort the counts in descending order
# sorted_name_counts = name_counts.sort_values(ascending=False)

# # Display the counts
# print(sorted_name_counts)

















# def convert_to_float(value):
#     """
#     Converts a string with a comma as the decimal separator to a float.
    
#     Parameters:
#     value (str): The string to be converted.

#     Returns:
#     float: The converted floating-point number.
#     """
#     if pd.isna(value):
#         return value
#     return float(value.replace(',', '.'))

# # Load the dataset
# df = pd.read_csv('./CSVs/current_accounts.csv')


# # Convert 'Debit' and 'Credit' columns to numeric, setting errors='coerce' to handle non-numeric values
# # This will replace non-numeric values with NaN, which we can then handle appropriately
# # df['Debit'] = pd.to_numeric(df['Debit'], errors='coerce')
# # df['Credit'] = pd.to_numeric(df['Credit'], errors='coerce')

# df['Debit'] = df['Debit'].apply(convert_to_float)
# df['Credit'] = df['Credit'].apply(convert_to_float)

# # Check for missing values that may have been introduced during the conversion
# print('Missing values in Debit:', df['Debit'].isnull().sum())
# print('Missing values in Credit:', df['Credit'].isnull().sum())

# # Now, let's group by 'Status' and calculate the sum of 'Debit' and 'Credit' for each status
# grouped = df.groupby('Status')[['Debit', 'Credit']].sum().reset_index()

# # Display the grouped data to examine the relationship
# print(grouped)






# # Load the dataset
# df = pd.read_csv('./CSVs/current_accounts.csv')

# top_breaks = df.sort_values(by='duration_column', ascending=False).head(15)












# import pandas as pd

# # Load the dataset
# file_path = './CSVs/Mantenimiento_Gruas-PARTE.csv'
# data = pd.read_csv(file_path)

# # Convert the start and end times to datetime objects
# data['Hora del INICIO del trabajo'] = pd.to_datetime(data['Hora del INICIO del trabajo'], format='%I:%M:%S %p', errors='coerce')
# data['Hora de FINALIZACIÓN del trabajo'] = pd.to_datetime(data['Hora de FINALIZACIÓN del trabajo'], format='%I:%M:%S %p', errors='coerce')

# # Calculate the duration of work
# data['Duracion de trabajo'] = (data['Hora de FINALIZACIÓN del trabajo'] - data['Hora del INICIO del trabajo']).dt.total_seconds() / 3600  # Duration in hours

# # Sort the data by duration and select the top 15
# top_15 = data.sort_values(by='Duracion de trabajo', ascending=False).head(15)

# # Display the top 15 durations and their respective 'Causa de la falla'
# print(top_15[['Duracion de trabajo', 'Causa de la falla']])




'''
    THIS ONE FINDS CERTAIN ROWS OF INTEREST AND SUMS THEM TOGETHER
    WHICH IS SOMETHING NEW TO ME AND USEFUL!!
'''




# # Load the nz.csv data and inspect the first few rows to understand the structure and content.
# df_nz = pd.read_csv('./CSVs/nz.csv')

# # Check the data types and the head of the dataframe to ensure proper loading and to get an overview.
# # df_nz.info()
# # print(df_nz.head())


# # Convert the 'value' column to numeric, coercing any errors to NaN (which will then be handled).
# df_nz['value'] = pd.to_numeric(df_nz['value'], errors='coerce')

# # Now let's retry the pivot and calculation of profit margins.
# # Filter the data for 'Total income' and 'Total expenditure'
# df_profit = df_nz[df_nz['variable'].isin(['Total income', 'Total expenditure'])]

# # Pivot the table to have separate columns for income and expenditure
# df_pivot = df_profit.pivot_table(index=['year', 'industry_name_ANZSIC'], columns='variable', values='value', aggfunc='sum').reset_index()

# # print(df_pivot)

# # Calculate profit margin as (Total income - Total expenditure) / Total income
# df_pivot['profit_margin'] = (df_pivot['Total income'] - df_pivot['Total expenditure']) / df_pivot['Total income']

# # print(df_pivot['profit_margin'])

# # Calculate the average profit margin for each industry over the years
# df_avg_profit_margin = df_pivot.groupby('industry_name_ANZSIC')['profit_margin'].mean().reset_index()



# # Sort the industries by average profit margin
# df_avg_profit_margin_sorted = df_avg_profit_margin.sort_values(by='profit_margin', ascending=False)

# # Display the sorted average profit margins for each industry
# #print(df_avg_profit_margin_sorted)











'''
    USE THE SEABORN LIBRARY TO CREATE A HEATMAP!
'''



'''
    VERSION 1: Julius
'''

# # Import the necessary library for plotting
# import seaborn as sns


# # Load the simulated_customers.csv data and inspect the first few rows to understand the structure and content.
# df_customers = pd.read_csv('./CSVs/simulated_customers.csv')

# # Check the data types and the head of the dataframe to ensure proper loading and to get an overview.
# df_customers.info()
# print(df_customers.head())


# # Handle NaN values in 'relationship' and 'children' columns before converting to integers
# df_customers['relationship'] = df_customers['relationship'].fillna(0).astype(int)
# df_customers['children'] = df_customers['children'].fillna(0).astype(int)


# '''
#     FIND OUT WHAT THIS CODE BLOCK DOES EXACTLY!
# '''


# # Define the income classes
# bins = [i for i in range(0, int(df_customers['income'].max() + 2500), 2500)]
# labels = [f'{i}-{i + 2499}' for i in range(0, int(df_customers['income'].max()), 2500)]
# df_customers['income classes'] = pd.cut(df_customers['income'], bins=bins, labels=labels, right=False)

# # Encode 'gender' column as numeric
# df_customers['gender'] = df_customers['gender'].map({'M': 1, 'W': 0})

# # Calculate the correlation matrix for the specified columns
# correlation_matrix = df_customers[['relationship', 'gender', 'children', 'income classes']].corr()

# # Plot the heatmap
# plt.figure(figsize=(15, 15))
# sns.heatmap(correlation_matrix, annot=True, fmt='.2f', cmap='coolwarm', square=True)
# plt.title('Correlation Heatmap of Customers')
# plt.tight_layout()
# plt.show()



'''
    VERSION 2: DA - model B
'''

# # Load the data
# file_path = './CSVs/simulated_customers.csv'
# df = pd.read_csv(file_path)

# # Function to assign income classes
# def assign_income_class(income):
#     return (income // 2500) * 2500

# # Add a new column 'income classes' based on the income values
# df['income classes'] = df['income'].apply(assign_income_class)

# # Replace NaN values with a default value for 'relationship' and 'children' columns
# df['relationship'] = df['relationship'].fillna(False).astype(int)
# df['children'] = df['children'].fillna(False).astype(int)

# # THIS (gender map) WAS FROM VERSION 1 AND IT RESULTED IN THIS 
# # CODE RUNNING GOOD, BUT I DID NOT CONFIRM ITS CORRECT!

# # Encode 'gender' column as numeric
# df['gender'] = df['gender'].map({'M': 1, 'W': 0})

# # Calculate the correlation matrix
# corr_matrix = df[['relationship', 'gender', 'children', 'income classes']].corr()

# # Plotting the heatmap
# plt.figure(figsize=(15, 15))
# sns.heatmap(corr_matrix, annot=True, cmap='coolwarm')
# plt.title('Correlation Heatmap of Customers')
# plt.tight_layout()
# plt.show()




'''
    VERSION 3: GPT
'''

# # Load the data
# customers_data = pd.read_csv('path_to_simulated_customers.csv')

# # Define a function to classify income
# def classify_income(income):
#     return (income // 2500) * 2500

# # Apply the function to create a new 'income classes' column
# customers_data['income classes'] = customers_data['income'].apply(classify_income)

# # Convert categorical columns to numerical for correlation analysis
# customers_data['gender'] = customers_data['gender'].astype('category').cat.codes
# customers_data['relationship'] = customers_data['relationship'].astype('category').cat.codes

# # Compute the correlation matrix
# corr_matrix = customers_data[['relationship', 'gender', 'children', 'income classes']].corr()

# # Generate a heatmap
# plt.figure(figsize=(15, 15))
# sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap='coolwarm')
# plt.title('Correlation Heatmap of Customers')
# plt.xticks(rotation=45)
# plt.yticks(rotation=45)
# plt.show()










'''
    CLEAN UP MONEY ENTRIES
'''



# # Load the data
# file_path = './CSVs/Bancos-Bodega.xlsxcsv-Global.csv'  # Update the file path as needed
# data_exchange = pd.read_csv(file_path)

# # Cleaning 'Monto' column
# data_exchange['Monto'] = pd.to_numeric(
#     data_exchange['Monto'].replace({'\$': '', ',': ''}, regex=True).str.strip(),
#     errors='coerce'
# )

# # Cleaning 'Monto en USD' column
# data_exchange['Monto en USD'] = pd.to_numeric(
#     data_exchange['Monto en USD'].replace({'-': '', 'USD': '', ',': ''}, regex=True).str.strip(),
#     errors='coerce'
# )

# # Define the threshold for higher exchange rates (e.g., above the 75th percentile)
# exchange_rate_threshold = data_exchange['Tipo de cambio'].quantile(0.75)
# #print(exchange_rate_threshold)

# # Filter data for higher exchange rates
# high_exchange_rate_data = data_exchange[data_exchange['Tipo de cambio'] > exchange_rate_threshold]
# high_exchange_rate_data2 = data_exchange[data_exchange['Tipo de cambio'] == 359.00]

# # Calculate the total amount made during periods of higher exchange rates
# total_amount_high_exchange_rate = high_exchange_rate_data['Monto'].sum()
# total_amount_high_exchange_rate2 = high_exchange_rate_data2['Monto'].sum()

# total_amount_high_exchange_rate3 = high_exchange_rate_data['Monto en USD'].sum()
# total_amount_high_exchange_rate4 = high_exchange_rate_data2['Monto en USD'].sum()

# print("Total amount made during periods of higher exchange rates: ${:,.2f}".format(total_amount_high_exchange_rate))
# print("Total amount made during periods of higher exchange rates: ${:,.2f}".format(total_amount_high_exchange_rate2))

# print("Total amount made during periods of higher exchange rates in USD: ${:,.2f}".format(total_amount_high_exchange_rate3))
# print("Total amount made during periods of higher exchange rates in USD: ${:,.2f}".format(total_amount_high_exchange_rate4))












# Load the dataset
# file_path = './CSVs/Cleaned_Laptop_data.csv'
# laptop_data = pd.read_csv(file_path)

# # Grouping by processor_gnrtn and brand, and calculating the sum of ratings
# grouped_data = laptop_data.groupby(['processor_gnrtn', 'processor_brand'])['ratings'].sum().reset_index()

# # Finding the brand with the highest ratings sum for each processor generation
# highest_ratings = grouped_data.loc[grouped_data.groupby('processor_gnrtn')['ratings'].idxmax()]

# # Sorting by processor generation for clarity
# highest_ratings_sorted = highest_ratings.sort_values(by='processor_gnrtn')

# print(highest_ratings_sorted)


# Another version

# laptop_data = pd.read_csv(file_path)

# # Grouping the data by processor_brand and processor_gnrtn, and summing the ratings
# brand_ratings = laptop_data.groupby(['processor_brand', 'processor_gnrtn'])['ratings'].sum().reset_index()

# # Finding the brand with the highest ratings sum for each processor generation
# max_brand_ratings = brand_ratings.loc[brand_ratings.groupby('processor_gnrtn')['ratings'].idxmax()]

# # Preparing the table for display
# table = max_brand_ratings[['processor_brand', 'processor_gnrtn', 'ratings']]
# table.rename(columns={'ratings': 'total_ratings_sum'}, inplace=True)
# table.reset_index(drop=True, inplace=True)
# print(table)








'''
    USE THE TIME COLUMN TO CREATE 3 NEW COLUMNS, I.E., USE THE TIME
    COLUMN TO GROUP EVENTS BY MORNING, AFTERNOON, AND EVENING.
'''



# # Load the dataset
# file_path = './CSVs/Legacy_Baton_Rouge_Traffic_Incidents.csv'
# df = pd.read_csv(file_path)

# # Convert 'CRASH TIME' to datetime and extract the hour
# # Assuming 'CRASH TIME' is in a format that includes AM/PM
# df['CRASH TIME'] = pd.to_datetime(df['CRASH TIME'], format='%I:%M %p', errors='coerce')
# df['Hour'] = df['CRASH TIME'].dt.hour

# # Define time ranges for morning (5am-12pm), afternoon (12pm-5pm), and evening (5pm-10pm)
# morning = df[(df['Hour'] >= 6) & (df['Hour'] < 12)]
# afternoon = df[(df['Hour'] >= 12) & (df['Hour'] < 18)]
# evening = df[(df['Hour'] >= 18) & (df['Hour'] < 22)]

# # Filter for pedestrian incidents
# morning_pedestrian = morning[morning['PEDESTRIAN'] == 'X']
# afternoon_pedestrian = afternoon[afternoon['PEDESTRIAN'] == 'X']
# evening_pedestrian = evening[evening['PEDESTRIAN'] == 'X']

# # Group by district and count incidents
# morning_district = morning_pedestrian['DISTRICT'].value_counts()
# afternoon_district = afternoon_pedestrian['DISTRICT'].value_counts()
# evening_district = evening_pedestrian['DISTRICT'].value_counts()

# # Plot pie charts
# fig, axs = plt.subplots(1, 3, figsize=(18, 6))

# axs[0].pie(morning_district, labels=morning_district.index, autopct='%1.1f%%', startangle=90)
# axs[0].set_title('Morning Pedestrian Incidents by District')

# axs[1].pie(afternoon_district, labels=afternoon_district.index, autopct='%1.1f%%', startangle=90)
# axs[1].set_title('Afternoon Pedestrian Incidents by District')

# axs[2].pie(evening_district, labels=evening_district.index, autopct='%1.1f%%', startangle=90)
# axs[2].set_title('Evening Pedestrian Incidents by District')

# plt.tight_layout()
# plt.show()
# print('Pie charts generated.')









'''
    FILTERING TWEETS WITH URLs HERE!
'''


# # Load the dataset
# ukraine_df = pd.read_csv('./CSVs/ukraine_conflict.csv')

# # Convert 'time' to datetime and extract the hour
# ukraine_df['hour'] = pd.to_datetime(ukraine_df['time'], format='%H:%M:%S', errors='coerce').dt.hour

# # Drop rows where 'hour' is NaN
# ukraine_df = ukraine_df.dropna(subset=['hour'])

# # Filter tweets that contain URLs
# url_tweets = ukraine_df[ukraine_df['urls'].astype(str) != '[]']

# # Group by hour and count tweets
# hourly_tweet_counts = url_tweets.groupby('hour').size()

# # Find the hour with the most tweets
# peak_hour = hourly_tweet_counts.idxmax()
# peak_count = hourly_tweet_counts.max()

# # Plot a bar chart of tweet counts by hour
# plt.figure(figsize=(12, 6))
# plt.bar(hourly_tweet_counts.index, hourly_tweet_counts.values)
# plt.title('Tweet Counts by Hour with URLs')
# plt.xlabel('Hour of the Day')
# plt.ylabel('Number of Tweets')
# plt.axvline(x=peak_hour, color='red', linestyle='--', label='Peak Hour')
# plt.legend()
# plt.grid(True)
# plt.show()

# print('Peak hour for tweets with URLs: ' + str(peak_hour) + ' with ' + str(peak_count) + ' tweets.')








'''
    VERY WEIRD DATA SET HERE! I NEED A CLEVER WAY OF DOING THIS ONE!
'''

sales_df = pd.read_csv('./CSVs/Sales-PrintingOffice2023v1-Sheet1.csv')

# Assuming the 'Number' column contains the month names and 'TOTAL',
# we need to find the index of the 'TOTAL' row for October and November

# # Find the index of the row that contains 'OCTOBER' and 'NOVEMBER'
# october_index = sales_df[sales_df['Number'].str.contains('OCTOBER', na=False)].index.max()
# november_index = sales_df[sales_df['Number'].str.contains('NOVEMBER', na=False)].index.max()

# # Find the 'TOTAL' row for October and November
# # The 'TOTAL' row should be the last row of the month's data
# october_total_row = sales_df.iloc[october_index+1]
# november_total_row = sales_df.iloc[november_index+1]

# # Extract the 'Final Price' and convert it to float after removing any non-numeric characters
# october_total = float(october_total_row['Final Price'].replace('$', '').replace(',', ''))
# november_total = float(november_total_row['Final Price'].replace('$', '').replace(',', ''))

# # Calculate the combined total sales for October and November
# total_sales_oct_nov = october_total + november_total

# print('Total sales for October:', october_total)
# print('Total sales for November:', november_total)
# print('Combined total sales for October and November:', total_sales_oct_nov)




# # Find the index of the row that contains 'OCTOBER' and 'NOVEMBER'
# october_index = sales_df[sales_df['Number'].str.contains('OCTOBER', na=False)].index.max()
# november_index = sales_df[sales_df['Number'].str.contains('NOVEMBER', na=False)].index.max()
# print(october_index)
# print(november_index)

# # Assuming the 'TOTAL' row is the last row of each month's block, we can find the start of the next month to define the block
# # The start of November's block will be the end of October's block
# start_of_november = november_index
# end_of_october = start_of_november - 1

# # Sum up the 'Final Price' for October
# # We need to convert the 'Final Price' to a numeric type, ignoring non-numeric issues with coerce
# october_final_prices = pd.to_numeric(sales_df.loc[october_index+1:end_of_october, 'Final Price'].str.replace('[^\d.]', '', regex=True), errors='coerce')
# october_sum = october_final_prices.sum()

# # For November, we need to find the end of the block, which we assume is before the 'TOTAL' row for November
# # We will find the 'TOTAL' row index for November
# november_total_index = sales_df[sales_df['Number'].str.contains('TOTAL', na=False) & (sales_df.index > november_index)].index.min()
# end_of_november = november_total_index - 1

# # Sum up the 'Final Price' for November
# november_final_prices = pd.to_numeric(sales_df.loc[november_index+1:end_of_november, 'Final Price'].str.replace('[^\d.]', '', regex=True), errors='coerce')
# november_sum = november_final_prices.sum()

# # Output the results
# print('Total sum of Final Price for October:', october_sum)
# print('Total sum of Final Price for November:', november_sum)


'''HERE I HARDCODE THE RANGE TO CLAMP THE TRAVERSAL'''

# sales_df = pd.read_csv('./CSVs/Sales-PrintingOffice2023v1-Sheet1.csv')

# # Hard code the start and end indices for October and November
# start_of_november = 4
# end_of_november = 18
# start_of_october = 21
# end_of_october = 30

# # Clean up the 'Final Price' column by removing quotes, dollar signs, and commas
# sales_df['Final Price Cleaned'] = pd.to_numeric(
#     sales_df['Final Price']
#     .str.replace('"', '')
#     .str.replace('$', '')
#     .str.replace(',', ''),
#     errors='coerce'
# )

# # Sum up the 'Final Price' for October and November
# october_sum = sales_df.loc[start_of_october:end_of_october, 'Final Price Cleaned'].sum()
# november_sum = sales_df.loc[start_of_november:end_of_november, 'Final Price Cleaned'].sum()

# # Output the results
# print('Total sum of Final Price for October:', october_sum)
# print('Total sum of Final Price for November:', november_sum)





















# # Load the file to check its structure
# file_path = './CSVs/caja-dia-a-dia-no-Pii.csv'
# data = pd.read_csv(file_path)

# # Display the first few rows of the file to understand its structure
# # data.head()

# # Convert the 'Fecha' column to datetime and filter for the year 2021
# data['Fecha'] = pd.to_datetime(data['Fecha'])
# data_2021 = data[data['Fecha'].dt.year == 2021]

# # Fill missing values in 'Monto HABER' and 'Monto DEBE' with 0
# data_2021['Monto HABER'] = data_2021['Monto HABER'].fillna(0)
# data_2021['Monto DEBE'] = data_2021['Monto DEBE'].fillna(0)

# # Calculate net amount for each row (Monto HABER - Monto DEBE)
# data_2021['Net Amount'] = data_2021['Monto HABER'] - data_2021['Monto DEBE']

# # Group by month and sum the net amounts
# monthly_accumulation = data_2021.groupby(data_2021['Fecha'].dt.to_period('M'))['Net Amount'].sum()

# print(monthly_accumulation)
















# import pandas as pd 
# import matplotlib.pyplot as plt 
# df = pd.read_csv("Cities with the Best Work-Life Balance 2022.csv") 
# grouped = df.groupby("Country")[["Covid Impact", "Covid Support"]].mean() 
# grouped["Ratio"] = grouped["Covid Impact"] / grouped["Covid Support"] 
# grouped["Ratio"].plot(kind="bar", figsize=(10, 6)) 
# plt.ylabel("Covid Impact to Support Ratio") 
# plt.title("Covid Impact to Support Ratio by Country") 
# plt.tight_layout() 
# plt.show()














# Re-importing pandas after a reset
# import pandas as pd

# # Load the data from the newly uploaded file
# file_path_chardonnay_sales = './CSVs/Ventas_primer_semestre-wines.xlsxcsv-Hoja1.csv'

# # Attempting to read the CSV file again after the reset
# try:
#     chardonnay_sales_data = pd.read_csv(file_path_chardonnay_sales)
    
#     # Display the first few rows to understand its structure
#     display = chardonnay_sales_data.head()
# except Exception as e:
#     display = str(e)

# # # display


# # # Filtering for Chardonnay sales
# chardonnay_sales = chardonnay_sales_data[chardonnay_sales_data['Varietal'].str.contains("Chardonnay", case=False)]

# # print(chardonnay_sales)

# # Grouping by month and counting the number of sales (assuming each row represents a sale)
# sales_by_month = chardonnay_sales['Comp. - Año / Mes Contab. (AAAA/MM)'].value_counts().reset_index()
# sales_by_month.columns = ['Month', 'Sales Count']

# # Finding the month with the most Chardonnay sales
# max_sales_month = sales_by_month.loc[sales_by_month['Sales Count'].idxmax()]

# print(max_sales_month)














'''
    HERE, WE ARE SUMMING TOTAL SALES FOR A SPECIFIC YEAR (2022). SINCE THERE ARE BOTH CREDIT AND DEBIT SALES, WE HAVE TO CREATE A CONDITIONAL BASED ON WHAT THE ROW/COL CELL ENTRY IS. EVERY SALE IS EITHER A DEBIT OR CREDIT, SO WE HAVE AN OR CONDITION!
'''


# Load the data from the uploaded file
file_path_accounts = './CSVs/caja-dia-a-dia-no-Pii.csv'

# Reading the CSV file
accounts_data = pd.read_csv(file_path_accounts)

# Display the first few rows to understand its structure and identify the relevant columns
# accounts_data.head()

# Handling 'Monto HABER' and 'Monto DEBE' as specified
# Coalesce 'Monto HABER' with 'Monto DEBE', prioritizing 'Monto HABER' unless it's empty, zero, null, or NaN
# accounts_data['Effective Monto'] = accounts_data['Monto HABER'].fillna(0) + accounts_data['Monto DEBE'].fillna(0)
# accounts_data.loc[accounts_data['Monto HABER'].isnull() | (accounts_data['Monto HABER'] == 0), 'Effective Monto'] = accounts_data['Monto DEBE']

# # Filter the data for the year 2022
# data_2022_effective = accounts_data[accounts_data['Fecha'].dt.year == 2022]

# # Aggregating the effective amounts by account name
# effective_amounts_by_account_2022 = data_2022_effective.groupby('Nombre de la cuenta DEBE')['Effective Monto'].sum().reset_index()

# # Sorting to find the top 10 accounts with highest effective amounts
# top_10_effective_accounts_2022 = effective_amounts_by_account_2022.sort_values('Effective Monto', ascending=False).head(10)

# # Creating a bar chart for the top 10 accounts with highest effective amounts in 2022, after adjustment
# plt.figure(figsize=(10, 8))
# plt.barh(top_10_effective_accounts_2022['Nombre de la cuenta DEBE'], top_10_effective_accounts_2022['Effective Monto'], color='skyblue')
# plt.xlabel('Effective Total Amount')
# plt.ylabel('Account Name')
# plt.title('Top 10 Accounts by Highest Effective Amount in 2022')
# plt.gca().invert_yaxis()  # To display the highest amount at the top
# plt.show()







# import numpy as np

# # Correcting the approach with the right column names
# # Convert 'Fecha' to datetime to extract the year
# accounts_data['Fecha'] = pd.to_datetime(accounts_data['Fecha'])

# # Filter data for the year 2022
# data_2022 = accounts_data[accounts_data['Fecha'].dt.year == 2022]

# # Adjusting the approach to consider both credit and debit transactions
# # Use 'Monto HABER' if available, otherwise 'Monto DEBE'
# accounts_data['Monto'] = np.where(accounts_data['Monto HABER'].notna() & (accounts_data['Monto HABER'] != 0), accounts_data['Monto HABER'], accounts_data['Monto DEBE'])

# # Filter data for the year 2022 again, now with the adjusted amounts
# data_2022 = accounts_data[accounts_data['Fecha'].dt.year == 2022]

# # Sum amounts by account name and sort
# amounts_by_account_2022 = data_2022.groupby('Nombre de la cuenta DEBE')['Monto'].sum().sort_values(ascending=False).head(10)

# print(amounts_by_account_2022)

# Plot with labels and names
# plt.figure(figsize=(12, 8))
# amounts_by_account_2022.plot(kind='bar', color='skyblue')
# plt.title('Top 10 Accounts by Amount in 2022')
# plt.xlabel('Account Name')
# plt.ylabel('Amount')
# plt.xticks(rotation=45, ha='right')
# plt.tight_layout()
# plt.show()






# import pandas as pd

# # Read the CSV file into a Pandas DataFrame
# df = pd.read_csv('./CSVs/caja-dia-a-dia-no-Pii.csv')

# # Add a "Year" column extracted from the "Fecha" column using pd.to_datetime() and dt.year
# df["Year"] = pd.to_datetime(df["Fecha"]).dt.year

# # Filter the DataFrame to keep rows where "Year" is 2022
# df_filtered = df[df["Year"] == 2022]

# # print(df_filtered)

# # Group the filtered DataFrame by the "Nombre cuenta HABER" column
# grouped_df_c = df_filtered.groupby("Nombre cuenta HABER")
# grouped_df_d = df_filtered.groupby("Nombre de la cuenta DEBE")

# # Calculate the sum of the "Monto HABER" column for each group
# summed_df_credit = grouped_df_c["Monto HABER"].sum().reset_index()
# summed_df_debit = grouped_df_d["Monto DEBE"].sum().reset_index()

# merged_df = pd.merge(summed_df_credit, summed_df_debit, on="Nombre de la cuenta", how="outer")
# merged_df.fillna(0, inplace=True)
# print(merged_df)

# Sort the resulting DataFrame by the sum of "Monto HABER" in descending order
# sorted_df_c = summed_df_credit.sort_values(by="Monto HABER", ascending=False)
# sorted_df_d = summed_df_debit.sort_values(by="Monto DEBE", ascending=False)

# # Display the top 10 rows using `df.head(10)` to show the top 10 accounts and their total amounts
# top_10_df_c = sorted_df_c.head(10)
# top_10_df_d = sorted_df_d.head(10)

# print(top_10_df_c)
# print(top_10_df_d)














# import pandas as pd

# # Load the dataset
# df = pd.read_csv('./CSVs/ByrdBiteAdData.csv')
# data = pd.read_csv('./CSVs/ByrdBiteAdData.csv')

# Assuming there's a column named 'Campaign Source' that indicates the platform, including Instagram
# You'll need to replace 'Campaign Source' with the actual column name if it's different
# instagram_campaigns = df[df['Campaign name'] == 'Instagram'] # Adjust the condition based on actual data

# # Calculate the percentage
# percentage_instagram = (len(instagram_campaigns) / len(df)) * 100

# print(f"Percentage of Instagram campaigns: {percentage_instagram:.2f}%")




# Count the total number of campaigns
# total_campaigns = data['Campaign name'].nunique()

# print(total_campaigns)

# # Count the number of Instagram campaigns
# instagram_campaigns = data[data['Campaign name'].str.contains('Instagram', na=False)]['Campaign name'].nunique()

# # Calculate the percentage
# percentage_instagram_campaigns = (instagram_campaigns / total_campaigns) * 100 if total_campaigns > 0 else 0
# print(percentage_instagram_campaigns)




# Filter the DataFrame to include only rows where 'Campaign name' contains 'Instagram'
# instagram_rows = data[data['Campaign name'].str.contains('Instagram', case=False, na=False)]
# ig_rows = data[data["Ad Set Name"] == "Instagram Post"]

# # Print the filtered rows to see the variations
# # print(instagram_rows)
# print(ig_rows)
# print(len(data))


# Group by 'Campaign name' and count the ads
# campaign_ad_counts = data.groupby('Campaign name').size()

# Sort the counts
# campaign_ad_counts_sorted = campaign_ad_counts.sort_values(ascending=False)

# Display the sorted table
# print(campaign_ad_counts)


# ads_per_campaign = data['Campaign name'].value_counts()

# # Convert the series to a dataframe for better display
# ads_per_campaign_df = ads_per_campaign.reset_index()
# ads_per_campaign_df.columns = ['Campaign Name', 'Number of Ads']

# print(ads_per_campaign_df)












'''
    DIRTY CLEAN-UP HERE!
'''




# import pandas as pd

# # Load the data
# sales_data = pd.read_csv('./CSVs/sales_memos.csv')

# # Clean the "Commission" column
# sales_data['Commission Cleaned'] = sales_data['Commission'].str.replace('[^\d,]', '', regex=True).str.replace(',', '')

# # Convert the cleaned column to a numeric type
# sales_data['Commission Cleaned'] = pd.to_numeric(sales_data['Commission Cleaned'], errors='coerce')

# # Optionally, fill NaN values with 0 or another placeholder if needed
# # sales_data['Commission Cleaned'] = sales_data['Commission Cleaned'].fillna(0)

# # Verify the results
# print(sales_data[['Commission', 'Commission Cleaned']].head())

# # Assuming 'CustomerID' and 'SaleAmount' are the relevant columns
# # Summarize sales by customer
# sales_by_customer = sales_data.groupby('Customer')['Commission Cleaned'].sum().reset_index()

# # Sort customers by total sales in descending order
# sales_by_customer_sorted = sales_by_customer.sort_values('Commission Cleaned', ascending=False)

# # Calculate the cumulative sales
# sales_by_customer_sorted['Cumulative Sales'] = sales_by_customer_sorted['Commission Cleaned'].cumsum()

# # Calculate the total sales
# total_sales = sales_by_customer_sorted['Commission Cleaned'].sum()

# # Calculate the cumulative percentage of total sales
# sales_by_customer_sorted['Cumulative Percentage'] = 100 * sales_by_customer_sorted['Cumulative Sales'] / total_sales

# # Identify customers representing 80% of sales
# top_customers = sales_by_customer_sorted[sales_by_customer_sorted['Cumulative Percentage'] <= 80]

# print(top_customers)











































































