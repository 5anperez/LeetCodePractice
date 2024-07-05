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

# sales_df = pd.read_csv('./CSVs/Sales-PrintingOffice2023v1-Sheet1.csv')

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
# file_path_accounts = './CSVs/caja-dia-a-dia-no-Pii.csv'

# # Reading the CSV file
# accounts_data = pd.read_csv(file_path_accounts)

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



















# import pandas as pd

# Load the data
# file_path = './CSVs/DATA_ECOM_VAS_v1-.xlsx-Grossreport.csv'
# data = pd.read_csv(file_path)

# # Display the first few rows of the dataframe to understand its structure
# data.head()

# # Checking unique values for "Payment System Name" and "Status"
# unique_payment_systems = data['Payment System Name'].unique()
# unique_statuses = data['Status'].unique()

# (unique_payment_systems, unique_statuses)

# # Filtering data for "Faturado" status to count approved payments
# approved_payments = data[data['Status'] == 'Faturado']

# # Counting total and approved payments for each payment system
# total_payments_by_gateway = data['Payment System Name'].value_counts()
# approved_payments_by_gateway = approved_payments['Payment System Name'].value_counts()

# print(total_payments_by_gateway)
# print(approved_payments_by_gateway)

# # Calculating approval rate for each payment gateway
# approval_rate_by_gateway = (approved_payments_by_gateway / total_payments_by_gateway).fillna(0) * 100

# # Sorting the result for better visualization
# approval_rate_by_gateway.sort_values(ascending=False)

# print(approval_rate_by_gateway)
















# import pandas as pd
# import matplotlib.pyplot as plt

# Step 1: Load the CSV file into a DataFrame
# file_path = './CSVs/Bancos-Bodega.xlsxcsv-Global.csv'
# data = pd.read_csv(file_path)

# Data Cleaning: Clean up the "Monto" column
# Remove dollar signs, commas, and spaces, then convert to float
# data['Monto'] = data['Monto'].str.replace('$', '').str.replace(',', '').str.replace('-', '').str.strip().astype(float)

# Step 2: Calculate the total costs and percentages using column names
# Group by 'Clasificación' and sum up 'Monto' for each category
# total_costs_by_category = data.groupby('Rubro')['Monto'].sum()

# Calculate the percentage of total for each category
# total_costs = total_costs_by_category.sum()
# percentages = (total_costs_by_category / total_costs) * 100

# Step 3: Plot the graph
# plt.figure(figsize=(10, 8))
# percentages.plot(kind='bar')
# plt.title('Costs and Their Percentages by Category')
# plt.xlabel('Category')
# plt.ylabel('Percentage of Total Costs')
# plt.xticks(rotation=45)
# plt.show()



# Correcting the issue with the "Monto" column by handling spaces and potential negative signs.
# Remove spaces, then handle negative values correctly before converting to float.
# data['Monto'] = data['Monto'].str.replace(' ', '')  # Remove any spaces
# data['Monto'] = data['Monto'].apply(lambda x: float(x.replace(',', '').replace('$', '')) if isinstance(x, str) else x)

# # Filter the data to include only rows that represent costs based on 'Clasificación' or 'Rubro'.
# # Assuming 'Costos' in 'Clasificación' or 'Rubro' indicates a cost.
# cost_data = data[data['Clasificación'].str.contains('Costos') | data['Rubro'].str.contains('Costos')]

# # Calculate the total cost
# total_cost = cost_data['Monto'].sum()

# # Calculate the percentage of each cost
# cost_data['Percentage'] = (cost_data['Monto'] / total_cost) * 100

# # Aggregate the data by 'Clasificación' or 'Rubro' to sum costs and percentages for similar items
# aggregated_data = cost_data.groupby('Clasificación').agg({'Monto': 'sum', 'Percentage': 'sum'}).reset_index()

# # Display the aggregated data
# print(aggregated_data)

















# import pandas as pd

# # Load the dataset
# nz_data = pd.read_csv('./CSVs/nz.csv')

# # Convert the 'value' column to numeric to correct the TypeError
# nz_data['value'] = pd.to_numeric(nz_data['value'], errors='coerce')

# # Reattempt filtering for industries with a total income of less than 500 million dollars
# filtered_industries = nz_data[(nz_data['variable'] == 'Total income') & 
#                               (nz_data['value'] < 500) & 
#                               (nz_data['unit'] == 'DOLLARS(millions)')
# ].copy()

# print(filtered_industries.head())
# print(filtered_industries)

# Grouping the filtered industries by their names and summing their total income
# industry_grouped = filtered_industries.groupby('industry_name_ANZSIC')['value'].sum().reset_index()

# # print(industry_grouped)

# # Filtering the grouped data for industries with a total income of less than 500 million dollars
# industry_grouped_filtered = industry_grouped[industry_grouped['value'] < 500]
# print(industry_grouped_filtered)

# # Counting the number of unique industries with a total income of less than 500 million dollars
# unique_industries_count = industry_grouped_filtered.shape[0]

# print('Number of unique industries with a total income of less than 500 million dollars:', unique_industries_count)
# print(industry_grouped_filtered)

















'''
    Here, we deal with a dataset that has a column where the entries are ranges, e.g. 24-34, so we deal with them in a clever way by mapping them onto their respective midpoints. 
'''


# import pandas as pd
# import matplotlib.pyplot as plt

# # Correcting the file path and trying again
# file_path_corrected = './CSVs/Student_Mental_health.csv'  # Correcting the file name case
# data_corrected = pd.read_csv(file_path_corrected)

# # Normalize the "Your current year of Study" column to ensure consistency
# data_corrected['Your current year of Study'] = data_corrected['Your current year of Study'].str.lower().str.strip()

# # Convert CGPA ranges to their midpoints for calculation
# cgpa_mapping = {
#     '3.00 - 3.49': 3.245,
#     '3.50 - 4.00': 3.75,
#     '2.50 - 2.99': 2.745,
#     '2.00 - 2.49': 2.245,
#     '0 - 1.99': 0.995
# }
# data_corrected['CGPA Midpoint'] = data_corrected['What is your CGPA?'].map(cgpa_mapping)

# # Calculate average CGPA for each year of study
# avg_cgpa_by_year = data_corrected.groupby('Your current year of Study')['CGPA Midpoint'].mean()

# # Calculate the prevalence of depression for each year of study
# depression_prevalence_by_year = data_corrected.groupby('Your current year of Study')['Do you have Depression?'].apply(lambda x: (x == 'Yes').mean())

# # Plotting
# fig, ax = plt.subplots(1, 2, figsize=(14, 6))

# # Bar chart for average CGPA by year of study
# avg_cgpa_by_year.plot(kind='bar', ax=ax[0], color='skyblue')
# ax[0].set_title('Average CGPA by Year of Study')
# ax[0].set_xlabel('Year of Study')
# ax[0].set_ylabel('Average CGPA')
# ax[0].set_xticklabels(avg_cgpa_by_year.index, rotation=45)

# # Line chart for prevalence of depression by year of study
# depression_prevalence_by_year.plot(kind='line', ax=ax[1], marker='o', linestyle='-', color='tomato')
# ax[1].set_title('Prevalence of Depression by Year of Study')
# ax[1].set_xlabel('Year of Study')
# ax[1].set_ylabel('Prevalence of Depression')
# ax[1].set_xticks(range(len(depression_prevalence_by_year)))
# ax[1].set_xticklabels(depression_prevalence_by_year.index, rotation=45)

# plt.tight_layout()
# plt.show()


















# import matplotlib.pyplot as plt
# import pandas as pd

# pd.set_option('display.max_rows', None)
# pd.set_option('display.max_columns', None)

# # Read the CSV file into a DataFrame
# df = pd.read_csv("./CSVs/Netflix_TV_Shows_and_Movies.csv")

# # Create subplots
# fig, ax = plt.subplots(1, 2, figsize=(12, 6))

# # Plot histogram for TV shows
# ax[0].hist(df[df['type'] == 'SHOW']['release_year'], bins=20, label='TV Shows')
# ax[0].set_xlabel('Release Year')
# ax[0].set_ylabel('Frequency')
# ax[0].set_title('Release Year Distribution (TV Shows)')
# ax[0].legend()

# # Plot histogram for movies
# ax[1].hist(df[df['type'] == 'MOVIE']['release_year'], bins=20, label='Movies')
# ax[1].set_xlabel('Release Year')
# ax[1].set_ylabel('Frequency')
# ax[1].set_title('Release Year Distribution (Movies)')
# ax[1].legend()

# # Show the plot
# plt.tight_layout()
# plt.show()










# import matplotlib.pyplot as plt

# # # Read the CSV file into a DataFrame
# df = pd.read_csv("./CSVs/Netflix_TV_Shows_and_Movies.csv")

# Drop null values in `imdb_score`
# df.dropna(subset = ['imdb_score'], inplace=True)

# # Create a scatter plot between `release_year` and `imdb_score`
# plt.scatter(df['release_year'], df['imdb_score'])

# # Set the title as 'Scatter plot of imdb_score by release_year'
# plt.title('Scatter plot of imdb_score by release_year')

# # Set the x-axis label as 'Release Year' and y-axis label as 'IMDB Score'
# plt.xlabel('Release Year')
# plt.ylabel('IMDB Score')

# # Display the plot using plt.show()
# plt.show()





# import pandas as pd
# import matplotlib.pyplot as plt

# # Load the data
# file_path = './CSVs/Netflix_TV_Shows_and_Movies.csv'
# data = pd.read_csv(file_path)

# # Grouping the data by 'release_year' and calculating the average 'imdb_score'
# avg_imdb_score_by_year = data.groupby('release_year')['imdb_score'].mean().reset_index()

# # Plotting
# plt.figure(figsize=(12, 6))
# plt.scatter(avg_imdb_score_by_year['release_year'], avg_imdb_score_by_year['imdb_score'], color='blue', alpha=0.5)
# plt.title('Average IMDb Score by Release Year')
# plt.xlabel('Year')
# plt.ylabel('Average IMDb Score')
# plt.grid(True)
# plt.show()
















# import matplotlib.pyplot as plt
# from scipy.stats import linregress
# import pandas as pd

# file = './CSVs/Spotify_2000.csv'
# df = pd.read_csv(file)


# # Filter the data to only include tracks from the past 10 years
# df_recent = df[df['Year'] >= 2013]

# # Create the scatter plot with trend line
# plt.figure(figsize=(12, 6))
# plt.scatter(df_recent['Beats Per Minute (BPM)'], df_recent['Danceability'])

# # Calculate and add the trend line
# slope, intercept, r_value, p_value, std_err = linregress(
#     df_recent['Beats Per Minute (BPM)'], df_recent['Danceability']
# )
# trend_line = slope * df_recent['Beats Per Minute (BPM)'] + intercept
# plt.plot(df_recent['Beats Per Minute (BPM)'], trend_line, 'r--')

# # Add x and y axis labels
# plt.xlabel('Beats Per Minute (BPM)')
# plt.ylabel('Danceability')

# # Add title
# plt.title('Relationship between Beats Per Minute (BPM) and Danceability in Recent Tracks')

# # Display the plot
# plt.show()














'''
    NOTE: using a different encoding
'''


# import pandas as pd

# file_path = './CSVs/hotel_room.csv'

# # Attempt to read the CSV file again using a different encoding
# try:
#     df = pd.read_csv(file_path, encoding='ISO-8859-1')
# except Exception as e:
#     print(e)

# If successful, display the first few rows to understand its structure
# df.head()

# # Filter the rows where the property name contains "Samui"
# samui_rooms = df[df['property name '].str.contains("Samui", case=False, na=False)]

# # Calculate the average number of review counts for these rooms
# average_reviews = samui_rooms['review_count'].mean()

# print(average_reviews)




'''
    THIS IS AN EXAMPLE OF THE MODEL USING THE "errors='coerce'" FUNCTIONALITY TO GENEARTE A RESULT INCORRECTLY. THE MODEL IS TRYING TO TAKE A SHORTCUT AND NOT CLEAN UP THE DATA.
'''



# import pandas as pd

# Load the dataset
# df = pd.read_csv('hotel_room.csv', encoding='Windows-1252')

# Convert review_count to numeric, errors='coerce' will set invalid parsing to NaN
# df['review_count'] = pd.to_numeric(df['review_count'], errors='coerce')
# print(df['review_count'])

# # Filter rows where 'property name' contains 'Samui'
# df_samui = df[df['property name '].str.contains('Samui', case=False, na=False)]

# # Calculate the average number of review counts for rooms with 'Samui' in their property name
# average_reviews_samui = df_samui['review_count'].mean()

# print('Average review count for rooms with "Samui" in property name:', average_reviews_samui)


'''
    HERE, WE CLEAN UP THOSE VALUES. NOTE THAT THERE IS NO NEED TO COERCE!
'''


# Clean the 'review_count' column by removing commas and handling non-numeric values
# df['review_count'] = df['review_count'].str.replace(',', '')
# # Remove any non-digit characters from 'review_count'
# df['review_count'] = df['review_count'].str.extract('(\d+)', expand=False)
# # Convert 'review_count' to numeric
# df['review_count'] = pd.to_numeric(df['review_count'])
# # print(df['review_count'])

# # Filter rows where 'property name' contains 'Samui'
# df_samui = df[df['property name '].str.contains('Samui', case=False, na=False)]

# # Calculate the average number of review counts for rooms with 'Samui' in their property name
# average_reviews_samui = df_samui['review_count'].mean()


# print('Cleaned average review count for rooms with "Samui" in property name:', average_reviews_samui)














# import pandas as pd
# import matplotlib.pyplot as plt

# Load the dataset
# df_bollywood = pd.read_csv('./CSVs/Top_1000_Bollywood_Movies.csv')

# Filter movies with 'Verdict' of 'All Time Blockbuster' or 'Blockbuster'
# df_filtered = df_bollywood[df_bollywood['Verdict'].isin(['All Time Blockbuster', 'Blockbuster'])]

# # Sort by 'India Net' in descending order and select top 10
# df_top10 = df_filtered.sort_values(by='India Gross', ascending=False).head(10)

# # Plotting
# fig, ax1 = plt.subplots(figsize=(10, 6))
# fig.patch.set_facecolor('white')

# # Primary y-axis for 'Worldwide'
# ax1.bar(df_top10['Movie'], df_top10['Worldwide'], color='b', label='Worldwide')
# ax1.set_xlabel('Movie')
# ax1.set_ylabel('Worldwide Earnings', color='b')

# # Secondary y-axis for 'India Net'
# ax2 = ax1.twinx()
# ax2.plot(df_top10['Movie'], df_top10['India Net'], color='g', marker='o', label='India Net', linestyle='None')
# ax2.set_ylabel('India Net Earnings', color='g')

# # Third y-axis for 'Budget'
# ax3 = ax1.twinx()
# ax3.spines['right'].set_position(('outward', 60))
# ax3.plot(df_top10['Movie'], df_top10['Budget'], color='r', label='Budget', linestyle='-')
# ax3.set_ylabel('Budget', color='r')

# # Rotate movie names for better visibility
# plt.xticks(rotation=45)

# # Title and legend
# plt.title('Top 10 Highest-Grossing Bollywood Movies: Worldwide vs. India Net Earnings and Budget')
# fig.legend(loc='upper left', bbox_to_anchor=(0.1, 0.9))

# plt.tight_layout()
# plt.show()








# import matplotlib.pyplot as plt

# # Filter the data to only high-gross and blockbuster movies, then sort by net income
# top_10_grossing = df_bollywood.sort_values(by='India Gross', ascending=False).head(10)
# df_blockbusters = top_10_grossing[top_10_grossing['Verdict'].isin(['All Time Blockbuster', 'Blockbuster'])].sort_values(
#     by='India Net', ascending=False
# ).head(10)

# # Set up the plotting area
# fig, axs = plt.subplots(3, 1, figsize=(15, 10), sharex=True)

# # Plot Worldwide earnings
# axs[0].scatter(df_blockbusters['Movie'], df_blockbusters['Worldwide'])
# axs[0].set_title('Worldwide Earnings')
# axs[0].set_ylabel('Earnings (in crores)')

# # Plot India Net earnings
# axs[1].scatter(df_blockbusters['Movie'], df_blockbusters['India Net'])
# axs[1].set_title('India Net Earnings')
# axs[1].set_ylabel('Earnings (in crores)')

# # Plot Budget
# axs[2].plot(df_blockbusters['Movie'], df_blockbusters['Budget'])
# axs[2].set_title('Budget')
# axs[2].set_ylabel('Budget (in crores)')
# axs[2].set_xlabel('Movie')

# # Customize the plot
# plt.xticks(rotation=45)
# plt.tight_layout()
# plt.show()












# import pandas as pd

# # Load the Credit Score Classification dataset
# credit_score_df = pd.read_csv('./CSVs/Credit_score_classification.csv')

# # Display the first few rows to understand its structure
# credit_score_df.head()

# # Remove rows with non-numeric "Age" values
# # First, ensure that all "Age" values are strings to safely apply regex and conversion operations
# credit_score_df['Age'] = credit_score_df['Age'].astype(str)

# # Keep only rows where "Age" consists of digits
# credit_score_df = credit_score_df[credit_score_df['Age'].str.isdigit()]

# # Convert "Age" back to integers
# credit_score_df['Age'] = credit_score_df['Age'].astype(int)

# # Convert "Annual_Income" to numeric, errors='coerce' will turn invalid parsing into NaN, then drop these NaN values
# credit_score_df['Annual_Income'] = pd.to_numeric(credit_score_df['Annual_Income'], errors='coerce')
# credit_score_df.dropna(subset=['Annual_Income'], inplace=True)

# # Group by "Age" and calculate the average annual income
# age_income_group = credit_score_df.groupby('Age')['Annual_Income'].mean().reset_index()

# # Rename columns for clarity
# age_income_group.columns = ['Age Group', 'Average Annual Income']

# age_income_group













'''
    HERE, WE GENERATE THE STATS THAT ARE INBEDDED WITHIN A EXCEL FOLDER WITH SEVERAL SHEETS AND WE USE A NEW METHOD TO GET THE DATA FROM THE PARTICULAR SHEET WE NEED, WHICH IS THE SOUTH AMERICA DATA SET SHEET.
'''
# import pandas as pd

# # Load the South America tab from the Excel file
# data = pd.read_excel('./CSVs/population_and_age.xlsx', sheet_name='South America')

# # Calculate the average age and population
# average_age = data['Average Age'].mean()
# average_population = data['Population'].mean() / 1e6  # Convert to millions

# # Print the results
# print(f"Average Age in South America: {average_age} years")
# print(f"Average Population in South America: {average_population} million")













'''
    HERE, TO CLEAN UP THE DATE COLUMN, THE FUNCTION SHOULD CONSIDER THE FIRST CHAR OF EACH ENTRY TO DISTINGUISH THE FORMATS. THE FUNCTION NEEDS TO BE UPDATED TO HANDLE THIS.
'''

# import pandas as pd
# import matplotlib.pyplot as plt
# from datetime import datetime

# # Load the hoa_transactions.csv data
# hoa_transactions_data = pd.read_csv("./CSVs/hoa_transactions.csv")

# # Helper function to parse and standardize date formats
# def standardize_date(date_str):
#     # Check if the date is empty
#     if pd.isna(date_str):
#         return pd.NaT  # Return Not-a-Time for empty dates

#     # Try parsing day-month format first
#     try:
#         return datetime.strptime(date_str, '%d-%b').replace(year=2020)
#     except ValueError:
#         # If the first format fails, try month-day format
#         try:
#             return datetime.strptime(date_str, '%b-%y')
#         except ValueError:
#             # Return Not-a-Time if both formats fail
#             return pd.NaT

# # Apply the standardize_date function to the 'Date' column
# hoa_transactions_data['Standardized Date'] = hoa_transactions_data['Date'].apply(standardize_date)
# print(hoa_transactions_data)

# # Drop rows with NaT in 'Standardized Date' if needed
# cleaned_transactions_data = hoa_transactions_data.dropna(subset=['Standardized Date'])
# print(cleaned_transactions_data)

# Count the number of transactions by standardized date
# transactions_by_standardized_date = cleaned_transactions_data['Standardized Date'].value_counts().sort_index()

# # Plot
# plt.figure(figsize=(14, 7))
# plt.plot(transactions_by_standardized_date.index, transactions_by_standardized_date.values, marker='o', linestyle='-')
# plt.title('Number of HOA Transactions by Standardized Date')
# plt.xlabel('Date')
# plt.ylabel('Number of Transactions')
# plt.xticks(rotation=45)
# plt.grid(True)
# plt.tight_layout()  # Adjust layout to make room for the rotated x-axis labels
# plt.show()












'''
    HERE, WE MAKE A VERY COOL 3D INTERACTIVE GRAPH!
'''
# import pandas as pd
# import plotly.express as px

# # Load the dataset
# world_population_data = pd.read_csv("./CSVs/world-population-by-country-2020.csv")

# # Data cleaning and type conversion
# world_population_data['Population 2020'] = world_population_data['Population 2020'].str.replace(',', '').astype(int)
# world_population_data['Land Area (Km²)'] = world_population_data['Land Area (Km²)'].str.replace(',', '').astype(int)
# world_population_data['Density  (P/Km²)'] = world_population_data['Density  (P/Km²)'].str.replace(',', '').astype(int)
# # world_population_data['Med. Age'] = pd.to_numeric(world_population_data['Med. Age'], errors='coerce')

# # Create the 3D scatter plot
# fig = px.scatter_3d(world_population_data, x='Population 2020', y='Land Area (Km²)', z='Med. Age',
#                     color='Density  (P/Km²)',
#                     hover_name='Country (or dependency)',
#                     hover_data={
#                         'Population 2020': True,
#                         'Land Area (Km²)': True,
#                         'Med. Age': True,
#                         'Density  (P/Km²)': True,
#                         'no': False
#                     },
#                     title="World Population: Population vs. Land Area vs. Median Age",
#                     labels={'Population 2020': 'Population', 'Land Area (Km²)': 'Land Area (km²)', 'Med. Age': 'Median Age', 'Density  (P/Km²)': 'Population Density (P/Km²)'})

# # Enhance layout for better readability
# fig.update_layout(margin=dict(l=0, r=0, b=0, t=30))

# # Show the plot
# fig.show()







'''
    HERE, WE USE A FUNCTION TO CLEAN UP A MONEY COLUMN! IT ILLUSTRATES HOW TO USE A FUNCTION ON A COLUMN! 

    WE ALSO MODIFY AND EDIT THE DATE COLUMN.
'''



# import pandas as pd

# # Load the data
# sales_memos = pd.read_csv('./CSVs/sales_memos.csv', encoding='utf-8')

# # Function to normalize commission values
# def normalize_commission(value):
#     # Remove currency symbols and thousands separator
#     value = value.replace('Eur', '').replace('u$s', '').replace('$', '').replace('.', '')
#     # Replace comma with dot for decimal
#     value = value.replace(',', '.')
#     return float(value)

# # Apply the normalization function to the 'Commission' column
# sales_memos['Commission_Clean'] = sales_memos['Commission'].apply(normalize_commission)

# # Display the head of the dataframe to verify the changes
# # print(sales_memos.head())

# # Convert the 'Date' column to datetime format and extract the month and year
# sales_memos['Date'] = pd.to_datetime(sales_memos['Date'], dayfirst=True)
# # print(sales_memos['Date'])
# sales_memos['Month_Year'] = sales_memos['Date'].dt.to_period('M')
# # print(sales_memos['Month_Year'])

# # Group by 'Month_Year' and sum the 'Commission_Clean'
# monthly_sales = sales_memos.groupby('Month_Year')['Commission_Clean'].sum().reset_index()

# # Find the month with the lowest sales
# lowest_month = monthly_sales.loc[monthly_sales['Commission_Clean'].idxmin()]

# # print(lowest_month)

# # Group by 'Month_Year' and count the number of sales
# monthly_sales_count = sales_memos.groupby('Month_Year').size().reset_index(name='Sales_Count')
# # print(monthly_sales_count)

# # Find the month with the fewest recorded sales
# fewest_sales_month = monthly_sales_count.loc[monthly_sales_count['Sales_Count'].idxmin()]

# # print(fewest_sales_month)

# # Re-checking the data to include all years correctly
# # Group by month (ignoring the year) and count the number of sales
# monthly_sales_count_corrected = sales_memos['Date'].dt.month.value_counts().sort_index()
# print(monthly_sales_count_corrected)

# # Find the month with the fewest recorded sales considering all years correctly
# fewest_sales_month_corrected = monthly_sales_count_corrected.idxmin()

# # Display the corrected month with the fewest recorded sales and the count
# print('Month with the fewest recorded sales:', fewest_sales_month_corrected)
# print('Number of sales in that month:', monthly_sales_count_corrected[fewest_sales_month_corrected])












# import pandas as pd

# # Load the dataset
# mobile_brands_df = pd.read_csv('./CSVs/mobile-phone-brands-by-country.csv')

# Filter for Asian and South American countries
# asia_df = mobile_brands_df[mobile_brands_df['Region'] == 'Asia']
# south_america_df = mobile_brands_df[mobile_brands_df['Region'] == 'South America']

# # Group by country and count the number of brands for each country in those regions
# asia_brand_counts = asia_df.groupby('Country').size().reset_index(name='Brand_Count')
# south_america_brand_counts = south_america_df.groupby('Country').size().reset_index(name='Brand_Count')
# # print(asia_brand_counts)
# # print(south_america_brand_counts)

# # Calculate the median number of brands for Asian and South American countries
# median_asia = asia_brand_counts['Brand_Count'].median()
# median_south_america = south_america_brand_counts['Brand_Count'].median()

# print('Median number of brands for Asian countries:', median_asia)
# print('Median number of brands for South American countries:', median_south_america)













# import pandas as pd

# # Load the dataset
# current_accounts_df = pd.read_csv('./CSVs/current_accounts.csv', encoding='utf-8')

# # Attempting a different approach to convert Debit and Credit columns to numeric
# # First, replace commas with nothing and convert to float
# current_accounts_df['Debit'] = pd.to_numeric(current_accounts_df['Debit'].str.replace(',', '.'))
# current_accounts_df['Credit'] = pd.to_numeric(current_accounts_df['Credit'].str.replace(',', '.'))

# # Count the number of unique entries in the 'Supplier' column
# current_accounts_df['Supplier'] = current_accounts_df['Supplier'].str.strip() # Remove any leading/trailing whitespace
# unique_suppliers_count = current_accounts_df['Supplier'].nunique()

# # Calculate the average debit and credit amounts for each uniquely named supplier
# unique_avg_debit_credit = current_accounts_df.groupby('Supplier').agg(
#     Average_Debit=('Debit', 'mean'),
#     Average_Credit=('Credit', 'mean')
# ).reset_index()

# # Check if the number of avg debit/credit predictions equals the number of unique entries
# predictions_count = unique_avg_debit_credit.shape[0]

# # Format the two average columns to print the monetary values with commas and two decimal points
# unique_avg_debit_credit['Average_Debit'] = unique_avg_debit_credit['Average_Debit'].apply(lambda x: f'{x:,.2f}' if pd.notnull(x) else x)
# unique_avg_debit_credit['Average_Credit'] = unique_avg_debit_credit['Average_Credit'].apply(lambda x: f'{x:,.2f}' if pd.notnull(x) else x)


# # Display the first few rows of the average debit and credit for each supplier
# print(unique_avg_debit_credit)








'''
    USING A FUNCTION TO ASSIGN MORNING, AFTERNOON, AND EVENING LABELS TO DIFFERENT TIMES OF THE DAY.
'''




# import matplotlib.pyplot as plt
# from datetime import datetime
# import pandas as pd

# # Load the second dataset
# file_path_traffic = './CSVs/Legacy_Baton_Rouge_Traffic_Incidents.csv'
# data_traffic = pd.read_csv(file_path_traffic)

# # Filter for pedestrian incidents
# pedestrian_data = data_traffic[data_traffic['PEDESTRIAN'] == 'X']

# # Function to categorize time of day
# def categorize_time_of_day(time_str):
#     """Categorize time into Morning, Afternoon, and Evening."""
#     time_obj = datetime.strptime(time_str, '%I:%M %p')
#     if time_obj.hour < 12:
#         return 'Morning'
#     elif 12 <= time_obj.hour < 18:
#         return 'Afternoon'
#     else:
#         return 'Evening'

# # Re-apply the function to categorize 'CRASH TIME'
# pedestrian_data['Time of Day'] = pedestrian_data['CRASH TIME'].apply(categorize_time_of_day)
# print(pedestrian_data)

# Regroup by district and time of day, and count incidents
# grouped_data = pedestrian_data.groupby(['DISTRICT', 'Time of Day']).size().unstack(fill_value=0)



# # Plotting pie charts for each time of day
# fig, ax = plt.subplots(1, 3, figsize=(18, 6))

# time_periods = ['Morning', 'Afternoon', 'Evening']

# for i, time_period in enumerate(time_periods):
#     ax[i].pie(grouped_data[time_period], labels=grouped_data.index, autopct='%1.1f%%', startangle=140)
#     ax[i].set_title(f'Distribution of Pedestrian Incidents in the {time_period}')

# plt.tight_layout()
# plt.show()



















# import pandas as pd

# pd.set_option('display.max_rows', None)
# pd.set_option('display.max_columns', None)

# # Read the CSV file into a DataFrame
# df = pd.read_csv('./CSVs/Legacy_Baton_Rouge_Traffic_Incidents.csv')

# # Convert CRASH TIME to datetime
# df['CRASH TIME'] = pd.to_datetime(df['CRASH TIME']).dt.time
# print(df['CRASH TIME'])

# # Extract hour from CRASH TIME
# df['CRASH HOUR'] = df['CRASH TIME'].apply(lambda x: x.hour)

# # Categorize hour into morning, afternoon, and evening
# def categorize_hour(hour):
#   if 6 <= hour < 12:
#     return 'morning'
#   elif 12 <= hour < 18:
#     return 'afternoon'
#   else:
#     return 'evening'

# df['TIME_CATEGORY'] = df['CRASH HOUR'].apply(categorize_hour)
# # print(df)

# # Filter rows where PEDESTRIAN is not null
# df_pedestrian = df[df['PEDESTRIAN'].notnull()]

# # Compute counts by DISTRICT and time category
# df_grouped = df_pedestrian.groupby(['DISTRICT', 'TIME_CATEGORY']).size().unstack(fill_value=0)

# import matplotlib.pyplot as plt

# # Plot pie charts
# fig, axes = plt.subplots(1, 3, figsize=(15, 5))
# fig.suptitle('Pedestrian Accidents by District and Time of Day')

# axes[0].pie(df_grouped['morning'], labels=df_grouped.index, autopct='%1.1f%%')
# axes[0].set_title('Morning (6am-12pm)')

# axes[1].pie(df_grouped['afternoon'], labels=df_grouped.index, autopct='%1.1f%%')
# axes[1].set_title('Afternoon (12pm-6pm)')

# axes[2].pie(df_grouped['evening'], labels=df_grouped.index, autopct='%1.1f%%')
# axes[2].set_title('Evening (6pm-12am)')

# # Equal aspect ratio for all pie charts
# for ax in axes:
#   ax.axis('equal')

# plt.show()
















# Convert 'Date Issue' to datetime format and check the conversion
# import pandas as pd

# # Assuming 'data' is your DataFrame after loading and cleaning
# file_path = './CSVs/cdlm_purchases.csv'
# df = pd.read_csv(file_path)

# # Convert 'Date Issue' to datetime format
# df['Date Issue'] = pd.to_datetime(df['Date Issue'], format='%d/%m/%Y')

# # Clean up 'Total Amount' by replacing commas with dots and converting to float
# df['Total Amount'] = df['Total Amount'].str.replace(',', '.').astype(float)

# # Now, you can analyze the distribution of payment amounts across dates and business units
# # For example, to get a summary of total amounts by date and shop:
# summary = df.groupby(['Date Issue', 'Shop'])['Total Amount'].sum().reset_index()

# # If you want to visualize this distribution, you might consider line plots for each business unit over time, or bar plots for comparing total amounts by business units on specific dates.

# # Example of plotting total amounts by business unit over time
# import matplotlib.pyplot as plt

# fig, ax = plt.subplots(figsize=(10, 6))
# for shop in df['Shop'].unique():
#     shop_data = summary[summary['Shop'] == shop]
#     ax.plot(shop_data['Date Issue'], shop_data['Total Amount'], label=shop)

# ax.set_xlabel('Date')
# ax.set_ylabel('Total Amount')
# ax.set_title('Distribution of Payment Amounts by Business Unit Over Time')
# ax.legend()

# plt.show()








# import pandas as pd
# import matplotlib.pyplot as plt

# # Assuming 'data' is your DataFrame after loading and cleaning
# file_path = './CSVs/cdlm_purchases.csv'
# data = pd.read_csv(file_path)

# data['Date Issue'] = pd.to_datetime(data['Date Issue'], format='%d/%m/%Y')
# data['Total Amount'] = data['Total Amount'].str.replace(',', '.').astype(float)

# # Time Series Plot
# # Aggregate data by 'Date Issue'
# time_series_data = data.groupby('Date Issue')['Total Amount'].sum().reset_index()

# plt.figure(figsize=(12, 6))
# plt.plot(time_series_data['Date Issue'], time_series_data['Total Amount'], marker='o', linestyle='-')
# plt.title('Total Payment Amounts Over Time')
# plt.xlabel('Date')
# plt.ylabel('Total Payment Amount')
# plt.xticks(rotation=45)  # Rotate date labels for better readability
# plt.tight_layout()  # Adjust layout to make room for the rotated date labels
# plt.show()

# # Box Plot
# plt.figure(figsize=(12, 6))
# plt.boxplot([data.loc[data['Shop'] == shop, 'Total Amount'] for shop in data['Shop'].unique()], labels=data['Shop'].unique())
# plt.title('Distribution of Payment Amounts Across Business Units')
# plt.xlabel('Business Unit (Shop)')
# plt.ylabel('Payment Amount')
# plt.xticks(rotation=45)  # Rotate shop labels for better readability
# plt.tight_layout()  # Adjust layout to make room for the rotated shop labels
# plt.show()












'''
    HERE, WE ANALYZE AN EXCEL SHEET THAT IS STRUCTURED LIKE A MULTIDIMENSIONAL MATRIX, WHICH IS UNUSUAL. THEREFORE, WE HAVE TO IDENTIFY THE RELEVANT COLUMNS AND THEN CREATE ROW AND COL RANGES SUCH THAT WE CLAMP THE ANALYSIS TO SPAN THE CELLS REGARDING INCOME. THE REST OF THE CELLS ARE EXPENSES, SO WE IGNORE THEM, BUT THE CLAMP/RANGES IS VERY USEFUL.
'''

# import pandas as pd
# import matplotlib.pyplot as plt

# df = pd.read_excel("./CSVs/outcomes_incomes_fs.xlsx")

# # Aggregate the total monthly income across all categories
# monthly_income = df.iloc[2:11, 2:13].sum()

# # Create a list of month names for plotting
# months = ['January', 'February', 'March', 'April', 'May', 'June', 'July', 'August', 'September', 'October', 'November']

# # Plotting
# plt.figure(figsize=(10, 6), facecolor='white')
# plt.plot(months, monthly_income, marker='o', linestyle='-', color='blue')
# plt.title('Total Monthly Income Across All Categories')
# plt.xlabel('Month')
# plt.ylabel('Total Income')
# plt.xticks(rotation=45)
# plt.tight_layout()
# plt.savefig('total_monthly_income_chart.png')
# plt.show()
# print('Line chart created and saved as total_monthly_income_chart.png.')











# import matplotlib.pyplot as plt
# import pandas as pd

# # Read the CSV file into a DataFrame
# df = pd.read_csv('./CSVs/compras-wines.csv')

# # Drop null values (Invoiced Amount Loc. (Loc. likely stands for Local))
# df.dropna(subset=['Ítem - Impte. Fact. Loc.'], inplace=True)

# # Convert `Comp. - F. Emisión` to datetime (Date of Issue)
# df['Comp. - F. Emisión'] = pd.to_datetime(df['Comp. - F. Emisión'], format='%d/%m/%Y')

# # Group by `Comp. - F. Emisión` and sum `Ítem - Impte. Fact. Loc.`
# df_grouped = df.groupby('Comp. - F. Emisión')['Ítem - Impte. Fact. Loc.'].sum().reset_index()

# # Sort by `Comp. - F. Emisión` in ascending order
# df_grouped = df_grouped.sort_values(by='Comp. - F. Emisión')

# # Print the grouped DataFrame
# # print(df_grouped.to_markdown(index=False, numalign="left", stralign="left"))

# # Plot line graph
# plt.plot(df_grouped['Comp. - F. Emisión'], df_grouped['Ítem - Impte. Fact. Loc.'])

# # Add labels and title
# plt.xlabel('Date', fontsize=14)
# plt.ylabel('Total Invoice Amount', fontsize=14)
# plt.title('Total Invoice Amount by Date', fontsize=14)

# # Rotate X ticks 45 degrees
# plt.xticks(rotation=45)

# Display plot
# plt.show()














'''
    HERE, WE CLEAN UP DATES BY EXTRACTING THE YEAR THEN COUNTING THE NUMBER OF NAN VALUES THERE ARE. WE ALSO CAN EXTRACT THEIR ACTUAL ROWS OR ROW INDEXES.
'''


# Let's first load the uploaded CSV file to understand its structure and contents
# import pandas as pd
# import matplotlib.pyplot as plt
# import seaborn as sns

# # Load the dataset
# file_path = './CSVs/Top5000.csv'
# data = pd.read_csv(file_path)

# Split the 'rel_date' column by spaces and take the last part, assuming the year is always the last
# data['year_extracted'] = data['rel_date'].str.split().str[-1]
# print(f'The years extracted column: {data["year_extracted"]}')

# Count the number of NaNs and display
# nan_count_year_extracted = data['year_extracted'].isna().sum()
# print(f'We have {nan_count_year_extracted} NaN values in year extracted!')

# Clean up the num_rat column
# data['num_rat'] = data['num_rat'].str.replace(',', '').astype(float)

# Count the number of NaNs and display
# nan_count_rel_date = data['rel_date'].isna().sum()
# nan_count_rel_year = data['year'].isna().sum()
# print(f'We have {nan_count_rel_date} NaN values in rel_date!')
# print(f'We have {nan_count_rel_year} NaN values in year!')

# Create a boolean mask where True indicates NaN entries
# nan_mask = data['rel_date'].isna()

# Use the mask to filter the DataFrame and extract rows with NaN in 'rel_date'
# nan_rows = data[nan_mask]
# print(f'These are the entries: {nan_rows}')

# If you only need the indexes
# nan_indexes = data.index[nan_mask].tolist()
# print(f'These are their indexes: {nan_indexes}')

# Analyze the distribution of genres
# genre_counts = data['gens'].str.split(', ').explode().value_counts().head(10)

# # Analyze the average rating over years
# rating_over_years = data.groupby('year_extracted')['avg_rat'].mean()

# # Relationship between danceability and energy
# sns.scatterplot(data=data, x='danceability', y='energy')
# plt.title('Relationship between Danceability and Energy')
# plt.xlabel('Danceability')
# plt.ylabel('Energy')
# # plt.show()

# print(genre_counts) 
# # Show the latest 10 years for trend analysis
# print(rating_over_years.tail(10))  










'''
    HERE, WE CREATE A REALLY COOL ASTHETIC GRAPH THAT COULD BE CALLED A "DARK MODE" GRAPH! ITS ALL BLACK WITH NEON BARS!
'''


# Import required library
# import matplotlib.pyplot as plt
# import seaborn as sns
# import pandas as pd

# pd.set_option('display.max_rows', None)
# pd.set_option('display.max_columns', None)

# # Read the CSV file into a DataFrame
# gas_prices_file_path = './CSVs/Gas_Prices.csv'

# # Attempt to load the Gas Prices dataset with a different encoding
# try:
#     df = pd.read_csv(gas_prices_file_path, encoding='ISO-8859-1')
# except Exception as e:
#     load_error = e
# else:
#     load_error = None
#     # Display the first few rows of the dataset to understand its structure
#     gas_prices_data_head = df.head()



# '''
#     Step 1. First, we get the top 15 countries with the highest oil consumption.
# '''


# # Convert `Daily Oil Consumption (Barrels)` to numeric after removing ','
# df['Daily Oil Consumption (Barrels)'] = df['Daily Oil Consumption (Barrels)'].astype(str).str.replace(',', '')
# df['Daily Oil Consumption (Barrels)'] = pd.to_numeric(df['Daily Oil Consumption (Barrels)'])

# # Sort by "Daily Oil Consumption (Barrels)" and select the top 15
# top_15_daily_oil_consumption = df.sort_values(by='Daily Oil Consumption (Barrels)', ascending=False).head(15)

# # Extract the relevant columns to display
# top_15_countries_oil_consumption = top_15_daily_oil_consumption[['Country', 'Daily Oil Consumption (Barrels)']]

# print(top_15_countries_oil_consumption)


# '''
#     Step 2. Then, after identifying the top 15 countries based on their daily oil consumption, this column will provide the data needed to plot the annual share of gallons per capita for each of these countries.
# '''


# # Filter the dataset for only the top 15 countries by daily oil consumption
# top_15_countries_list = top_15_countries_oil_consumption['Country'].tolist()
# top_15_by_daily_consumption = df[df['Country'].isin(top_15_countries_list)]

# # Ensure the countries are ordered by their daily oil consumption for the chart
# top_15_by_daily_consumption = top_15_by_daily_consumption.set_index('Country').loc[top_15_countries_list].reset_index()

# # Create the bar chart for the annual share of gallons per capita for these top 15 countries
# plt.figure(figsize=(14, 8))
# # Neon-like colors for the bars
# neon_colors = ["#39FF14", "#DFFF00", "#FF355E", "#FD5B78", "#FF6037", "#FF9966", "#FF9933", "#FFCC33", "#FFFF66", "#CCFF00", "#66FF66", "#AAF0D1", "#50BFE6", "#FF6EFF", "#EE34D2"]
# sns.barplot(x='Country', y='Yearly Gallons Per Capita', data=top_15_by_daily_consumption, palette=neon_colors)

# # Set the title and labels with white color
# plt.title('Annual Share of Gallons Per Capita for Top 15 Countries by Daily Oil Consumption', fontsize=16, color='white')
# plt.xlabel('Country', fontsize=12, color='white')
# plt.ylabel('Yearly Gallons Per Capita', fontsize=12, color='white')

# # Rotate the x-axis labels and set them to white color
# plt.xticks(rotation=45, ha='right', color='white')
# plt.yticks(color='white')

# # Change the color of the axes, ticks and border to black
# plt.gca().spines['bottom'].set_color('black')
# plt.gca().spines['left'].set_color('black')
# plt.gca().spines['right'].set_color('black')
# plt.gca().spines['top'].set_color('black')
# plt.tick_params(colors='white', which='both') # changes the color of the ticks

# # Set the background color to black
# plt.gca().set_facecolor('black')
# plt.gcf().set_facecolor('black')

# # Version 1
# # plt.show()


# '''
#     Here, we exclude China
# '''


# # Since China is excluded, select the top 16 to ensure we have 15 countries after excluding China
# top_16_daily_oil_consumption_excluding_china = df.sort_values(by='Daily Oil Consumption (Barrels)', ascending=False).head(16)
# top_15_excluding_china_corrected = top_16_daily_oil_consumption_excluding_china[top_16_daily_oil_consumption_excluding_china['Country'] != 'China']

# # Ensure the countries are ordered by their daily oil consumption for the chart
# top_15_countries_list_corrected = top_15_excluding_china_corrected['Country'].tolist()
# top_15_by_daily_consumption_corrected = df[df['Country'].isin(top_15_countries_list_corrected)]
# top_15_by_daily_consumption_corrected = top_15_by_daily_consumption_corrected.set_index('Country').loc[top_15_countries_list_corrected].reset_index()

# # Create the corrected bar chart for the annual share of gallons per capita for the correct top 15 countries excluding China
# plt.figure(figsize=(14, 8))
# sns.barplot(x='Country', y='Yearly Gallons Per Capita', data=top_15_by_daily_consumption_corrected, palette=neon_colors)

# # Set the title and labels with white color
# plt.title('Annual Share of Gallons Per Capita for Top 15 Countries by Daily Oil Consumption (Excluding China)', fontsize=16, color='white')
# plt.xlabel('Country', fontsize=12, color='white')
# plt.ylabel('Yearly Gallons Per Capita', fontsize=12, color='white')

# # Rotate the x-axis labels and set them to white color
# plt.xticks(rotation=45, ha='right', color='white')
# plt.yticks(color='white')

# # Change the color of the axes, ticks and border to black
# plt.gca().spines['bottom'].set_color('black')
# plt.gca().spines['left'].set_color('black')
# plt.gca().spines['right'].set_color('black')
# plt.gca().spines['top'].set_color('black')
# plt.tick_params(colors='white', which='both') # changes the color of the ticks

# # Set the background color to black
# plt.gca().set_facecolor('black')
# plt.gcf().set_facecolor('black')

# # Version 2
# plt.show()














'''
    HERE, I HAD TO COUNT THE NUMBER OF NANS THAT ALSO HAVE A SPECIFIC ENTRY IN THE VARIABLE COLUMN. THIS WAS TO CONFIRM THAT WE CAN USE THE COERCE OPTION TO CLEAN THE DATA WITHOUT OMITTED USEFUL DATA!
'''


# import pandas as pd
# import matplotlib.pyplot as plt

# df_nz = pd.read_csv('./CSVs/nz.csv')

# Inspecting the first few rows to understand the structure of the dataset
# print(df_nz.head())

# Filter the DataFrame for rows where 'variable' column contains 'Activity unit'
# filtered_df = df_nz[df_nz['variable'] == 'Activity unit']

# # Count the NaN values in the 'value' column of the filtered DataFrame
# nan_count = filtered_df['value'].isna().sum()

# print(f'Number of NaNs in the "value" column with "Activity unit" in the "variable" column: {nan_count}')

# Converting 'value' column to numeric type
# df_nz['value'] = pd.to_numeric(df_nz['value'], errors='coerce')

# # Count the number of NaNs and display
# nan_count = df_nz['value'].isna().sum()
# print(f'We have {nan_count} NaN values in value!')

# Re-attempting the filtering with the corrected 'value' column type
# df_activity_units = df_nz[(df_nz['variable'] == 'Activity unit') & (df_nz['value'] > 2000)]

# # Grouping by industry name to sum up the Activity unit values (in case of multiple entries per industry)
# df_grouped = df_activity_units.groupby('industry_name_ANZSIC')['value'].sum().reset_index()

# # Sorting the values for better visualization
# df_grouped_sorted = df_grouped.sort_values(by='value', ascending=False)

# # Plotting
# plt.figure(figsize=(10, 8), facecolor='white')
# plt.barh(df_grouped_sorted['industry_name_ANZSIC'], df_grouped_sorted['value'], color='skyblue')
# plt.xlabel('Activity Unit Value')
# plt.ylabel('Industry Name')
# plt.title('NZ Industries with Activity Unit Value > 2000')
# plt.tight_layout()
# plt.show()








# # Let's first load the uploaded CSV file to understand its structure and content
# import pandas as pd

# # Load the data
# df_nz = pd.read_csv('./CSVs/nz.csv')

# # Display the first few rows of the dataframe to understand its structure
# df_nz.head()

# # Filter the dataset for "Activity unit" variable with values more than 2000
# activity_units_filter = (df_nz['variable'] == 'Activity unit') & (df_nz['value'] > 2000)

# # Focused dataset
# activity_units_data = df_nz[activity_units_filter]

# # Summarize data to get total activity units for each industry
# summary_data = activity_units_data.groupby('industry_name_ANZSIC')['value'].sum().reset_index()

# # Sorting the data for better visualization
# summary_sorted = summary_data.sort_values(by='value', ascending=False)

# # Importing visualization library
# import matplotlib.pyplot as plt

# # Plot
# plt.figure(figsize=(10, 8))
# plt.barh(summary_sorted['industry_name_ANZSIC'], summary_sorted['value'], color='skyblue')
# plt.xlabel('Total Activity Units')
# plt.ylabel('Industry')
# plt.title('NZ Industries with Activity Unit Value > 2000')
# plt.gca().invert_yaxis()  # Invert y-axis to have the largest bar on top
# plt.tight_layout()
# plt.show()











'''
    HERE, THERE ARE SOME DIVIDE BY ZERO CALCS GOING ON SO I HAD TO SET A CONDITIONAL THAT ONLY DIVIDES IF THE DENOMINATOR IS NOT ZERO! I FOUND OUT THAT THIS WAS HAPPENING BECAUSE IT WAS SPITTING OUT 'inf' VALUES, SO AFTER I COUNTED THE NUMBER OF inf VALUES AND FOUND THAT THERE WERE NOT TOO MANY, I DECIDED TO SIMPLY SKIP THE ONES THAT WERE NO GOOD!
'''




# import pandas as pd
# import numpy as np

# # Loading the dataset
# monkey_pox_df = pd.read_csv('./CSVs/Monkey_Pox_Cases_Worldwide.csv')

# Count the number of infinity values in the 'Hospitalization_Rate' column
# num_inf_in_rate = (monkey_pox_df['Hospitalized'] == float('inf')).sum()
# print(f'We have {num_inf_in_rate} inf values in Hospitalized!')

# # Count the number of infinity values in the 'Hospitalization_Rate' column
# num_inf_in_rate = (monkey_pox_df['Confirmed_Cases'] == float('inf')).sum()
# print(f'We have {num_inf_in_rate} inf values in Confirmed_Cases!')


# Assuming your DataFrame is named monkey_pox_df
# monkey_pox_df['Hospitalization_Rate'] = np.where(monkey_pox_df['Confirmed_Cases'] > 0, 
#                                                  monkey_pox_df['Hospitalized'] / monkey_pox_df['Confirmed_Cases'], 
                                                #  0)


# Calculating hospitalization rate for each country
# monkey_pox_df['Hospitalization_Rate'] = monkey_pox_df['Hospitalized'] / monkey_pox_df['Confirmed_Cases']
# print(monkey_pox_df['Hospitalization_Rate'])

# Count the number of NaNs and display
# nan_count = monkey_pox_df['Hospitalization_Rate'].isna().sum()
# print(f'We have {nan_count} NaN values in Rate!')

# Count the number of infinity values in the 'Hospitalization_Rate' column
# num_inf_in_rate = (monkey_pox_df['Hospitalization_Rate'] == float('inf')).sum()
# print(f'We have {num_inf_in_rate} inf values in Rate!')


# Getting the minimum and maximum hospitalization rate values
# min_hospitalization_rate = monkey_pox_df['Hospitalization_Rate'].min()
# max_hospitalization_rate = monkey_pox_df['Hospitalization_Rate'].max()

# # Calculating the difference between the maximum and minimum hospitalization rates
# difference = max_hospitalization_rate - min_hospitalization_rate

# print('Minimum Hospitalization Rate:', min_hospitalization_rate)
# print('Maximum Hospitalization Rate:', max_hospitalization_rate)
# print('Difference:', difference)











# import pandas as pd
# import calendar

# Load the dataset
# df = pd.read_csv('./CSVs/ttc-bus-delay-data-2022.csv')

# We need to find the accident entries
# unique_incidents = df['Incident'].unique()

# Print out all unique entries in the 'Incident' column
# print(unique_incidents)

# We need to find the months this dataset spans
# unique_dates = df['Date'].unique()

# Print out all unique entries in the 'Date' column
# print(unique_dates)

# Focusing on 'Collision - TTC' incidents and generating a table with the accident count per month
# collision_df = df[df['Incident'] == 'Collision - TTC']

# # Convert the 'Date' column to datetime format
# collision_df['Date'] = pd.to_datetime(collision_df['Date'])

# # Group by month and count the number of collisions
# monthly_collisions = collision_df.groupby(collision_df['Date'].dt.strftime('%B')).size().reset_index(name='Accident Count')

# # Sorting by month to ensure chronological order
# monthly_collisions['Month'] = pd.Categorical(monthly_collisions['Date'], categories=list(calendar.month_name[1:]), ordered=True)
# monthly_collisions.sort_values('Month', inplace=True)
# monthly_collisions.drop('Date', axis=1, inplace=True)

# print(monthly_collisions)







'''
    HERE WE HAVE TO EXTRACT THE DATE FROM A LONGER STRING THAT DENOTES A RANGE. THE RANGE IS THE DURATION THAT SOMEONE STAYED AT THE AIR BNB.
'''

# import pandas as pd

# # Load the airbnb_reviews_ruben.csv dataset
# file_path_airbnb_reviews = './CSVs/airbnb_reviews_ruben.csv'
# airbnb_reviews_data = pd.read_csv(file_path_airbnb_reviews)

# # Display the first few rows of the dataset to understand its structure
# # print(airbnb_reviews_data.head())

# # Extract the month from the date column and count the occurrences of each month
# airbnb_reviews_data['month'] = pd.to_datetime(airbnb_reviews_data['date'].str.extract('([A-Za-z]+)')[0], format='%b').dt.month_name()

# # Count the number of bookings per month
# bookings_per_month = airbnb_reviews_data['month'].value_counts()

# # Identify the month with the most bookings
# most_bookings_month = bookings_per_month.idxmax()
# most_bookings_count = bookings_per_month.max()

# print(f'Most booking month: {most_bookings_month}') 
# print(f'Most bookings count: {most_bookings_count}')
















'''
    HERE, WE GENERATE A SERIES OF GRAPHS USING A FOR LOOP TO SPIT THEM OUT!
'''

# import matplotlib.pyplot as plt
# import pandas as pd

# pd.set_option('display.max_rows', None)
# pd.set_option('display.max_columns', None)

# # Read the CSV file into a DataFrame
# df = pd.read_csv('./CSVs/Top_1000_Bollywood_Movies.csv')

# # Convert `Budget` to billions
# df['Budget'] = df['Budget'] / 1e9

# # Drop rows with null values in `Budget` or `Verdict`
# df.dropna(subset=['Budget', 'Verdict'], inplace=True)

# # Create subplots with one row and as many columns as there are unique values in `Verdict`
# fig, axes = plt.subplots(1, len(df['Verdict'].unique()), figsize=(15, 5))

# # Iterate through each unique `Verdict` value and create a histogram of `Budget` for that `Verdict` in the corresponding subplot
# for i, verdict in enumerate(df['Verdict'].unique()):
#     df_verdict = df[df['Verdict'] == verdict]
#     axes[i].hist(df_verdict['Budget'], alpha=0.5, label=verdict)
#     axes[i].set_title(verdict)
#     axes[i].set_xlabel('Budget (in billions)')
#     axes[i].set_ylabel('Frequency')

# # Set the title of the overall figure
# fig.suptitle('Distribution of Budget by Verdict')

# # Add a legend outside the plot
# plt.legend(loc='upper left', bbox_to_anchor=(1, 1))

# # Show the plot
# plt.show()














'''
    HERE, WE GENERATE A COOL LINE GRAPH WITH THREE COLORS WHERE EACH COLOR REPRESENTS A COLUMN, E.G., INCOME, EXPENSES, AND CASH FLOW. WE INVOKE A NEAT STRATEGY TO EXTRACT THE YEAR FROM A SPANISH DATE IN THIS FORMATE: "MAYO 2020". WE OMIT THE MONTH AND GRAB THE YEAR.
'''

# import matplotlib.pyplot as plt
# import pandas as pd

# # Load the dataset to understand its structure
# file_path = './CSVs/business_unit_system_cash_flow.csv'
# cash_flow_data = pd.read_csv(file_path)

# # Correcting the approach to handle NaN values properly
# cash_flow_data['Year'] = cash_flow_data['Período'].str.extract('(\d{4})')
# cash_flow_data.dropna(subset=['Year'], inplace=True)  # Dropping rows where year couldn't be extracted
# cash_flow_data['Year'] = cash_flow_data['Year'].astype(int)  # Converting years to integers

# # Grouping the data by 'Year' again after cleanup
# annual_summary_corrected = cash_flow_data.groupby('Year').sum()

# # Plotting the corrected annual trends
# plt.figure(figsize=(12, 6))
# plt.plot(annual_summary_corrected.index, annual_summary_corrected['Ingresos'], label='Ingresos (Income)', marker='o')
# plt.plot(annual_summary_corrected.index, annual_summary_corrected['Egresos'], label='Egresos (Expenses)', marker='x')
# plt.plot(annual_summary_corrected.index, annual_summary_corrected['Efectivo'], label='Efectivo (Cash Flow)', marker='s')

# plt.title('Corrected Annual Trends of Income, Expenses, and Cash Flow')
# plt.xlabel('Year')
# plt.ylabel('Value')
# plt.legend()
# plt.grid(True)
# plt.show()










# import pandas as pd
# import matplotlib.pyplot as plt
# import seaborn as sns

# Load the dataset to take a look at the first few rows and understand its structure
# data_path = './CSVs/Top5000.csv'
# df = pd.read_csv(data_path)

# Extract the year from the 'rel_date' column
# Split the 'rel_date' column by spaces and take the last part, bc the year is always the last
# df['release_year'] = df['rel_date'].str.split().str[-1]





# Convert 'rel_date' to datetime format 
# NOTE: THIS APPROACH CAUSES 1658 NaNs! I am comparing the two approaches
# to see how much of an effect results from omitting all those data
# df['rel_date'] = pd.to_datetime(df['rel_date'], errors='coerce')

# # Extract the year from the 'rel_date' column
# df['release_year'] = df['rel_date'].dt.year

# # Drop rows where 'release_year' is NaN after conversion
# df = df.dropna(subset=['release_year'])






# Convert 'release_year' to integer
# df['release_year'] = df['release_year'].astype(int)

# Count the number of NaNs and display
# nan_count_year_extracted = df['rel_date'].isna().sum()
# print(f'We have {nan_count_year_extracted} NaN values in year extracted!')

# Plotting the histogram of music releases over time
# plt.figure(figsize=(14, 7))
# sns.histplot(df['release_year'], bins=50, kde=False)
# plt.title('Music Releases Over Time')
# plt.xlabel('Release Year')
# plt.ylabel('Number of Releases')
# plt.xticks(rotation=45)
# plt.show()



# Filter the dataframe for the last 10 years
# df_last_10_years = df[df['release_year'] >= (df['release_year'].max() - 10)]

# # Group by 'release_year' and calculate the average 'avg_rat'
# avg_rating_per_year = df_last_10_years.groupby('release_year')['avg_rat'].mean().reset_index()

# # Plotting the average album rating over the last ten years
# plt.figure(figsize=(12, 6))
# sns.lineplot(data=avg_rating_per_year, x='release_year', y='avg_rat', marker='o')
# plt.title('Average Album Rating Over the Last Ten Years')
# plt.xlabel('Release Year')
# plt.ylabel('Average Rating')
# plt.xticks(avg_rating_per_year['release_year'])
# plt.grid(True)
# plt.show()













# import pandas as pd
# import matplotlib.pyplot as plt

# # Load the new dataset
# df_dates = pd.read_csv('./CSVs/caja-dia-a-dia-no-Pii.csv')

# Inspect the first few rows to understand the structure and identify the date column
# df_dates.head()

''' HERE WE USE PANDAS TO GENERATE THE HISTOGRAM'''
# Convert the 'Fecha' column to datetime format
# df_dates['Fecha'] = pd.to_datetime(df_dates['Fecha'])
# # Plotting the histogram
# plt.figure(figsize=(10, 6))
# df_dates['Fecha'].hist(bins=50, color='skyblue', edgecolor='black')
# plt.title('Distribution of Dates in Data')
# plt.xlabel('Date')
# plt.ylabel('Frequency')
# plt.xticks(rotation=45)
# plt.tight_layout()  # Adjust layout to not cut off labels
# plt.show()

''' HERE WE USE MATPLOTLIB TO GENERATE THE HISTOGRAM'''
# df_dates["Fecha"] = pd.to_datetime(df_dates["Fecha"], format="%Y-%m-%d")
# plt.hist(df_dates["Fecha"])
# plt.xlabel("Date")
# plt.ylabel("Frequency")
# plt.xticks(rotation=45)
# plt.tight_layout()
# plt.show()







'''
    HERE, WE INVOKE THE BEST POSSIBLE WAY TO CLEAN UP DATA... USING A REGEX!
'''

# import pandas as pd

# # Load the new dataset
# df_spotify = pd.read_csv('./CSVs/Spotify_2000.csv')

# # Convert 'Length (Duration)' to numeric, removing any non-numeric characters first
# df_spotify['Length (Duration)'] = df_spotify['Length (Duration)'].str.replace('[^\d]', '', regex=True).astype(int)

# # Calculate the standard deviation of song lengths
# duration_std = df_spotify['Length (Duration)'].std()

# # Find the song with the maximum length
# max_length_song = df_spotify.loc[df_spotify['Length (Duration)'].idxmax()]

# print(f'Standard deviation of song lengths: {duration_std:.2f}')
# print('Song with the maximum length:', max_length_song[['Title', 'Artist', 'Length (Duration)']])













# import pandas as pd
# import matplotlib.pyplot as plt
# from datetime import datetime

# # First, let's load the 'Data Set #13 report.csv' dataset to understand its structure and identify the relevant columns.
# data_set_13_path = './CSVs/Data Set #13 report.csv'
# data_set_13 = pd.read_csv(data_set_13_path)

# # Convert 'LAST REC DATE' to datetime
# # Assuming the format is Month-Year, we need to handle cases where the year might be two digits.
# def convert_date(date_str):
#     try:
#         return datetime.strptime(date_str, '%B-%y')
#     except ValueError:
#         # Handle cases where the year is already in four digits or other anomalies.
#         try:
#             return datetime.strptime(date_str, '%B-%d')
#         except:
#             return None

# data_set_13['LAST REC DATE'] = data_set_13['LAST REC DATE'].apply(convert_date)

# # To generate a line graph between 'month_year' and 'qty_on_hand units', we first need to prepare 'month_year' from 'LAST REC DATE'.
# # It seems there is no direct 'qty_on_hand units' column in the dataset based on the initial view. We'll assume 'ATS units' is the relevant column for "quantity on hand".

# # Create 'month_year' column for plotting
# data_set_13['month_year'] = data_set_13['LAST REC DATE'].dt.to_period('M')

# # Group by 'month_year' and calculate the sum of 'qty_on_hand units'
# monthly_qty_on_hand = data_set_13.groupby('month_year')['qty_on_hand units'].sum().reset_index()

# # Convert 'month_year' back to datetime for plotting (necessary after grouping by period)
# monthly_qty_on_hand['month_year'] = monthly_qty_on_hand['month_year'].dt.to_timestamp()

# # Plotting the line graph
# plt.figure(figsize=(14, 7))
# plt.plot(monthly_qty_on_hand['month_year'], monthly_qty_on_hand['qty_on_hand units'], marker='o', linestyle='-', color='blue')
# plt.title('Monthly Quantity on Hand')
# plt.xlabel('Month-Year')
# plt.ylabel('Quantity on Hand Units')
# plt.xticks(rotation=45)
# plt.grid(True)
# plt.tight_layout()  # Adjust layout to make room for the rotated x-axis labels
# plt.show()











'''
    HERE, WE COMBINE TWO EXCEL SHEETS WITH A FOR-LOOP, WHICH IS THE AUTOMATIC WAY, AS OPPOSED TO THE MANUAL WAY, WHICH REQUIRES YOU TO KNOW THE SHEET NAMES IN ADVANCED!
'''


# import pandas as pd

# # Load the Excel file
# bar_sales_path = './CSVs/bar_sales.xlsx'
# bar_sales_data = pd.read_excel(bar_sales_path)

# # Load the Excel file and list all sheets
# xls = pd.ExcelFile(bar_sales_path)
# sheet_names = xls.sheet_names

# # Load all sheets into a dictionary of dataframes
# all_sheets_data = {sheet: pd.read_excel(xls, sheet_name=sheet) for sheet in sheet_names}

# # Display sheet names and the first few rows of each sheet to verify
# sheet_names, {name: df.head(1) for name, df in all_sheets_data.items()}

# # Concatenate data from both sheets into a single DataFrame
# combined_data = pd.concat([all_sheets_data[sheet] for sheet in sheet_names]).reset_index(drop=True)

# # Filtering out rows where 'Menu Group' is NaN
# combined_filtered_data = combined_data.dropna(subset=['Menu Group'])

# # Calculating average sales quantity per menu group
# combined_avg_sales_quantity = combined_filtered_data.groupby('Menu Group')['Item Qty'].mean().sort_values(ascending=False).head(5)

# # Calculating average net amount per menu group
# combined_avg_net_amount = combined_filtered_data.groupby('Menu Group')['Net Amount'].mean().sort_values(ascending=False).head(5)

# print(combined_avg_sales_quantity)
# print(combined_avg_net_amount)

















'''
    HERE IS A VERY SIMPLE SCRIPT THAT GENERATES THE DISTRIBUTION OF THE DIFFERENT TYPES OF DEGREEES AMONGST SOME POPULATION.
'''


# import pandas as pd
# import matplotlib.pyplot as plt

# # Load the dataset
# df_customers = pd.read_csv('./CSVs/simulated_customers.csv')

# # Count the frequency of each degree status
# degree_counts = df_customers['degree'].value_counts()

# # Create a bar plot
# plt.figure(facecolor='white')
# degree_counts.plot(kind='bar', color='skyblue')
# plt.title('Distribution of Customers by Degree')
# plt.xlabel('Degree')
# plt.ylabel('Number of Customers')
# plt.xticks(rotation=45)
# plt.tight_layout()
# plt.show()













'''
    HERE, WE NEEDED TO USE A DIFFERENT ENCODING AND THEN WE SEARCHED FOR SPECIFIC SUB-STRINGS WITHIN THE STRING ENTRIES OF A SPECIFIC COLUMN.
'''


# import pandas as pd

# # Load the hotel_room.csv file to examine its structure
# hotel_data_path = './CSVs/hotel_room.csv'

# # Attempting to read the CSV with a different encoding ('ISO-8859-1')
# hotel_data = pd.read_csv(hotel_data_path, encoding='ISO-8859-1')

# # Display the first few rows of the dataframe and its columns to understand its structure
# # hotel_data.head(), hotel_data.columns

# # Clean up the column names by stripping whitespace
# hotel_data.columns = hotel_data.columns.str.strip()

# # Display the cleaned column names
# # print(hotel_data.columns)

# # Filter the data for "Hilton Pattaya"
# hilton_pattaya_review_count = hotel_data[hotel_data['property name'].str.contains("Hilton Pattaya", case=False, na=False)]

# print(hilton_pattaya_review_count[['property name', 'review_count']])


















'''
    HERE WE CREAT A PRETTY COOL GRAPH WHERE EACH BAR IS SEGMENTED INTO THREE TIERS!
'''



# import matplotlib.pyplot as plt
# import pandas as pd

# # Load the data from the Excel file
# payroll_data_path = './CSVs/PAYROLL_MAY.xlsx'
# payroll_data = pd.read_excel(payroll_data_path)

# # Display the first few rows of the dataframe and its column names to understand its structure
# payroll_data.head(), payroll_data.columns

# # Calculate the maximum value in the "PLA_A_SCTR PENSIONS" column
# max_pension = payroll_data['PLA_A_SCTR PENSIONS'].max()

# # Define thresholds for the tiers
# tier_1_threshold = 0.30 * max_pension
# tier_2_threshold = 0.60 * max_pension

# # Function to determine the tier based on the pension value
# def assign_tier(pension):
#     if pension < tier_1_threshold:
#         return 'Tier 1'
#     elif pension < tier_2_threshold:
#         return 'Tier 2'
#     else:
#         return 'Tier 3'

# # Apply the function to create a new 'Tier' column
# payroll_data['Tier'] = payroll_data['PLA_A_SCTR PENSIONS'].apply(assign_tier)

# # Select relevant columns including the new 'Tier' column
# tiered_payroll_data = payroll_data[['Cod- unif', 'Employee', 'JOB', 'PLA_A_SCTR PENSIONS', 'GRAND TOTAL', 'YEAR', 'Tier']]

# # Display the first few rows of the modified dataframe
# tiered_payroll_data.head(), tiered_payroll_data['Tier'].value_counts()

# # Group the data by 'YEAR' and 'Tier', and sum the 'GRAND TOTAL' for each group
# grouped_data = tiered_payroll_data.groupby(['YEAR', 'Tier'])['GRAND TOTAL'].sum().unstack(fill_value=0)

# # Create a stacked bar chart
# fig, ax = plt.subplots(figsize=(12, 8))
# grouped_data.plot(kind='bar', stacked=True, ax=ax, color=['#1f77b4', '#ff7f0e', '#2ca02c'])  # colors for Tier 1, Tier 2, Tier 3

# ax.set_title('Grand Total Contributions by Tier per Year', fontsize=15)
# ax.set_xlabel('Year', fontsize=12)
# ax.set_ylabel('Grand Total', fontsize=12)
# ax.legend(title='Tier', title_fontsize='13', fontsize='11')

# plt.show()













'''
    HERE, WE SIMPLY COMBINE TWO COLUMNS TO MAKE ONE AND THE ENTRIES ARE STRINGS.
'''



# import pandas as pd

# # Load the dataset
# file_path = './CSVs/Invoices Dic - Facturas.tsv'
# invoices_data = pd.read_csv(file_path, delimiter='\t')

# # Combine 'Payment Status' and 'Invoice Status' into a single column
# invoices_data['Payment - Invoice Status'] = invoices_data['Payment Status'] + ' - ' + invoices_data['Invoice Status']

# # Display the first few rows to confirm the changes
# print(invoices_data.head(n=15))


















'''
    HERE, WE EXPLORE AN EXCEL SHEET WITH MULTIPLE SHEETS. THIS IS THE PROPER WAY TO OPEN IT AND GET THE SHEET NAMES! WE EXPLORE SEVERAL OF THESE SHEETS AND DEMO HOW TO STORE THEM INDIVIDUALLY AND OPERATE ON THEM.
'''


# import pandas as pd

# # Load the Excel file
# file_path = './CSVs/grades.xlsx'
# xl = pd.ExcelFile(file_path)

# # Check the sheet names to understand the structure
# sheet_names = xl.sheet_names
# print(sheet_names)

# # Load the "Period 2" sheet data
# period_2_data = pd.read_excel(file_path, sheet_name='Period 2')

# # Display the first few rows of the dataframe to inspect it and count the columns
# print(period_2_data.head())
# print(period_2_data.shape[1])

# # Load the "Period 4" sheet data
# period_4_data = pd.read_excel(file_path, sheet_name='Period 4')

# # Find student A's midterm score
# student_a_midterm_score = period_4_data.loc[period_4_data['Student'] == 'A', 'Midterm'].iloc[0]
# print(student_a_midterm_score)














'''
    HERE IS SOME INTERESTING FUNCTION THAT YOU SHOULD LEARN HOW TO USE! THERE IS TWO DIFF ATTEMPTS BELOW
'''

'''ATTEMPT1'''
# import pandas as pd

# pd.set_option('display.max_rows', None)
# pd.set_option('display.max_columns', None)

# # Read the CSV file into a DataFrame
# df = pd.read_csv('./CSVs/blizzard_games.csv')



# def get_top_people(df: pd.DataFrame, col_name: str) -> pd.DataFrame:
#   """
#   Splits the strings in the specified column by commas and spaces, explodes the list to have one row per name,
#   groups by name, counts the occurrences, and sorts by frequency in descending order.

#   Args:
#       df (pd.DataFrame): The DataFrame containing the data.
#       col_name (str): The name of the column to process.

#   Returns:
#       pd.DataFrame: A DataFrame with the top people and their frequencies.
#   """

#   return (
#       df[col_name]
#       .astype(str)
#       .str.split(r",\s*|\s+")
#       .explode()
#       .groupby(level=0)
#       .size()
#       .sort_values(ascending=False)
#   )


# columns_to_analyze = [
#     "Developer(s)",
#     "Publisher(s)",
#     "Producer(s)",
#     "Programmer(s)",
#     "Artist(s)",
#     "Composer(s)",
#     "Designer(s)",
#     "Writer(s)",
#     "Director(s)",
# ]

# for col in columns_to_analyze:
#   print(f"\n### Top 5 {col.replace('(s)', '')}\n")
#   print(get_top_people(df, col).head())






'''ATTEMPT2'''
# from collections import Counter

# # Helper function to clean and split the entries which are lists in string form
# def clean_and_count(column_data):
#     # Join all non-null data, split by ',' to handle lists, strip extra characters and count occurrences
#     all_entries = column_data.dropna().apply(lambda x: x.strip("[]'").split("', '")).explode()
#     return Counter(all_entries).most_common(5)

# # Top 5 contributors for each role
# top_developers = clean_and_count(df['Developer(s)'])
# top_publishers = clean_and_count(df['Publisher(s)'])
# top_producers = clean_and_count(df['Producer(s)'])
# top_programmers = clean_and_count(df['Programmer(s)'])
# top_artists = clean_and_count(df['Artist(s)'])
# top_composers = clean_and_count(df['Composer(s)'])
# top_designers = clean_and_count(df['Designer(s)'])
# top_writers = clean_and_count(df['Writer(s)'])
# top_directors = clean_and_count(df['Director(s)'])

# # Converting to DataFrames for easy display
# df_top_developers = pd.DataFrame(top_developers, columns=['Developer', 'Count'])
# df_top_publishers = pd.DataFrame(top_publishers, columns=['Publisher', 'Count'])
# df_top_producers = pd.DataFrame(top_producers, columns=['Producer', 'Count'])
# df_top_programmers = pd.DataFrame(top_programmers, columns=['Programmer', 'Count'])
# df_top_artists = pd.DataFrame(top_artists, columns=['Artist', 'Count'])
# df_top_composers = pd.DataFrame(top_composers, columns=['Composer', 'Count'])
# df_top_designers = pd.DataFrame(top_designers, columns=['Designer', 'Count'])
# df_top_writers = pd.DataFrame(top_writers, columns=['Writer', 'Count'])
# df_top_directors = pd.DataFrame(top_directors, columns=['Director', 'Count'])

# print(f"\n{df_top_developers}\n {df_top_publishers}\n {df_top_producers}\n {df_top_programmers}\n {df_top_artists}\n {df_top_composers}\n {df_top_designers}\n {df_top_writers}\n {df_top_directors}")














'''
    HERE IS A COOL WAY OF SPLITTING STRINGS WITHIN ENTRIES. WE TAKE AN ENTRY, WHICH IS A STRING, AND SPLIT IT INTO TWO STRINGS EFFECTIVELY MAKING TWO DISTINCT COLUMNS OUT OF ONE.
'''


# import pandas as pd

# # Load the Excel file
# file_path = './CSVs/SOLDFOOD2023 - Fall.xlsx'
# data = pd.read_excel(file_path)

# # Display the first few rows of the dataframe and the column names to understand its structure
# data.head(), data.columns

# # It appears that the actual headers may be in the third row of the dataset (index 2 in zero-indexed Python)
# # We'll reload the data with the correct header row.

# data_corrected = pd.read_excel(file_path, header=2)
# data_corrected.head(), data_corrected.columns


# # Renaming columns based on the data in the first row
# new_headers = data_corrected.iloc[0]
# data_clean = data_corrected[1:]
# data_clean.columns = new_headers

# # Resetting the index for cleanliness
# data_clean.reset_index(drop=True, inplace=True)

# # Display the updated data
# data_clean.head(), data_clean.columns


# # Convert all entries in 'DESCRIPTION' to strings and perform the split operation
# data_clean['DESCRIPTION'] = data_clean['DESCRIPTION'].astype(str)
# data_clean['Style'] = data_clean['DESCRIPTION'].apply(lambda x: ' '.join(x.split()[:-1]))
# data_clean['Product Category'] = data_clean['DESCRIPTION'].apply(lambda x: x.split()[-1])

# # Display the updated DataFrame to verify the new columns
# print(data_clean[['DESCRIPTION', 'Style', 'Product Category']].head())





'''ATTEMPT2: (PROPER WAY)'''
# import pandas as pd

# # Load the Excel file
# file_path = './CSVs/SOLDFOOD2023 - Fall.xlsx'

# # Attempt to load the Excel file again with a more precise header row adjustment
# df_adjusted = pd.read_excel(file_path, header=3)

# # Adjust the split_description function to handle non-string values safely
# def split_description_safe(desc):
#     if pd.isna(desc):
#         return '', ''  # Handle NaN values
#     desc_str = str(desc)  # Ensure the description is treated as a string
#     parts = desc_str.split()
#     if len(parts) > 1:
#         return " ".join(parts[:-1]), parts[-1]
#     else:
#         return desc_str, ''

# # Apply the function to split 'DESCRIPTION' into 'Style' and 'Product Category' with the correct DataFrame
# df_adjusted['Style'], df_adjusted['Product Category'] = zip(*df_adjusted['DESCRIPTION'].apply(split_description_safe))

# # Show the updated DataFrame to verify the changes
# print(df_adjusted[['DESCRIPTION', 'Style', 'Product Category']].head())


















'''
    HERE, WE CREATE A COOL VISUALIZATION WITH PLENTY OF CONSTRAINTS!

    1. Split the 'income' column to income classes from 0-2500, 2501-5000 and so on as a new column named as 'income classes'. 

    2.1. Generate a heatmap to display the correlation between the 'relationship' column, the 'gender' column, the 'children' column and the newly calculated  'income classes' column.

    2.2. The plot has to be of size 15x15. The title should be Correlation Heatmap of Customers. Enough padding and spacing should be present to have legible x-axis and y-axis labels. This should be plotted using the matplotlib library and the code for both the pandas manipulation and the final plot result should be provided.
'''



# import pandas as pd
# import matplotlib.pyplot as plt
# import seaborn as sns

# # Read the CSV file into a DataFrame
# df = pd.read_csv('./CSVs/simulated_customers.csv')

# # Display the first 5 rows
# print(df.head().to_string(index=False))

# # Print the column names and their data types
# print(df.info())

# # Function to categorize income
# def income_group(income):
#   if income <= 2500:
#     return '0-2500'
#   elif income <= 5000:
#     return '2501-5000'
#   elif income <= 7500:
#     return '5001-7500'
#   elif income <= 10000:
#     return '7501-10000'
#   elif income <= 12500:
#     return '10001-12500'
#   else:
#     return '12501+'

# # Create a new column by applying the income_group function to the income column
# df['income_classes'] = df['income'].apply(income_group)

# # Drop the `income` column
# df = df.drop('income', axis=1)

# # Select columns of interest
# columns_to_select = ['relationship', 'gender', 'children', 'income_classes']
# df_selected = df[columns_to_select]

# # Create dummy variables
# df_dummies = pd.get_dummies(df_selected)

# # Calculate the correlation matrix
# corr = df_dummies.corr()

# # Create the heatmap
# plt.figure(figsize=(15, 15))
# sns.heatmap(corr, annot=True, fmt='.2f', cmap='coolwarm')
# plt.title('Correlation Heatmap of Customers', fontsize=14)
# plt.xticks(rotation=45, ha='right', fontsize=12)
# plt.yticks(fontsize=12)
# plt.tight_layout()

# # Show the plot
# plt.show()













'''
    HERE IS A CLEVER TYPE OF VISUALIZATION THAT FINDDS AND COUNTS THE NUMBER OF UNIQUE COMBINATIONS OF THREE VARIABLES AND PLOTS THEM IN A HEATMAP.
'''



# import pandas as pd
# import matplotlib.pyplot as plt
# import seaborn as sns

# # Read the CSV file into a DataFrame
# df = pd.read_csv('./CSVs/patient_list-_patient_list.csv')

# # Create age groups
# bins = [0, 30, 40, 100]
# labels = ['Under 30', '30 to 40', 'Over 40']
# df['age_group'] = pd.cut(df['age'], bins=bins, labels=labels, right=False)
# # print(df['age_group'])

# # Group data by age group, severity score, and frequency, then count the occurrences
# session_counts = df.groupby(['age_group', 'severity_score', 'frequency']).size().reset_index(name='Count')
# print(session_counts)

# # Calculate the total number of sessions
# total_sessions = session_counts['Count'].sum()

# # Calculate the proportion of each combination
# session_counts['Proportion'] = session_counts['Count'] / total_sessions

# # Aggregate data by `age_group`, `severity_score`, summing up the `Count` and `Proportion`
# session_counts_agg = session_counts.groupby(['age_group', 'severity_score'])[['Count', 'Proportion']].sum().reset_index()

# # Pivot the data to create a heatmap-friendly format
# heatmap_data = session_counts_agg.pivot(index='age_group', columns='severity_score', values='Proportion').fillna(0)

# # Create the heatmap using Seaborn
# plt.figure(figsize=(10, 8))
# sns.heatmap(heatmap_data, annot=True, fmt=".2%", cmap="YlGnBu", cbar_kws={'label': 'Proportion of Sessions'})
# plt.title('Distribution of Session Frequency by Age Group and Severity Score')
# plt.ylabel('Age Group')
# plt.xlabel('Severity Score')
# plt.show()

# Print the pivot table
# print(heatmap_data)

















# import pandas as pd

# Load the data from the Excel file
# hospital_survey_data = pd.read_excel('./CSVs/Hospital_Survey_Data_Alcohol_Drug_Abuse.xlsx', header=1)

# # Display the first few rows and the columns to understand the structure of the data
# print(hospital_survey_data.head())
# print(hospital_survey_data.columns)

# # Define a function to categorize the DRG Definition
# def categorize_drg(description):
#     if 'W/O REHABILITATION THERAPY' in description:
#         return 'Without Rehabilitation Therapy'
#     elif 'W REHABILITATION THERAPY' in description:
#         return 'With Rehabilitation Therapy'
#     elif 'LEFT AMA' in description:
#         return 'Left AMA'
#     else:
#         return 'Other'

# # Apply the function to create a new column for the categorization
# hospital_survey_data['Therapy Status'] = hospital_survey_data['DRG Definition'].apply(categorize_drg)

# # Display the updated DataFrame to verify the new column
# print(hospital_survey_data[['DRG Definition', 'Therapy Status']].head())



















'''
    HERE WE CREATE A NEW DATASET WITH NEW COLUMNS AND SAVE THE NEW SET TO A TSV FILE.
'''


# import pandas as pd

# # Load the dataset
# file_path = './CSVs/last60.csv'
# data = pd.read_csv(file_path)

# # Display the first few rows of the dataset to understand its structure
# print(data.head())

# # Calculate required values
# grouped_data = data.groupby('Brand').agg(
#     Total_Cost = pd.NamedAgg(column = 'Cost', aggfunc = 'sum'),
#     Average_SuggestedRetail = pd.NamedAgg(column = 'SuggestedRetail', aggfunc = 'mean')
# )

# # Create a combined dimension column
# data['Dimension'] = data['Length'].astype(str) + "x" + data['Width'].astype(str) + "x" + data['Height'].astype(str)

# # Get the most common dimension for each brand
# most_common_dimension = data.groupby('Brand')['Dimension'].agg(lambda x: x.mode().iloc[0])
# grouped_data = grouped_data.join(most_common_dimension)

# Save to a TSV file
# output_file_path = './CSVs/Totals.csv'
# grouped_data.to_csv(output_file_path, sep='\t', index=True)

# print(grouped_data.head())

















'''
    HERE WE CREATE A COOL HEAT MAP BUT WE USE A NEW METHOD TO SPLIT UP AND CATEGORIZE THE AGE GROUPS AUTOMATICALLY.
'''



# import pandas as pd
# import matplotlib.pyplot as plt
# import seaborn as sns

# # Load the Excel file to examine its contents
# excel_path = './CSVs/LIFE INS ISSUE AGE AUDIT.xlsx'
# life_ins_data = pd.read_excel(excel_path)

# # Display the first few rows and the column names
# print(life_ins_data.head())
# print(life_ins_data.columns)

# # Create `Age Category` column using quantile-based discretization
# life_ins_data['Age Category'] = pd.qcut(life_ins_data['Issue Age'], 3, labels=['Young', 'Middle', 'Old'])

# # Group the data by 'Issue Age' and 'Additional Insurance with Company' and calculate the mean 'Mode Premium'
# grouped_data = life_ins_data.groupby(['Age Category', 'Additional Insurance with Company']).agg({'Mode Premium': 'mean'}).reset_index()

# # Pivot the table to create a heatmap format
# heatmap_data = grouped_data.pivot(index='Age Category', columns='Additional Insurance with Company', values='Mode Premium')

# # Create the heatmap
# plt.figure(figsize=(10, 8))
# sns.heatmap(heatmap_data, annot=True, fmt=".2f", cmap='YlGnBu', linewidths=.5)
# plt.title('Heatmap of Average Mode Premium by Issue Age and Additional Insurance')
# plt.xlabel('Additional Insurance with Company')
# plt.ylabel('Issue Age')
# plt.show()













'''
    HERE WE USE A NEW METHOD TO SELECT THE LARGEST 'n' VALUES IN AN ARBITRARY COLUMN.
'''

# import pandas as pd

# # Load the data from the uploaded CSV file
# file_path = './CSVs/business_unit_system_cash_flow.csv'
# cash_flow_data = pd.read_csv(file_path)

# # Display the first few rows of the dataset to understand its structure
# print(cash_flow_data.head())

# # Aggregate total income, expenses, and net cash flow by business unit
# business_unit_summary = cash_flow_data.groupby('Unidad de Negocio').agg(
#     Total_Ingresos=pd.NamedAgg(column='Ingresos', aggfunc='sum'),
#     Total_Egresos=pd.NamedAgg(column='Egresos', aggfunc='sum'),
#     Total_Neto=pd.NamedAgg(column='Total', aggfunc='sum')
# ).reset_index()

# business_unit_summary.sort_values(by='Total_Neto', ascending=False)

# # Finding the business units with the highest income
# top_income_units = business_unit_summary.nlargest(3, 'Total_Ingresos')

# # Finding the business units with the highest expenses
# top_expense_units = business_unit_summary.nlargest(3, 'Total_Egresos')

# print(top_income_units) 
# print(top_expense_units)





















'''
    HERE WE CLEAN UP A UNIQUE DATASET WITH ENTRIES DELIMITED BY PIPES WHICH MEANS THAT THE EXCEL SHEET WAS A SINGLE COLUMN, I.E., THE DATA SET IS ORIGINALLY A NX1 MATRIX. THIS MEANT WE HAD TO GO IN AND EXCTRACT EACH ENTRY AND CREATE NEW COLUMNS FOR THEM. WE ALSO HAD TO SKIP AN ARBITRARY ROW.
'''


# import pandas as pd
# import matplotlib.pyplot as plt

# # Load the data from the uploaded XLSX file
# xlsx_file_path = './CSVs/DynamicBiz_insight.xlsx'

# # Load the Excel file without attempting to split columns initially
# insight_data_raw = pd.read_excel(xlsx_file_path, header=None)

# # Split the single column into multiple columns by '|'
# split_data = insight_data_raw[0].str.split('|', expand=True)

# # Use the first row as header
# split_data.columns = split_data.iloc[0].apply(lambda x: x.strip())
# split_data = split_data[2:]  # Remove the header row from the data

# # Clean up the data by trimming whitespace
# split_data = split_data.map(lambda x: x.strip() if isinstance(x, str) else x)

# print(split_data.head(n=20))

# # Re-examine the data types and ensure correct conversion and filtering
# split_data['Units Sold'] = pd.to_numeric(split_data['Units Sold']) 

# # Drop rows with NaN values in 'Units Sold' which result from coercion
# split_data = split_data.dropna(subset=['Units Sold'])

# # Aggregate total 'Units Sold' by 'Region' again
# units_per_region = split_data.groupby('Region')['Units Sold'].sum()

# # Plotting the bar chart
# plt.figure(figsize=(10, 6))
# units_per_region.plot(kind='bar', color='skyblue')
# plt.title('Total Units Sold Per Region')
# plt.xlabel('Region')
# plt.ylabel('Units Sold')
# plt.xticks(rotation=45)
# plt.grid(axis='y', linestyle='--', alpha=0.7)
# plt.tight_layout()
# plt.show()




















'''
    HERE... WE HAVE A MUTI-SHEET EXCEL FILE WHERE WE HAD TO COMBINE ALL SHEETS THEN EXTRACT STATS BASED ON THE ENTIRE DATASET. THE COLOMN NAMES REPRESENT THE SAME THING BUT THEY WERE NAMED SLIGHTLY DIFFERENT SO WE HAD TO STANDARDIZE THE NAMES AS WELL.
'''
# import pandas as pd

# # Path to your Excel file
# file_path = './CSVs/population_and_age_1.xlsx'

# # Load the Excel file
# xls = pd.ExcelFile(file_path)
# all_data = []

# # Function to find and standardize column names
# def standardize_col_names(df):
#     for col in df.columns:
#         if 'age' in col.lower():
#             df.rename(columns={col: 'Age'}, inplace=True)
#         elif 'population' in col.lower():
#             df.rename(columns={col: 'Population'}, inplace=True)
#         elif 'country' in col.lower():
#             df.rename(columns={col: 'Country'}, inplace=True)
#     return df

# # Process each sheet
# for sheet_name in xls.sheet_names:
#     data = pd.read_excel(xls, sheet_name=sheet_name)
    
#     # Standardize column names
#     data = standardize_col_names(data)
    
#     # Append the DataFrame to the list
#     all_data.append(data)

# # Concatenate all DataFrames into one
# combined_data = pd.concat(all_data)

# # Calculate the average age and total population
# average_age = combined_data['Age'].mean()
# total_population = combined_data['Population'].mean()

# print(f"Average Age: {average_age}, Average Population: {total_population}")


                  


                  












'''
    HERE WE HAVE THREE DIFFERENT WAYS OF GETTING THE DISTRIBUTION OF A COLUMN AND WE HAVE TWO DIFFERENT WAYS OF PLOTTING IT.
'''


# import pandas as pd
# import matplotlib.pyplot as plt

# # Loading the CSV file
# ma_population_df = pd.read_csv('./CSVs/population_ma.csv')
# ma_population_df['Population'] = ma_population_df['Population'].str.replace(',', '').astype(int)
# ma_population_df['City'] = ma_population_df['City'].astype(str)

# # Displaying the first few rows of the dataframe to understand its structure
# print(ma_population_df.head())

# # Calculate the population distribution for each city
# ma_population_df['Distribution'] = ma_population_df['Population']/ma_population_df['Population'].sum() * 100

# Display the distribution in descending order
# print(ma_population_df[['City', 'Distribution']].sort_values(by='Distribution', ascending=False))
                 

# Plotting the population distribution for cities in Massachusetts
# plt.figure(figsize=(20, 6), facecolor='white')
# plt.bar(ma_population_df['City'], ma_population_df['Population'], color='blue')
# plt.xlabel('City')
# plt.ylabel('Population')
# plt.title('Population Distribution in Massachusetts Cities')
# plt.xticks(rotation=90)
# plt.tight_layout()
# plt.show()


# # Assuming the columns are named 'City' and 'Population'
# population_distribution = ma_population_df.groupby('City')['Population'].sum()

# # Plotting the population distribution
# plt.figure(figsize=(30, 8))
# population_distribution.sort_values(ascending=False).plot(kind='bar')
# plt.title('Population Distribution for Cities in Massachusetts')
# plt.xlabel('City')
# plt.ylabel('Population')
# plt.xticks(rotation=45)
# plt.show()





















'''
    HERE WE SHOW HOW TO LOOP OVER COLUMNS AND PRINT THEIR ENTRIES
'''
# import pandas as pd

# # Load the data
# data_path = './CSVs/car_price_prediction.csv'
# car_price_data = pd.read_csv(data_path)
# most_expensive_category = car_price_data.groupby('Category')['Price'].mean().idxmax()

# # This creates a subset with two columns: category and the avg price
# average_prices_by_category = car_price_data.groupby('Category')['Price'].mean()
# print(average_prices_by_category.head(n=15))

# # Sort the averages in descending order and get the top 3
# top_three_categories = average_prices_by_category.sort_values(ascending=False).head(3)

# print("Top 3 most expensive car categories and their average prices:")
# for category, avg_price in top_three_categories.items():
#     print(f"{category}: ${avg_price:,.2f}")

















'''
    HERE, WE USE THE CONCAT METHOD TO COMBINE TWO SHEETS OF A MULTI-SHEET EXCEL FILE.
'''


# import pandas as pd
# import matplotlib.pyplot as plt
# import seaborn as sns

# # Loading both sheets from the Excel file
# sheet1_df = pd.read_excel('./CSVs/bar_sales.xlsx', sheet_name=0)
# sheet2_df = pd.read_excel('./CSVs/bar_sales.xlsx', sheet_name=1)

# # Combining both sheets into one DataFrame
# combined_sales_df = pd.concat([sheet1_df, sheet2_df], ignore_index=True)

# # Plotting the spread of quantities sold per menu group
# plt.figure(figsize=(12, 6), facecolor='white')
# sns.boxplot(x='Menu Group', y='Item Qty', data=combined_sales_df)
# plt.title('Spread of Quantities Sold Per Menu Group')
# plt.xlabel('Menu Group')
# plt.ylabel('Quantity Sold')
# plt.xticks(rotation=45)
# plt.tight_layout()
# plt.show()

# # Calculating range, standard deviation, and mean for each menu group
# stats_df = combined_sales_df.groupby('Menu Group')['Item Qty'].agg(['min', 'max', 'std', 'mean'])
# stats_df['range'] = stats_df['max'] - stats_df['min']

# # Sorting by mean
# sorted_stats_df = stats_df.sort_values(by='mean', ascending=False)

# # Displaying the sorted statistics
# print(sorted_stats_df)





























'''
    HERE WE CREATE A PLOT WITH THREE DIFFERENT Y-AXIS VARIABLES SUPERIMPOSED ONTO ONE ANOTHER TO CREATE A REALLY COOL GRAPH THATS VERY INFORMATIVE.
'''


# import pandas as pd
# import matplotlib.pyplot as plt

# # Load the dataset
# file_path = './CSVs/Top_1000_Bollywood_Movies.csv'
# bollywood_data = pd.read_csv(file_path)
# print(bollywood_data.head())
# print(bollywood_data.info())

# # Filter data for 'All Time Blockbuster' or 'Blockbuster' verdicts
# filtered_data = bollywood_data[bollywood_data['Verdict'].isin(['All Time Blockbuster', 'Blockbuster'])]

# # Sort the filtered data by 'India Net' earnings in descending order
# sorted_filtered_data = filtered_data.sort_values(by='India Net', ascending=False)

# # Select the top 10 highest-grossing movies
# top_10_movies = sorted_filtered_data.head(10)

# # NOTE: is this line needed here? What is it doing?
# top_10_movies[['Movie', 'Worldwide', 'India Net', 'Budget', 'Verdict']]

# # Recreate the plot with corrected legend handling and x-tick labels
# fig, ax1 = plt.subplots(figsize=(12, 8))

# # Names of the movies
# movies = top_10_movies['Movie']

# # First axis for 'Worldwide' earnings
# color = 'tab:red'
# ax1.set_xlabel('Movie')
# ax1.set_ylabel('Worldwide Earnings (Billion INR)', color=color)
# lns1 = ax1.bar(movies, top_10_movies['Worldwide']/1e9, color=color, label='Worldwide Earnings', alpha=0.6)
# ax1.tick_params(axis='y', labelcolor=color)
# ax1.set_xticks(movies.index)
# ax1.set_xticklabels(movies, rotation=45, ha='right')

# # Second axis for 'India Net' earnings
# ax2 = ax1.twinx()
# color = 'tab:blue'
# ax2.set_ylabel('India Net Earnings (Billion INR)', color=color)
# lns2 = ax2.plot(movies, top_10_movies['India Net']/1e9, color=color, label='India Net Earnings', marker='x')
# ax2.tick_params(axis='y', labelcolor=color)

# # Third axis for 'Budget'
# ax3 = ax1.twinx()
# color = 'tab:green'
# ax3.set_ylabel('Budget (Billion INR)', color=color)
# lns3 = ax3.plot(movies, top_10_movies['Budget']/1e9, color=color, label='Budget', marker='o')
# ax3.tick_params(axis='y', labelcolor=color)
# ax3.spines['right'].set_position(('outward', 60))

# # Convert plot handles to a list for the legend
# lns = list(lns1) + list(lns2) + list(lns3)
# labs = [l.get_label() for l in lns]
# ax1.legend(lns, labs, loc=0)

# plt.title('Top 10 Highest-Grossing Bollywood Movies (Sorted by India Net Earnings)')
# plt.show()





















'''
    HERE WE GET THE MAX AND MIN DATES, THEN DISPLAY EVERY ENTRY THAT HAS A TRANSACTION ON THOSE DATES.
'''
# import pandas as pd

# # Load the Excel file
# file_path = './CSVs/FAL Projects NY - office NY - FAL Proyectos.xlsx'

# # Reload the data starting from row 9 (indexing starts from 0, so we use header=9)
# # The 1st 10 rows are not useful data!
# data = pd.read_excel(file_path, header=9)
# # print(data.head(n=15))
# # print(data.info(verbose=True))

# # Convert date column to datetime type and find the earliest and latest dates
# data['Create Date:'] = pd.to_datetime(data['Create Date:'])
# first_date = data['Create Date:'].min()
# last_date = data['Create Date:'].max()

# # Filter the transactions that occurred on the first and last date
# first_date_transactions = data[data['Create Date:'] == first_date]
# last_date_transactions = data[data['Create Date:'] == last_date]

# print(first_date) 
# print(last_date) 
# print(first_date_transactions) 
# print(last_date_transactions)

















# import pandas as pd
# import matplotlib.pyplot as plt
# import seaborn as sns

# # Load the TSV file
# inventory_data = pd.read_csv("./CSVs/inventory-snapshot-table_2024-01-15_ 01 - sheet1.tsv", sep='\t')

# Display the first few rows and the data structure to identify relevant columns
# inventory_data_head = inventory_data.head()
# inventory_data_info = inventory_data.info()
# print("\nThese are the first few rows:\n")
# print(f'Head:\n{inventory_data_head}') 
# print(f'Info:\n{inventory_data_info}')

# # Filter data for the 'Earrings' product category
# earrings_inventory = inventory_data[inventory_data['Product category'].str.contains('Earrings')]

# # Correcting the column names to remove any extra spaces
# earrings_inventory.columns = earrings_inventory.columns.str.strip()

# # Cleaning and converting 'Total value' to numerical for further analysis
# # NOTE: find out why this throws a warning...
# earrings_inventory['Total value'] = earrings_inventory['Total value'].str.replace(' €', '').astype(float)

# # Summary statistics and a glimpse of 'Earrings' inventory data
# earrings_inventory_describe = earrings_inventory.describe()
# earrings_inventory_head = earrings_inventory.head()
# print("\n\nSummary stats:\n")
# print(earrings_inventory_describe) 
# print(earrings_inventory_head)

# Setting up the visualizations
# plt.figure(figsize=(14, 6))

# plt.subplot(1, 2, 1)
# sns.histplot(earrings_inventory['Inventory'], bins=10, kde=True)
# plt.title('Distribution of Inventory Levels')
# plt.xlabel('Inventory')
# plt.ylabel('Frequency')

# plt.subplot(1, 2, 2)
# sns.histplot(earrings_inventory['Total value'], bins=10, kde=True)
# plt.title('Distribution of Total Value of Inventory')
# plt.xlabel('Total Value (in €)')
# plt.ylabel('Frequency')

# plt.tight_layout()
# plt.show()




















'''
    HERE WE HAVE TO PARSE ENTRIES INTO THREE DISTINCT CATEGORIES SUCH EACH CATEGORY GETS TRANSFORMED INTO ITS OWN COLUMN.
'''

'''
    APPROACH 0: CONDITIONAL EXTRACTION
'''
# import pandas as pd

# # Reload the data with the correct header
# hospital_data = pd.read_excel('./CSVs/Hospital_Survey_Data_Alcohol_Drug_Abuse.xlsx', header=1)

# # Display the corrected column headers and the first few rows to verify
# print(hospital_data.columns)
# print(hospital_data.head())
# print(hospital_data.info(verbose=True))

# NOTE: FIND OUT WHY THIS SPLITTING APPROACH DIDNT WORK......
# # Split the 'DRG Definition' column into new categories
# hospital_data[
#     ['Alcohol/Drug abuse with rehab', 'Alcohol/Drug abuse without rehab', 'Left AMA']
#     ] = hospital_data['DRG Definition'].str.extract(
#         '(.*REHABILITATION THERAPY.*)|(.*W/O REHABILITATION THERAPY.*)|(.*LEFT AMA.*)'
#         )

# # Display the first 10 rows to confirm the split
# print(hospital_data[['Alcohol/Drug abuse with rehab', 'Alcohol/Drug abuse without rehab', 'Left AMA']].head(10))





# import pandas as pd

# pd.set_option('display.max_rows', None)
# pd.set_option('display.max_columns', None)

# # Reload the data with the correct header
# df_alcohol_drug_abuse = pd.read_excel('./CSVs/Hospital_Survey_Data_Alcohol_Drug_Abuse.xlsx', header=1)

'''
    HERE IS A COOL WAY OF GETTING A COLUMN'S UNIQUE VALUES.
'''
# Inspect the unique values in the "DRG Definition" column to understand how to split the data
# print(df_alcohol_drug_abuse['DRG Definition'].unique())

# Display the first 5 rows
# print(df_alcohol_drug_abuse.head())

# Print the column names and their data types
# print(df_alcohol_drug_abuse.info())
                  
'''
    APPROACH 1: MANUALLY
'''
# # Create new columns and initialize with 0
# new_columns = ['Alcohol/Drug Abuse with Rehabilitation Therapy', 
#                'Alcohol/Drug Abuse without Rehabilitation Therapy', 
#                'Alcohol/Drug Abuse, Left AMA']
# df_alcohol_drug_abuse[new_columns] = 0

# # Populate the new columns based on 'DRG Definition'
# df_alcohol_drug_abuse['Alcohol/Drug Abuse with Rehabilitation Therapy'] = df_alcohol_drug_abuse['DRG Definition'].str.contains('W REHABILITATION THERAPY').astype(int)
# df_alcohol_drug_abuse['Alcohol/Drug Abuse without Rehabilitation Therapy'] = df_alcohol_drug_abuse['DRG Definition'].str.contains('W/O REHABILITATION THERAPY').astype(int)
# df_alcohol_drug_abuse['Alcohol/Drug Abuse, Left AMA'] = df_alcohol_drug_abuse['DRG Definition'].str.contains('LEFT AMA').astype(int)

# # print(df_alcohol_drug_abuse.head(n=20))

# # Aggregate the data at the provider level
# df_agg = df_alcohol_drug_abuse.groupby('Provider Name')[new_columns].sum().reset_index()

# # Sort the aggregated data
# df_agg['Total'] = df_agg[new_columns].sum(axis=1)
# df_agg_sorted = df_agg.sort_values('Total', ascending=False)

# # Print the aggregated and sorted data
# print(df_agg_sorted.head(10).to_markdown(index=False))

'''
    APPROACH 2: LAMBDA λ
'''
# Create three new columns based on the categories in the DRG Definition
# df_alcohol_drug_abuse['Without Rehab Therapy'] = df_alcohol_drug_abuse['DRG Definition'].apply(lambda x: 'Yes' if 'W/O REHABILITATION THERAPY' in x else 'No')
# df_alcohol_drug_abuse['Left AMA'] = df_alcohol_drug_abuse['DRG Definition'].apply(lambda x: 'Yes' if 'LEFT AMA' in x else 'No')
# df_alcohol_drug_abuse['With Rehab Therapy'] = df_alcohol_drug_abuse['DRG Definition'].apply(lambda x: 'Yes' if 'W REHABILITATION THERAPY' in x else 'No')

# # Display the first 10 rows with the new columns
# print(df_alcohol_drug_abuse[
#     ['DRG Definition', 'Without Rehab Therapy', 'Left AMA', 'With Rehab Therapy']
#     ].head(10))










                  











                  




'''
    HERE WE CREATE COOL SCATTER PLOT USING FOUR VARIABLES! THE AXES ARE X = DAYS AND Y = PRICE WHILE EACH BUBBLE IS A TYPE OF INVENTORY AND THE BUBBLE'S SIZE IS HOW MUCH OF THAT INVENTORY IS CURRENTLY ON HAND, I.E., AVAILABLE. THE MORE UNITS ON HAND, THEN THE BIGGER THE BUBBLE. WE TAKE TWO DIFFERENT APPROACHES WHICH SIMPLY USE TWO DIFFERENT MODULES.
'''




'''
    APPROACH 1: MATPLOTLIB PLOT 
'''

# import pandas as pd
# import matplotlib.pyplot as plt
# import numpy as np

# # Load the dataset from the provided CSV file
# inventory_data = pd.read_csv('./CSVs/Data Set #14 report .csv')

# # Remove '$' and ',' from the sell_price, AGE OF INVENTORY DAYS, and the qty_on_hand units column
# inventory_data['sell_price'] = inventory_data['sell_price'].replace('[\$,]', '', regex=True).astype(float)
# inventory_data['AGE OF INVENTORY DAYS'] = inventory_data['AGE OF INVENTORY DAYS'].str.replace(',', '').astype(int)
# inventory_data['qty_on_hand units'] = inventory_data['qty_on_hand units'].str.replace(',', '').astype(int)

# # Get unique inventory types
# inventory_types = inventory_data['inventory_type'].unique()

# # Generate random colors
# colors = np.random.rand(len(inventory_types), 3)

# # Create color dictionary
# color_dict = dict(zip(inventory_types, colors))

# # Create the scatter plot
# fig, ax = plt.subplots(figsize=(12, 8))
# for inventory_type in inventory_types:
#     type_data = inventory_data[inventory_data['inventory_type'] == inventory_type]
#     ax.scatter(type_data['AGE OF INVENTORY DAYS'], 
#                type_data['sell_price'], 
#                s = (type_data['qty_on_hand units'] * 0.5), 
#                label = inventory_type, 
#                alpha = 0.7, 
#                color = color_dict[inventory_type])

# # Add labels and title
# ax.set_xlabel('AGE OF INVENTORY DAYS', fontsize=17)
# ax.set_ylabel('Sell Price', fontsize=17)
# ax.set_title('Bubble Chart of Inventory by Age, Price, and Quantity', fontsize=17)

# # Add legend
# ax.legend(title='Inventory Type', fontsize=17)

# # Show the plot
# plt.show()




'''
    APPROACH 2: SEABORN PLOT 
'''

# import pandas as pd
# import numpy as np
# import matplotlib.pyplot as plt
# import seaborn as sns

# # Load the data from the CSV file
# file_path = './CSVs/Data Set #14 report .csv'
# data = pd.read_csv(file_path)

# # Cleaning data by removing commas and currency symbols, and converting to appropriate numeric types
# data['AGE OF INVENTORY DAYS'] = data['AGE OF INVENTORY DAYS'].str.replace(',', '').astype(int)
# data['sell_price'] = data['sell_price'].replace('[\$,]', '', regex=True).astype(float)
# data['qty_on_hand units'] = data['qty_on_hand units'].str.replace(',', '').astype(int)

# # Verify the changes
# print(data[['AGE OF INVENTORY DAYS', 'sell_price', 'qty_on_hand units']].head())
# print(data.dtypes)

# # Set the color palette for the different inventory types
# palette = sns.color_palette("hsv", len(data['inventory_type'].unique()))

# # Create the bubble chart
# plt.figure(figsize=(10, 8))
# bubble_chart = sns.scatterplot(data = data, 
#                                x = 'AGE OF INVENTORY DAYS', 
#                                y = 'sell_price', 
#                                size = 'qty_on_hand units', 
#                                hue = 'inventory_type', 
#                                sizes = (20, 1000), 
#                                palette = palette, 
#                                alpha = 0.6)

# plt.title('Bubble Chart: Age of Inventory Days vs Sell Price')
# plt.xlabel('Age of Inventory Days')
# plt.ylabel('Sell Price ($)')
# plt.grid(True)
# plt.legend(title='Inventory Type', loc='upper right', bbox_to_anchor=(1.25, 1))
# plt.show()


# # Find the proportion of inventory days that span a specific interval
# aged_inventory = data[(data['AGE OF INVENTORY DAYS'] >= 650) & (data['AGE OF INVENTORY DAYS'] <= 750)]
# count_in_range = aged_inventory.shape[0]
# total_count = data.shape[0]
# percentage = (count_in_range / total_count) * 100
# print(f"Count in Range (650-750 days): {count_in_range}")
# print(f"Percentage of Total: {percentage:.2f}%")























# import pandas as pd
# import matplotlib.pyplot as plt



# # Load the dataset
# caja_data = pd.read_csv('./CSVs/caja-dia-a-dia-no-Pii.csv')

# # Display the first few rows and the columns to understand its structure
# print(caja_data.head())
# print(caja_data.columns)
# print(caja_data.info(verbose=True))

# # Convert 'Fecha' to datetime and filter for the year 2022
# df_2022 = caja_data.copy()
# df_2022['Fecha'] = pd.to_datetime(df_2022['Fecha'])
# df_2022 = df_2022[df_2022['Fecha'].dt.year == 2022]

# # Count the occurrences of each 'Tipo Comp DEBE' type for 2022
# type_counts_D_2022 = df_2022['Tipo Comp DEBE'].value_counts()
# print(type_counts_D_2022)

# # Count the occurrences of each 'Tipo Comp h' type for 2022
# type_counts_H_2022 = df_2022['Tipo Comp HABER'].value_counts()
# print(type_counts_H_2022) 

# # Count the occurrences of each 'Nombre cuenta HABER' type for 2022
# type_counts_NH_2022 = df_2022['Nombre cuenta HABER'].value_counts()
# print(type_counts_NH_2022) 

# # Count the occurrences of each 'Nombre cliente DEBE' type for 2022
# type_counts_ND_2022 = df_2022['Nombre de la cuenta DEBE'].value_counts()
# print(type_counts_ND_2022)

# # Plot the results
# plt.figure(figsize=(10, 6))
# type_counts_H_2022.plot(kind='bar', color='limegreen')
# plt.title('Number of Records by Type for 2022')
# plt.xlabel('Type')
# plt.ylabel('Number of Records')
# plt.xticks(rotation=45)
# plt.show()





















'''
    HERE WE PRINT OUT PLENTY OF STATS. HOWEVER, THERE MIGHT BE A METHOD THAT DOES THE SAME THING AS THE BIG PRINT BLOCK AT THE BOTTOM OF THIS SCRIPT. IT'S CALLED THE .describe() METHOD... CHECK IT OUT AND SEE IF IT PRINTS THE SAME THING.
'''

# import pandas as pd

# # Load the dataset
# file_path = './CSVs/synop_evento_SAEZ.csv'
# data = pd.read_csv(file_path)

# # Display the first few rows of the dataset to understand its structure
# # print(data.head(n=15))

# # Calculate the average pressure at sea level
# average_pmar = data['pmar'].mean()

# # Filter data based on conditions
# filtered_data = data[(data['pmar'] > average_pmar) & (data['niebla'] == 'si') & (data['evento'] == 'si')]

# # Display the filtered data
# # print(filtered_data)


# # 1. Date range and the mean atmospheric pressure at sea level (pmar)
# date_range = (filtered_data['valid'].min(), filtered_data['valid'].max())
# mean_pmar = filtered_data['pmar'].mean()

# # 2. Mean year, range of atmospheric pressure at sea level
# mean_year = filtered_data['año'].mean()
# pmar_range = (filtered_data['pmar'].min(), filtered_data['pmar'].max())

# # 3. Visibility (plafond) and temperature (t) ranges and averages
# visibility_range = (filtered_data['vis'].min(), filtered_data['vis'].max())
# mean_visibility = filtered_data['vis'].mean()
# temperature_range = (filtered_data['t'].min(), filtered_data['t'].max())
# mean_temperature = filtered_data['t'].mean()

'''Should we use the .describe() method here?'''
# print(f'\nDates:{date_range}')
# print(f'\nAvg pmar: {mean_pmar}')
# print(f'\nAvg year:{mean_year}')
# print(f'\nRange of pmar:{pmar_range}')
# print(f'\nAvg. vis:{mean_visibility}')
# print(f'\nAvg. of vis:{visibility_range}')
# print(f'\nAvg. temp:{mean_temperature}')
# print(f'\nRange of temps:{temperature_range}')




















# import pandas as pd
# import seaborn as sns
# import matplotlib.pyplot as plt

# df_account_balance = pd.read_csv('./CSVs/Account Balance Base - Hoja1.tsv', sep='\t')

# # Display the first few rows to understand the structure of the data
# print(df_account_balance.head())

# # Clean the ' Balance' column while preserving negative values
# # THERE HAS TO BE A BETTER WAY!
# df_account_balance[' Balance'] = df_account_balance[' Balance'].str.strip()  # Trim spaces
# df_account_balance[' Balance'] = df_account_balance[' Balance'].replace({'\$': '', ',': ''}, regex=True)  # Remove dollar signs and commas
# df_account_balance[' Balance'] = df_account_balance[' Balance'].replace({'- ': '-'}, regex=True) 



# # NOTE: TEST THIS BLOCK OUT!!! YOU NEED AN ELEGANT WAY TO DELA WITH NEGATIVES!
# # Assuming df is your DataFrame and 'Balance' is the column in question
# df['Balance'] = df['Balance'].replace(r'[^\d.-]', '', regex=True)  # Remove everything except digits, decimal points, and minus sign
# df['Balance'] = pd.to_numeric(df['Balance'], errors='coerce')  # Convert to numeric, coercing errors
# df['Balance'].fillna(0, inplace=True)  # Replace NaN values with 0 (if any NaNs were generated)


# # Convert to numeric, ensuring negative values are handled correctly
# df_account_balance[' Balance'] = pd.to_numeric(df_account_balance[' Balance'], errors='coerce')
# # df_account_balance[' Balance'].fillna(0, inplace=True) 

# # Create a bar chart for the distribution of balances
# plt.figure(figsize=(10, 6), facecolor='white')
# sns.histplot(df_account_balance[' Balance'], bins=100, kde=False, color='purple')
# plt.title('Distribution of Account Balances')
# plt.xlabel('Balance')
# plt.ylabel('Frequency')
# plt.grid(True)
# plt.show()
  



















'''
    THIS IS A SIMPLE EXAMPLE ILLUSTRATING THE USE OF COMMON STAT METHODS LIKE CORR AND COV. ALSO, IT USES THE LOC METHOD
'''
# import pandas as pd

# df_hospital = pd.read_csv('./CSVs/Hospital_Survey_Data_Speticemia.csv', skiprows=1)
# # print(df_hospital.head())
# print(df_hospital.info(verbose=True))

# mean_hospital_rating = df_hospital['Hospital Rating'].mean()
# mean_total_payments = df_hospital['Average Total Payments ($)'].mean()

# # Calculate covariance and correlation between hospital rating and average total payments
# covariance = df_hospital[['Hospital Rating', 'Average Total Payments ($)']].cov().iloc[0,1]
# std_dev_rating = df_hospital['Hospital Rating'].std()
# std_dev_payments = df_hospital['Average Total Payments ($)'].std()
# correlation = df_hospital[['Hospital Rating', 'Average Total Payments ($)']].corr().iloc[0,1]

# # Display the results
# print('Mean Hospital Rating:', mean_hospital_rating)
# print('Mean Total Payments:', mean_total_payments)
# print('Covariance between Hospital Rating and Total Payments:', covariance)
# print('Standard Deviation of Hospital Rating:', std_dev_rating)
# print('Standard Deviation of Total Payments:', std_dev_payments)
# print('Correlation Coefficient between Hospital Rating and Total Payments:', correlation)
















'''
    HERE WE REPORT STATS THAT HAVE THE MOST FREQUENCY IN THE DATA SET. WE GET THE ENTRIES WITH THE MOST INSTANCES.
'''
# import pandas as pd

# df_dataset = pd.read_excel('./CSVs/dataset.xls')
# print(df_dataset.head())
# print(df_dataset.info(verbose=True))

# # Filter the DataFrame based on your conditions
# filtered_df = df_dataset[(df_dataset['Recommended Spares'] > 0) & (df_dataset['Storeroom Quantity'] == 0)]

# # Print the number of instances
# print(f'Number of instances: {len(filtered_df)}')
# print(f'\nInstances: {filtered_df}')

# # Get the frequency of each unique value in the 'Lifecycle Status' column
# value_counts = filtered_df['Lifecycle Status'].value_counts()

# # Find the most frequent value and its frequency
# most_frequent_status = value_counts.idxmax()
# most_frequent_status_count = value_counts.max()

# # Print the most frequent 'Lifecycle Status' and its frequency
# print(f"The most frequent value in 'Lifecycle Status' is: {most_frequent_status} with a frequency of {most_frequent_status_count}")

# avg_installed_qnt = filtered_df['Installed Quantity'].mean()
# print(f'Avg installed quantity: {avg_installed_qnt}')















'''
For each city (y-axis) and month (x-axis), draw a bubble where the size of the bubble depends on the average closing price for that city during that month.

    HERE, WE MAKE A COOL SCATTER PLOT USING BUBBLES AND THEIR SIZES TO PLOT A DISTRIBUTION OF PRICES 
'''

# import pandas as pd
# import matplotlib.pyplot as plt
# from matplotlib.cm import viridis
# from matplotlib.colors import to_hex
# import numpy as np

# df_buncombe = pd.read_csv('./CSVs/Buncombe_-_Closed_SFR_-_11_30_23_-_12_30_23_-_Sheet1.csv', header=1)
# print(df_buncombe.head())
# print(df_buncombe.columns)
# print(df_buncombe.info(verbose=True))

# # Clean up and formalize the dates and prices
# df_buncombe['Close Date'] = pd.to_datetime(df_buncombe['Close Date'])
# df_buncombe['Close Price'] = df_buncombe['Close Price'].replace('[\$,]', '', regex=True).astype(float)

# # Extract the months and their names
# df_buncombe['Month'] = df_buncombe['Close Date'].dt.month_name()

# # Group by City and Month
# bubble_data = df_buncombe.groupby(['City', 'Month'])['Close Price'].mean().reset_index()

# # Create the bubble chart
# plt.figure(figsize=(8, 8))
# for i, row in bubble_data.iterrows():
#     plt.scatter(
#         row['Month'], 
#         row['City'], 
#         s=row['Close Price']/1000, 
#         color=to_hex(viridis(np.random.rand())), 
#         alpha=0.5)

# plt.title('Average Closing Price by City and Month', fontsize=17)
# plt.xlabel('Month', fontsize=17)
# plt.ylabel('City', fontsize=17)
# plt.xticks(rotation=45)
# plt.grid(True)
# plt.show()




















# import pandas as pd
# import matplotlib.pyplot as plt
# import seaborn as sns

# # Load the dataset
# data = pd.read_csv('./CSVs/uk_universities.csv')
# print(data.head())
# print(data.info(verbose=True))

# # Remove percentage signs and convert to float
# data['Student_satisfaction'] = data['Student_satisfaction'].str.replace('%', '').astype(float)

# # Calculate the median of the CWUR_score
# median_cwur = data['CWUR_score'].median()

# # Calculate average satisfaction for universities above and below the median CWUR score
# avg_satisfaction_above_median = data[data['CWUR_score'] > median_cwur]['Student_satisfaction'].mean()
# avg_satisfaction_below_median = data[data['CWUR_score'] <= median_cwur]['Student_satisfaction'].mean()

# print('\nmedian of the CWUR_score:')
# print(median_cwur)
# print('\nAverage satisfaction for universities above')
# print(avg_satisfaction_above_median)
# print('\nAverage satisfaction for universities belo')
# print(avg_satisfaction_below_median)

# # Calculate the percentage difference in average satisfaction
# percentage_difference = ((avg_satisfaction_above_median - avg_satisfaction_below_median) / avg_satisfaction_below_median) * 100
# print('\nThe percentage difference between the two:')
# print(percentage_difference)

# # Calculate the correlation coefficient between 'Student_satisfaction' and 'CWUR_score'
# overall_correlation = data[['Student_satisfaction', 'CWUR_score']].corr()
# print(overall_correlation)

# # Group data by region and calculate average 'Student_satisfaction' and 'CWUR_score'
# regional_averages = data.groupby('Region')[['Student_satisfaction', 'CWUR_score']].mean()

# # Calculate correlation for each region
# regional_correlations = data.groupby('Region').apply(lambda x: x[['Student_satisfaction', 'CWUR_score']].corr().iloc[0,1])

# # Combine results into one dataframe
# regional_insights = pd.concat([regional_averages, regional_correlations.rename('Correlation')], axis=1)
# print(regional_insights)

# # Create a single scatter plot with different colors for each region
# plt.figure(figsize=(12, 8))
# sns.scatterplot(data=data, x='CWUR_score', y='Student_satisfaction', hue='Region', palette='viridis', s=100)
# plt.title('Scatter Plot of Student Satisfaction vs CWUR Score Across All Regions')
# plt.xlabel('CWUR Score')
# plt.ylabel('Student Satisfaction (%)')
# plt.legend(title='Region', bbox_to_anchor=(1.05, 1), loc='upper left')
# plt.grid(True)
# plt.show()

# print(regional_averages)



















'''
    HERE WE PERFORM A COHORT ANALYSIS AND SPLIT THE DATA INTO THE BINS INITIALIZED BELOW.
'''

# import pandas as pd

# # Load the data from the CSV file
# file_path = './CSVs/uk_universities.csv'
# uk_universities = pd.read_csv(file_path)

# # Display the first few rows and summary of the data
# print(uk_universities.head()) 
# print(uk_universities.info(verbose=True)) 
# print(uk_universities.describe())

# # Creating cohorts based on founded year
# bins = [0, 1900, 1950, 2000, 2025]  # Year bins
# labels = ['Pre-1900', '1900-1950', '1951-2000', 'Post-2000']
# uk_universities['Cohort'] = pd.cut(uk_universities['Founded_year'], bins=bins, labels=labels, right=False)

# # Calculating average CWUR scores per cohort
# cohort_cwur_scores = uk_universities.groupby('Cohort')['CWUR_score'].mean().reset_index()

# print(cohort_cwur_scores)















'''
    HERE WE HAVE THE STRAWBERRY QUERY
'''
# import pandas as pd
# import seaborn as sns
# import matplotlib.pyplot as plt

# # Load the data from the TSV file
# strawberry_sales_path = './CSVs/STRAWBERRY SALES 2023 - Sheet1.tsv'
# strawberry_sales_data = pd.read_csv(strawberry_sales_path, sep='\t', skiprows=2)

# # Display the first few rows and recheck the data summary
# print(strawberry_sales_data.head()) 
# print(strawberry_sales_data.info(verbose=True)) 
# print(strawberry_sales_data.describe())

# # Convert price fields from string to float and clean them
# strawberry_sales_data['BOX'] = strawberry_sales_data['$/BOX   '].replace('[\$,]', '', regex=True).astype(float)
# strawberry_sales_data['TOTAL'] = strawberry_sales_data['TOTAL       '].replace('[\$,]', '', regex=True).astype(float)
# strawberry_sales_data['DATE'] = pd.to_datetime(strawberry_sales_data['DATE                         '], format='mixed')
# strawberry_sales_data['BOXES'] = pd.to_numeric(strawberry_sales_data['#BOXES '])

# # Setting up the figure
# plt.figure(figsize=(18, 6))

# # Plotting price per box vs. number of boxes
# plt.subplot(1, 2, 1)
# sns.scatterplot(data=strawberry_sales_data, x='#BOXES ', y='BOX')
# plt.title('Price per Box vs. Number of Boxes Sold')
# plt.xlabel('Number of Boxes Sold')
# plt.ylabel('Price per Box ($)')

# # Plotting price per box vs. type of product
# plt.subplot(1, 2, 2)
# sns.boxplot(data=strawberry_sales_data, x='TYPE OF PRODUCT', y='BOX')
# plt.title('Price per Box by Type of Product')
# plt.xlabel('Type of Product')
# plt.ylabel('Price per Box ($)')

# plt.tight_layout()
# plt.show()

# Ensure we use the exact column name for 'Price per Box' from the dataframe
# correct_price_column = [col for col in strawberry_sales_data.columns if "BOX" in col][0]  # Selecting the right column name

# # Group by date to calculate average price and total boxes sold per day
# daily_data = strawberry_sales_data.groupby('DATE').agg(
#     Average_Price_per_Box=('BOX', 'mean'),
#     Total_Boxes_Sold=('BOXES', 'sum')
# ).reset_index()

# # Calculate percent changes for average price and total boxes sold
# daily_data['Price_Percent_Change'] = daily_data['Average_Price_per_Box'].pct_change() * 100
# daily_data['Boxes_Percent_Change'] = daily_data['Total_Boxes_Sold'].pct_change() * 100

# # Displaying the processed data
# print(daily_data.head())

# # Scatter plot for average price and total boxes sold, colored by percent change in price
# plt.figure(figsize=(14, 7))
# plt.scatter(daily_data['DATE'], daily_data['Average_Price_per_Box'],
#             s=daily_data['Total_Boxes_Sold']*10,  # Scale size of points to represent total boxes sold
#             c=daily_data['Price_Percent_Change'], cmap='coolwarm', alpha=0.6)
# plt.colorbar(label='Price Percent Change (%)')
# plt.title('Daily Average Price vs. Total Boxes Sold')
# plt.xlabel('Date')
# plt.ylabel('Average Price per Box ($)')
# plt.grid(True)

# # Show the plot
# plt.show()



# # Scatter plot for percent change in price vs. percent change in boxes sold
# plt.figure(figsize=(10, 6))
# plt.scatter(daily_data['Boxes_Percent_Change'], daily_data['Price_Percent_Change'],
#             color='blue', alpha=0.5)
# plt.title('Percent Change in Price vs. Percent Change in Boxes Sold')
# plt.xlabel('Percent Change in Boxes Sold (%)')
# plt.ylabel('Percent Change in Price (%)')
# plt.grid(True)

# # Show the plot
# plt.show()













'''THIS WAS MODEL A'S RESPONSE'''

# # Extract month and year from `DATE`
# strawberry_sales_data['MONTH'] = strawberry_sales_data['DATE'].dt.month
# strawberry_sales_data['YEAR'] = strawberry_sales_data['DATE'].dt.year

# # Group by `MONTH`, `YEAR` and calculate mean of `$/BOX` and sum of `#BOXES`
# df_agg = strawberry_sales_data.groupby(['MONTH', 'YEAR']).agg(
#     AVG_PRICE=('BOX', 'mean'),
#     TOTAL_BOXES=('BOXES', 'sum')
# ).reset_index()

# # Sort by `YEAR` and `MONTH`
# df_agg = df_agg.sort_values(['YEAR', 'MONTH'])

# # Calculate percent change in price from previous month
# df_agg['PRICE_CHANGE_PCT'] = df_agg['AVG_PRICE'].pct_change() * 100

# # Calculate percent change in boxes sold from previous month
# df_agg['BOXES_CHANGE_PCT'] = df_agg['TOTAL_BOXES'].pct_change() * 100

# # Print the first 5 rows
# print(df_agg.head())

# # Create a scatter plot with `BOXES_CHANGE_PCT` on the x-axis and `PRICE_CHANGE_PCT` on the y-axis
# plt.figure(figsize=(10, 6))
# plt.scatter(df_agg['BOXES_CHANGE_PCT'], df_agg['PRICE_CHANGE_PCT'])

# # Label the axes and add a title
# plt.xlabel('Percent Change in Boxes Sold', fontsize=12)
# plt.ylabel('Percent Change in Price', fontsize=12)
# plt.title('Relationship between Change in Boxes Sold and Change in Price', fontsize=14)

# # Show the plot
# plt.show()























'''
    HERE, WE PERFORM A IQR ANALYSIS TO TRY AND FIND OUTLIERS.
'''
# import pandas as pd

# # Load the data from the CSV file
# engagement_df = pd.read_csv('./CSVs/RAP_Journal_Engagement.csv')

# # Display the first few rows of the dataframe to understand its structure
# print(engagement_df.head())
# print(engagement_df.columns)

# # Calculate the interquartile range (IQR) to identify outliers in the 'Engagement rate'
# Q1 = engagement_df['Engagement rate'].quantile(0.25)
# Q3 = engagement_df['Engagement rate'].quantile(0.75)
# IQR = Q3 - Q1

# # Define outliers as those beyond 1.5 times the IQR from the Q1 and Q3
# lower_bound = Q1 - 1.5 * IQR
# upper_bound = Q3 + 1.5 * IQR

# # Filter the data to find outliers
# outliers = engagement_df[(engagement_df['Engagement rate'] < lower_bound) | (engagement_df['Engagement rate'] > upper_bound)]

# # Display the outliers
# print('Lower bound:', lower_bound, '\nUpper bound:', upper_bound)
# print('Outliers in Engagement Rate:')
# print(outliers[['Region', 'Engagement rate']])


















'''
    UNIQUE CODE HERE, FIGURE OUT WHAT IT DOES SPECIFICALLY!!!
'''
# import pandas as pd

# FILEPATH = './CSVs/outcomes_incomes_fs.xlsx'
# df = pd.read_excel(FILEPATH, header=1)

# print(df.info(verbose=True))

# # Print each column name
# print("Columns: " + ", ".join(df.keys()))

# # Print each column's entries
# for k, v in df.items():
#     # Strip whitespace where possible from column names 
#     # Need to check if isinstance(x, str) because some column names are numbers
#     try:
#         v = v.rename(columns=lambda x: x.strip() if isinstance(x, str) else x)
#     except:
#         pass

#     # Strip whitespace where possible from cells
#     try:
#         v = v.apply(lambda col: col.str.strip() if col.dtype == "object" else col)
#     except:
#         pass

#     df[k] = v
#     print('dataframe: '+ k)
#     print(v.head(15))




















'''
    HERE, WE CREATE A COOL 3D GRAPH AND USE A NEW MODULE!
'''


# import pandas as pd
# import matplotlib.pyplot as plt
# from mpl_toolkits.mplot3d import Axes3D

# # Load the dataset to see the first few rows and understand its structure
# file_path = './CSVs/EQUIP-CHEMICALS.csv'
# chemicals_data = pd.read_csv(file_path)
# print(chemicals_data.head())
# print(chemicals_data.info(verbose=True))

# # Check the unique values and their frequency in the 'Concentration' column
# print('\nTypes of concentration:\n')
# print(chemicals_data['Concentration'].value_counts())
# print(f'There are {chemicals_data["Concentration"].value_counts().sum()} types\n')

# # Function to categorize, group, and handle non-string concentration values
# def categorize_concentration(conc):
#     conc = str(conc)
#     # Physical 
#     if any(x in conc.lower() for x in ['crystal', 'powder', 'solid', 'liquid', 'pellets']):
#         return 1  
#     # Percentage
#     elif conc.endswith('%'):
#         try:
#             return float(conc.strip('%'))  
#         # Miscellaneous 1
#         except ValueError:
#             return 3
#     # Molar  
#     elif 'n' in conc.lower() or 'm' in conc.lower():
#         return 2  
#     # Miscellaneous 2
#     else:
#         return 3  

# # Apply categorization
# chemicals_data['Concentration_Cat'] = chemicals_data['Concentration'].apply(categorize_concentration)
# # Check the unique values and their frequency in the 'Concentration' column
# print('\nTypes of concentration:\n')
# print(chemicals_data['Concentration_Cat'].value_counts())
# print(f'There are {chemicals_data["Concentration_Cat"].value_counts().sum()} types\n')

# # Initialize the 3D scatter plot with the categorization
# fig = plt.figure(figsize=(10, 8))
# ax = fig.add_subplot(111, projection='3d')

# # Define the scatter plot
# sc = ax.scatter(chemicals_data['Qty'], 
#                 chemicals_data['Price'], 
#                 chemicals_data['Concentration_Cat'],
#                 c=chemicals_data['Concentration_Cat'], 
#                 cmap='viridis', 
#                 label='Concentration Categories', 
#                 s=50)

# # Labels and title
# ax.set_xlabel('Quantity')
# ax.set_ylabel('Price')
# ax.set_zlabel('Concentration Category')
# ax.set_title('3D Scatter Plot of Chemical Reagents')

# # Legend with colorbar
# cbar = plt.colorbar(sc, ax=ax)
# cbar.set_label('Concentration Category')

# plt.show()

'''2ND APPROACH: COMPARE AND CONTRAST THE TWO TYPES OF 3D GRAPHS'''

# # Plotting
# fig = plt.figure(figsize=(10, 7))
# ax = fig.add_subplot(111, projection='3d')

# x = df_plot['Qty']
# y = df_plot['Price']
# z = df_plot['Concentration_Num']

# ax.scatter(x, y, z, c='b', marker='o')

# ax.set_xlabel('Quantity')
# ax.set_ylabel('Price')
# ax.set_zlabel('Concentration')

# plt.title('3D Scatter Plot of Chemical Reagents')
# plt.show()














'''
    USE THIS WHEN YOU CANT READ A FILE "i cant read a file" to get its real name! 
    
    NOTE THE EXAMPLE BELOW AND THE FILE NAME. THE FILE NAME HAS A SPACE IN IT AND IT IS ENCODED DIFFERENTLY.
'''
# # Print current working directory
# import os
# print("Current Working Directory:", os.getcwd())

# # List files in the specific directory
# print("Files in './CSVs/':", os.listdir('./CSVs/'))





'''
    HERE WE COUNT THE FEMALE DOMINANCE ACCROSS A NUMBER OF COLUMNS WITH A CLEVER FUNCTION
'''


# import pandas as pd

# file_path = "./CSVs/Chainmaille_by_Yael_–\xa0Customer_Data_(2023).csv"
# df = pd.read_csv(file_path)

# # Display the head of the dataframe to understand its structure
# print(df.head())

# # Display column names and data types
# print('Column names and data types:')
# print(df.dtypes)

# # Display number of records
# print('Number of records:')
# print(len(df))

# # Check for any null values
# print('Null values in each column:')
# print(df.isnull().sum())

# # Check for any obvious outliers or inconsistencies in numeric data
# print('Descriptive statistics for numeric columns:')
# print(df.describe())


# # Calculate and print the number of missing values in each column
# print("\nMissing Values:")
# print(df.isnull().sum())

# # Calculate and print the number of unique values in each column
# print("\nUnique Values:")
# print(df.nunique())

# # Calculate and print descriptive statistics for numeric columns
# print("\nDescriptive Statistics for Numeric Columns:")
# print(df.describe())

# # Calculate and print descriptive statistics for object columns
# print("\nDescriptive Statistics for Object Columns:")
# print(df.describe(include=object))

# # Remove quotation marks from column names
# df.columns = [col.replace('"', '') for col in df.columns]

# # Retry grouping data by various categories and find the most common gender in each group, count occurrences where 'F' is most common
# gender_mode_by_category = df.groupby(['Purchase Site', 'Age of Buyer', 'Product Purchased', 
#                                         'Price', 'How often do you purchase jewelry?', 
#                                         'How often are your jewelry purchases from independent businesses?', 
#                                         'How often do you wear jewelry?', 
#                                         'How much do you usually spend on jewelry per year?'])['Sex of Buyer'].agg(lambda x: x.mode()[0])

# # Count how many times 'F' is the most common gender across all categories
# female_dominance_count = (gender_mode_by_category == 'F').sum()
# print(female_dominance_count)


















'''
    HERE ARE SOME COOL PLOTS LIKE A LINE PLOT AND WE SHOW HOW TO GET THE SKEWNESS
'''
# import pandas as pd
# import matplotlib.pyplot as plt
# import numpy as np

# # Load the TSV data from the uploaded file
# file_path_tsv = './CSVs/Sales_BAME_DATABASE.xlsx - Sales.tsv'
# sales_data = pd.read_csv(file_path_tsv, sep='\t')

# # # Display the first few rows of the dataset to understand its structure
# # print(sales_data.head())
# # print(sales_data.info(verbose=True))
# num_subs = sales_data['Subtotal'].value_counts().sum()
# print(f'Number of subtotal entries: {num_subs}')

# # Convert 'Subtotal' to a float after removing commas
# sales_data['Subtotal'] = sales_data['Subtotal'].str.replace(',', '').astype(float)
# # num_subs2 = sales_data['Subtotal'].value_counts().sum()
# # print(f'Number of subtotal entries (after conversion): {num_subs2}')

# # Provide summary statistics for the 'Subtotal' column
# subtotal_stats = sales_data['Subtotal'].describe()
# # print(subtotal_stats)

# # Plotting the distribution of the 'Subtotal' values
# plt.figure(figsize=(10, 6))
# plt.hist(sales_data['Subtotal'], bins=20, color='skyblue', edgecolor='black')
# plt.title('Distribution of Subtotal Values')
# plt.xlabel('Subtotal ($)')
# plt.ylabel('Frequency')
# plt.grid(True)
# # plt.show()

# # Calculate the skewness of the 'Subtotal' column
# subtotal_skewness = sales_data['Subtotal'].skew()
# print(subtotal_skewness)






'''
    (SAME ANALYSIS) USE A REGEX TO CLEAN UP AND THEN GENERATE A LINE PLOT.
'''

# # Remove commas from the `Subtotal` column
# sales_data['Subtotal'] = sales_data['Subtotal'].astype(str).str.replace(r'[,]', '', regex=True)

# # Convert the `Subtotal` column to numeric
# sales_data['Subtotal'] = pd.to_numeric(sales_data['Subtotal'])
# num_subs = sales_data['Subtotal'].value_counts().sum()
# print(f'Number of subtotal entries (after numeric): {num_subs}')

# # Convert the `Emition` column to datetime
# sales_data['Emition'] = pd.to_datetime(sales_data['Emition'])

# # Sort the dataframe by `Emition` in ascending order
# df = sales_data.sort_values(by='Emition')

# # Describe the `Subtotal` column
# print(df['Subtotal'].describe().round(2))

# # Calculate and print the correlation coefficient between `Subtotal` and `Emition` (converted to ordinal)
# correlation = np.corrcoef(df['Subtotal'], df['Emition'].map(lambda x : x.toordinal()))[0, 1]
# print(f'\nThe correlation coefficient between Subtotal and Emition is: {correlation:.3f}')

# # Plot a histogram of `Subtotal`
# plt.figure(figsize=(10, 6))
# plt.hist(df['Subtotal'], bins=20, edgecolor='k')
# plt.title('Distribution of Subtotal')
# plt.xlabel('Subtotal')
# plt.ylabel('Frequency')

# # Plot a line graph of `Subtotal` over time using `Emition`
# plt.figure(figsize=(10, 6))
# plt.plot(df['Emition'], df['Subtotal'], marker='o', linestyle='-')
# plt.title('Subtotal Over Time')
# plt.xlabel('Emition Date')
# plt.ylabel('Subtotal')
# plt.xticks(rotation=45)

# # Show the plots
# plt.show()

















'''
    HERE, YOU CAN SEE THE DIFFERENCE BETWEEN OVERLAPPING LABELS AND HOW TO FIX BY MODIFYING THE TICKS ROTATION DEGREES

    NOTE: FIND OUT HOW TO CHANGE COLORS WITH THE THIRD INSTANCE BELOW
'''


# import pandas as pd
# import matplotlib.pyplot as plt
# import seaborn as sns

# # Read the CSV file into a DataFrame
# df = pd.read_csv('./CSVs/Student_Mental_health.csv')

# # Display the first 5 rows
# print(df.head())

# # Print the column names and their data types
# print(df.info(verbose=True))

# # Create subplots with 1 row and 2 columns
# fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(15, 6))

# # Filter data for male and female students
# df_male = df[df['Choose your gender'] == 'Male'].dropna()
# df_female = df[df['Choose your gender'] == 'Female'].dropna()

# # Create boxplot for male students
# sns.boxplot(x='What is your course?', y='Age', data=df_male, ax=axes[0])
# axes[0].set_title('Male Students')
# axes[0].tick_params(axis='x', rotation=90)

# # Create boxplot for female students
# sns.boxplot(x='What is your course?', y='Age', data=df_female, ax=axes[1])
# axes[1].set_title('Female Students')
# axes[1].tick_params(axis='x', rotation=90)

# # Display all plots
# plt.show()

''''
    OVERLAPPING BELOW
'''

# import pandas as pd
# import seaborn as sns
# import matplotlib.pyplot as plt

# # Read the CSV file into a DataFrame
# df = pd.read_csv('./CSVs/Student_Mental_health.csv')

# # Display the first 5 rows
# print(df.head())

# # Print the column names and their data types
# print(df.info(verbose=True))

# # Creating separate plots for each gender with different colors for each course
# sns.set_theme(style='whitegrid')

# # Filter data for Male students
# male_df = df[df['Choose your gender'] == 'Male']
# plt.figure(figsize=(14, 8))
# ax1 = sns.boxplot(x='What is your course?', y='Age', data=male_df, palette='Set2')
# plt.title('Age Distribution by Course for Male Students')
# plt.xlabel('Course')
# plt.ylabel('Age')
# plt.xticks(rotation=45)
# plt.show()

# # Filter data for Female students
# female_df = df[df['Choose your gender'] == 'Female']
# plt.figure(figsize=(14, 8))
# ax2 = sns.boxplot(x='What is your course?', y='Age', data=female_df, palette='Set1')
# plt.title('Age Distribution by Course for Female Students')
# plt.xlabel('Course')
# plt.ylabel('Age')
# plt.xticks(rotation=45)
# plt.show()

'''3RD'''

# import pandas as pd
# import seaborn as sns
# import matplotlib.pyplot as plt

# df = pd.read_csv('./CSVs/Student_Mental_health.csv')

# # Creating separate plots for each gender with different colors for each course
# sns.set(style='whitegrid')

# # Filter data for Male students
# male_df = df[df['Choose your gender'] == 'Male']
# plt.figure(figsize=(14, 8))
# ax1 = sns.boxplot(x='What is your course?', y='Age', data=male_df, palette='Set2')
# plt.title('Age Distribution by Course for Male Students')
# plt.xlabel('Course')
# plt.ylabel('Age')
# plt.xticks(rotation=90)
# plt.show()

# # Filter data for Female students
# female_df = df[df['Choose your gender'] == 'Female']
# plt.figure(figsize=(14, 8))
# ax2 = sns.boxplot(x='What is your course?', y='Age', data=female_df, palette='Set1')
# plt.title('Age Distribution by Course for Female Students')
# plt.xlabel('Course')
# plt.ylabel('Age')
# plt.xticks(rotation=90)
# plt.show()















'''
    HERE WE HAVE A GRAPH WITH ANNOTATED POINTS. WE ILLUSTRATE THIS USING THE .annotate METHOD! ALTHOUGH THIS MIGHT NOT BE THE RIGHT GRAPH TO ANNOTATE BECAUSE SINCE THERE ARE A LLOT OF SAMPLE DATA POINTS, A LOT OF THE ANNOTATED LABLES ARE OVERLAPPING AND ITS HARD TO READ. 
    
    NOTE: HOW WOULD WE MAKE IT INTERACTIVE (I.E., ONLY SHOW ANNOTATIONS WHEN WE HOVER WITH MOUSE)? ALSO, FIND OUT WHAT THAT '+' SIGN IS DOING IN THE REPLACE METHOD.
    
    WE ALSO DEMONSTRATE HOW TO CLEAN UP COLUMN NAMES IN THE EVENT WHERE THEY HAVE HIDDEN WHITE SPACES.
'''


# import pandas as pd
# import matplotlib.pyplot as plt

# # Load the data from the CSV file
# file_path = './CSVs/produccion_acero.csv'
# steel_production_data = pd.read_csv(file_path)

# # Display the first few rows of the dataset
# print(steel_production_data.head())
# print(steel_production_data.info(verbose=True))

# # The columns need to be cleaned
# steel_production_data.columns = steel_production_data.columns.str.strip().str.replace(' +', ' ', regex=True)

# # Convert column names to lowercase for easier handling
# steel_production_data.columns = steel_production_data.columns.str.lower()

# # Prepare the plot
# fig, ax = plt.subplots(figsize=(10, 6))

# # Group data by cost center and sum up the relevant columns
# grouped_data = steel_production_data.groupby('c.costo').agg({
#     'total producido': 'sum',   # total produced
#     'merma rec.': 'sum'         # recovered waste
# }).reset_index()

# # Scatter plot to show the relationship
# scatter = ax.scatter(grouped_data['total producido'], grouped_data['merma rec.'], alpha=0.6)

# # Labeling
# ax.set_xlabel('Total Produced')
# ax.set_ylabel('Loss Recovered')
# ax.set_title('Relationship between Total Produced and Loss Recovered by Cost Center')

# # Adding cost center labels to points
# for i, txt in enumerate(grouped_data['c.costo']):
#     ax.annotate(txt, (grouped_data['total producido'][i], grouped_data['merma rec.'][i]))

# plt.show()




















'''
    GET THE TOTAL BY CLEANING UP THE COLUMN AND SUMMING. THEN, CROSS REFRENCING WITH OTHER TOTALS TO SEE IF THEY ALIGN.
'''
# import pandas as pd

# df = pd.read_csv('./CSVs/EMERGENCIA_2023_YAKU.xlsx_M.P.H-MATUCANA-Sheet1.csv')

# # Display the head of the dataframe to understand its structure
# # print(df.head())
# # print(df.columns)
# # print(df.info(verbose=True))

# count = df['TOTAL'].value_counts()
# uniques = df['TOTAL'].value_counts().sum()
# print(f'\n\nTotal before: {count}, and num of uniques: {uniques}')

# # Cleaning the 'TOTAL' column to ensure it's numeric
# df['TOTAL'] = df['TOTAL'].str.replace('.', '')
# df['TOTAL'] = df['TOTAL'].str.replace('S/', '').str.replace(',', '.').str.replace(' ', '').str.replace('-', '0').astype(float)

# count2 = df['TOTAL'].value_counts()
# uniques2 = df['TOTAL'].value_counts().sum()
# print(f'\nTotal after: {count2}, and num of uniques: {uniques2}\n\n')

# # Calculating the total sales amount for each driver and finding the driver with the highest total sales
# total_sales_by_driver = df.groupby('NOMBRE DEL DESPACHADOR')['TOTAL'].sum()

# # Finding the driver with the highest total sales
# max_sales_driver = total_sales_by_driver.idxmax()
# max_sales_amount = total_sales_by_driver.max()

# print('Driver with the most purchases:', max_sales_driver)
# print('Total sales amount:', max_sales_amount)



# # Replace ',' by '.' in `GALONES`
# df['GALONES'] = df['GALONES'].str.extract('([0-9,.]+)').replace(',', '', regex=True).astype(float)

# # Calculating the total sales amount for each driver and finding the driver with the highest total sales
# total_gals_by_driver = df.groupby('NOMBRE DEL DESPACHADOR')['GALONES'].sum()

# # Finding the driver with the highest total sales
# max_gals_driver = total_sales_by_driver.idxmax()
# max_gals_amount = total_sales_by_driver.max()

# print('Driver with the most gallons:', max_sales_driver)
# print('Total gallonss amount:', max_sales_amount)



# # Group the data by 'RAZON SOCIAL' (assuming it represents the driver) and sum the 'TOTAL_CLEAN' column
# grouped_data = df.groupby('RAZON SOCIAL')['TOTAL'].sum().reset_index()

# # Find the entry with the maximum total
# max_purchase = grouped_data.loc[grouped_data['TOTAL'].idxmax()]

# print(max_purchase)
 




















'''
    READ IN A MULTI-SHEET EXCEL FILE AND COMBINE ALL SHEETS INTO ONE. THEN GET THE AVERAGES FOR THE INDIVIDUAL SHEETS AND GET THE AVGS FOR THE COMBINED SHEET.
'''


# import pandas as pd

# # Load the Excel file
# file_path = './CSVs/population_and_age_1.xlsx'
# xlsx = pd.ExcelFile(file_path)

# # Get the names of all sheets in the Excel file
# sheet_names = xlsx.sheet_names

# # Dictionary to store results
# averages = {}
# combined = []

# # Process each sheet
# for idx, sheet_name in enumerate(sheet_names):

#     # Load the sheet into a DataFrame
#     df = pd.read_excel(xlsx, sheet_name=sheet_name)
    
#     # Dynamically identify the 'Age' and 'Population' columns
#     age_col = [col for col in df.columns if 'Age' in col]
#     population_col = [col for col in df.columns if 'Population' in col]
    
#     # Ensure there is exactly one match per category (i.e. one age col)
#     # otherwise, handle errors or ambiguity
#     if len(age_col) == 1 and len(population_col) == 1:
#         average_age = df[age_col[0]].mean()
#         average_population = df[population_col[0]].mean()
        
#         # Store the results
#         averages[sheet_name] = {
#             'Average Age': average_age,
#             'Average Population': average_population
#         }

#         # Subset the DataFrame to include only the relevant columns
#         df = df[[age_col[0], population_col[0]]]
#         df.columns = ['Age', 'Population']  # Normalize column names for concatenation
#         if idx == 0:
#             df['Population'] *= 1000000
#         combined.append(df)
#     else:
#         print(f"Error: Ambiguous or missing columns in '{sheet_name}'. Check the data.")
        
# # Convert results to a DataFrame for better visualization
# results_df = pd.DataFrame.from_dict(averages, orient='index')

# # Combine all data into a single DataFrame
# combined_df = pd.concat(combined, ignore_index=True)

# # Calculate the overall averages
# overall_average_age = combined_df['Age'].mean()
# overall_average_population = combined_df['Population'].mean()

# print('Overall Average Age:', overall_average_age)
# print('Overall Average Population:', overall_average_population)

# # Display the results
# print('Combined data:', results_df)

















'''
    PERCENT CHANGE
'''
# import pandas as pd

# file_path = "./CSVs/LIFE INS ISSUE AGE AUDIT.xlsx"
# life_ins_data = pd.read_excel(file_path)

# print(life_ins_data.info(verbose=True))

# # Get unique values in the "Issue Age" column
# unique_ages = life_ins_data['Issue Age'].unique()

# # Sort the ages to ensure proper ordering
# sorted_ages = sorted(unique_ages)
# print(sorted_ages)

# # Group the data by "Issue Age" and calculate the average "Mode Premium"
# average_premiums_per_age = life_ins_data.groupby('Issue Age')['Mode Premium'].mean().reset_index()

# # Rename columns for clarity
# average_premiums_per_age.columns = ['Issue Age', 'Average Premium']

# # Calculate the percentage change in average premiums between each consecutive age group
# average_premiums_per_age['Percentage Change'] = average_premiums_per_age['Average Premium'].pct_change() * 100

# # Display the table with the percentage changes
# print(average_premiums_per_age)
























'''
    HERE WE HAVE A MULTI-SHEET EXCEL, WITH 3 ROWS OF HEADER AND 1 ROW OF FOOTER, AND WE COMBINE THEM INTO ONE. WE THEN AGGREGATE TWO DISTINCT COLUMNS.
'''


# import pandas as pd

# # Load the Excel file
# file_path = './CSVs/SOLDFOOD2023 - Fall.xlsx'
# xls = pd.ExcelFile(file_path)

# # Sheet names
# sheet_names = xls.sheet_names

# # Load and combine data from all sheets, skip the first three rows and the last footer row
# combined_data = pd.concat(
#     [xls.parse(sheet_name, skiprows=3, skipfooter=1) for sheet_name in sheet_names],
#     ignore_index=True
# )

# # Convert necessary columns to appropriate data types for the 'QUANTITY' and 'TOTAL SALE' columns
# combined_data['QUANTITY'] = pd.to_numeric(combined_data['QUANTITY'])
# combined_data['TOTAL SALE'] = pd.to_numeric(combined_data['TOTAL SALE'])

# # Clean up NaN values that may have occurred due to conversion
# combined_data.dropna(subset=['QUANTITY', 'TOTAL SALE'], inplace=True)

# # Summarize total sales by product group for all months combined
# combined_sales_summary = combined_data.groupby('GROUP').agg({'TOTAL SALE': 'sum', 'QUANTITY': 'sum'})

# # Display the combined sales summary
# print(combined_sales_summary)
















'''
    THINGS WE DID: CONVERT TIME

    1. Convert using pd.to_timedelta: This function is designed to convert strings into timedeltas, which are suitable for arithmetic operations and can be easily analyzed.

    ~ Built-in Pandas Functionality: .to_timedelta is a built-in method in pandas designed to convert scalar, array, list, or series from a recognized timedelta format/strings into a Timedelta type.
    
    ~ Return Type: It returns a Timedelta object, which represents durations, the difference between two dates or times.
    
    ~ Usage: It is extremely useful and straightforward for converting well-formatted time duration strings (e.g., '1 days 00:00:00', '1:00:00') into Timedelta objects. It can handle multiple formats naturally supported by pandas.
'''

# import pandas as pd

# # Load the dataset to review it
# file_path = './CSVs/website_traffic_by_language_2020-2021.csv'
# df = pd.read_csv(file_path)

# # Display the first few rows of the dataset and some general information
# print(df.info(verbose=True))
# print(df.head())

# bounRateUniq = df['Bounce Rate'].unique()
# print(f'Unique values in the bounce rate column:\n{bounRateUniq}')

# avgSesseUniq = df['Avg. Session Duration'].unique()
# print(f'Unique values in the avg. session duration column:\n{avgSesseUniq}')

# totalUsersUniq = df['Total Users '].unique()
# print(f'Unique values in the total users column:\n{totalUsersUniq}')

# # Convert `Bounce Rate` to numeric after removing the '%' character
# df['Bounce Rate'] = df['Bounce Rate'].astype(str).str.replace('%', '', regex = False)
# df['Bounce Rate'] = pd.to_numeric(df['Bounce Rate'])

# # Convert `Avg. Session Duration` to timedelta type and extract total seconds
# df['Avg. Session Duration'] = pd.to_timedelta(df['Avg. Session Duration'])
# df['Avg. Session Duration'] = df['Avg. Session Duration'].dt.total_seconds()

# # Check for negative values in specified columns
# columns_to_check = ['Total Users ', 'Total New Users', 'Sessions', 'Pages / Session', 'Bounce Rate', 'Avg. Session Duration']
# negative_check = (df[columns_to_check] < 0).any().any()

# # Check if `Total New Users` ever exceeds `Total Users `
# new_users_exceed_total_users = (df['Total New Users'] > df['Total Users ']).any()

# # Check if `Sessions` is ever 0
# zero_sessions = (df['Sessions'] == 0).any()

# # Print results
# print(f"Any negative values in {', '.join(columns_to_check)}: {negative_check}\n")
# print(f"`Total New Users` ever exceeds `Total Users `: {new_users_exceed_total_users}\n")
# print(f"`Sessions` is ever 0: {zero_sessions}\n")


'''
    * ANOTHER WAY TO CONVERT THE TIME DURATION COLUMN:

    Custom Functionality: This function was specifically created to address a particular format in your dataset, including handling errors or irregularities (like '0.03:02').
    
    ~ Flexibility: Because it's a custom function, it can be tailored to handle specific, non-standard formats that .to_timedelta might not parse directly without pre-processing.
    
    ~ Return Type: This function was designed to return the total duration in seconds as an integer, making it immediately useful for numerical calculations and comparisons.

    
    * COMPARISON AND SUITABILITY

    ~ Ease of Use: .to_timedelta is generally easier and more robust for standard timedelta formats. It's part of pandas, so it integrates well with DataFrame operations.
    
    ~ Flexibility and Error Handling: The custom convert_to_seconds() function can be adjusted to handle specific cases that aren't directly supported by .to_timedelta, such as fixing formatting errors on the fly.
    
    ~ Performance: Using built-in pandas methods like .to_timedelta is typically more efficient and can handle arrays of data more effectively than iterating with a custom function.
    
    ~ Output Format: If you need output directly in seconds as integers, a custom function might be more straightforward, while .to_timedelta requires an additional step to convert the Timedelta object to seconds.


    * WHEN TO USE:

    ~ If the time data is well-formatted and you benefit from using Timedelta objects within pandas (e.g., for time-based computations), .to_timedelta is preferable.
    
    ~ If you need to handle non-standard formats or want a direct calculation in seconds (or another unit), a custom function might be necessary.
'''

# # Convert 'Bounce Rate' to a numeric value after removing the '%' character
# df['Bounce Rate'] = df['Bounce Rate'].str.replace('%', '').astype(float) / 100

# # Function to convert 'Avg. Session Duration' to seconds
# def convert_to_seconds(time_str):
#     try:
#         if ':' in time_str:
#             # Correct format for time string
#             h, m, s = time_str.split(':')
#             return int(h) * 3600 + int(m) * 60 + int(s)
#         else:
#             # Handling cases like '0.03:02' which are incorrect
#             h, m, s = time_str.replace('.', ':').split(':')
#             return int(h) * 3600 + int(m) * 60 + int(s)
#     except:
#         # If there's an error, return NaN to flag this entry
#         return pd.NA

# # Apply conversion on 'Avg. Session Duration'
# df['Avg. Session Duration'] = df['Avg. Session Duration'].apply(convert_to_seconds)

# # Check for negative values in all numeric columns
# negative_values = df.select_dtypes(include=['int64', 'float']).lt(0).any()

# # Check if 'Total New Users' is greater than 'Total Users'
# new_greater_than_total = (df['Total New Users'] > df['Total Users ']).any()

# # Check for sessions equal to zero
# sessions_zero = (df['Sessions'] == 0).any()

# # Output the results
# print(negative_values) 
# print(new_greater_than_total) 
# print(sessions_zero) 
# print(df.head())



















'''
    HERE IS A COOL WAY OF SPLITTING ENTRIES LIKE 17/90 AND 21-30 TO GET THE FIRST NUMBER AND OMMIT THE REST!
'''


# import pandas as pd
# import numpy as np

# # Load the newly uploaded dataset to review it
# file_path_caja = './CSVs/caja-dia-a-dia-no-Pii.csv'
# data_caja = pd.read_csv(file_path_caja)

# Display the first few rows of the dataset and some general information
# print(data_caja.info(verbose=True)) 
# print(data_caja.head())

# Print some stats
# quotaUniqs = data_caja['Cuota HABER'].unique()
# print(f'Unique values in the credit quota column:\n{quotaUniqs}')
# numQuotaUniqs = data_caja['Cuota HABER'].value_counts().sum()
# print(f'Total Uniq counts in credit quota (before numeric):\n{numQuotaUniqs}')
# dateUniqs = data_caja['Fecha'].value_counts()
# print(f'Unique values in the date column:\n{dateUniqs}')

# # Define the transformation function
# def extract_first_number(value):
#     if pd.isna(value):
#         return np.nan
#     else:
#         # Check and split based on '-' or '/'
#         if '-' in value:
#             # print(f'Made it in the function! Looking at {value} now')     # For testing
#             first_number = value.split('-')[0]
#             # print(f'This is what im returning: {first_number} \n')        # For testing
#         elif '/' in value:
#             # print(f'Made it in the function! Looking at {value} now')     # For testing
#             first_number = value.split('/')[0]
#             # print(f'This is what im returning: {first_number} \n')        # For testing
#         else:
#             first_number = value  # If no delimiter, assume the whole string is a number
        
#         return pd.to_numeric(first_number, errors='coerce')

# # Apply the transformation
# data_caja['Cuota HABER'] = data_caja['Cuota HABER'].astype(str).apply(extract_first_number)

# # Convert the 'Fecha' column to datetime
# data_caja['Fecha'] = pd.to_datetime(data_caja['Fecha']) 

# # Convert the 'Cuota HABER' column to numeric
# data_caja['Cuota HABER'] = data_caja['Cuota HABER'].astype(str).str.replace(r'[-/ ]', '', regex=True)
# data_caja['Cuota HABER'] = pd.to_numeric(data_caja['Cuota HABER'], errors='coerce')

# numQuotaUniqs2 = data_caja['Cuota HABER'].value_counts().sum()
# # print(f'Total Uniq counts in credit quota (after numeric):\n{numQuotaUniqs2}')        # For testing
# quotaUniqs2 = data_caja['Cuota HABER'].unique()
# # print(f'Unique values in the credit quota column (after numeric):\n{quotaUniqs2}')    # For testing

# # Filter for the first semester of 2022
# filtered_df = data_caja[(data_caja['Fecha'] >= '2022-01-01') & (data_caja['Fecha'] <= '2022-06-30')]  

# # Calculate the average of the 'Cuota HABER' column
# average_quota = filtered_df['Cuota HABER'].mean()  

# print("Average Quota for the first semester of 2022:", average_quota)



















'''
    HERE WE PERFORM A IN-DEPTH PATTERN RECOGNITION ANALYSIS TO LOOK FOR TRENDS 
'''
# import pandas as pd

# # Load the data from the CSV file
# file_path = './CSVs/Larkin_Audio_SMB_Data_Set.csv'
# data = pd.read_csv(file_path)

# # Display the first few rows of the dataset and summary information
# print(data.head())
# print(data.info(verbose=True))
# print(data.describe())

# # Print the unique values in the "On a scale of 1-10" column
# scale_uniques = data['On a scale of 1-10'].unique()
# print(f'Scale column uniques:\n{scale_uniques}')

# # Print the unique values in the "On a scale of 1-10" column
# total_cost_uniques = data[' Total Cost '].unique()
# print(f'Total Cost column uniques (BEFORE):\n{total_cost_uniques}')
# total_cost_count = data[' Total Cost '].value_counts().sum()
# print(f'Unique count:\n{total_cost_count}')


# # Convert ` Total Cost ` to numeric after removing '$' and ','
# data[' Total Cost '] = data[' Total Cost '].astype(str).str.replace(r'[$,]', '', regex=True)
# data[' Total Cost '] = pd.to_numeric(data[' Total Cost '])

# # Print the unique values in the "On a scale of 1-10" column
# total_cost_uniques2 = data[' Total Cost '].unique()
# print(f'Total Cost column uniques (AFTER):\n{total_cost_uniques2}')

# # Drop the column `On a scale of 1-10`
# data.drop(columns=['On a scale of 1-10'], inplace=True)

# # Get all survey questions
# survey_questions = data.columns[37:]

# print(f'Survey Questions unique values:\n{survey_questions.value_counts()}')

# # Calculate the average ratings for each survey question
# avg_ratings = data[survey_questions].mean()

# # Calculate the standard deviation for each survey question
# std_ratings = data[survey_questions].std()

# # Print the average ratings and standard deviations for the survey questions
# print("\nAverage Ratings:")
# print(avg_ratings)
# print("\nStandard Deviations:")
# print(std_ratings)

# # Get all `Studio Equipment` columns
# studio_equipment_columns = data.columns[11:37]

# # Print the count of `True` values for each `Studio Equipment` column
# print("\nStudio Equipment Counts:")
# print(data[studio_equipment_columns].sum())



















'''
    PICK SPECIFIC VALUES OR ENTRIES FROM SPECIFIC COLUMNS AND ROWS.
'''
# import pandas as pd

# # Load the TSV data
# file_path_tsv = './CSVs/WELLNESS_COST_2022_CW_V2  - Sheet1.tsv'
# tsv_data = pd.read_csv(file_path_tsv, sep='\t')

# # Display the first few rows to understand the structure and columns
# print(tsv_data.head())
# print(tsv_data.info(verbose=True))

# # Find the employee with a TOTAL value of 410
# employee_with_total_410 = tsv_data[tsv_data['TOTAL'] == 410]

# print(employee_with_total_410)

# # Filter the data for entries with "MATERNITY SUBSIDY" in the CONCEPT column
# maternity_subsidy_data = tsv_data[tsv_data['CONCEPT'] == 'MATERNITY SUBSIDY']

# # Get unique job descriptions and the number of unique employees for each job description
# unique_job_desc = maternity_subsidy_data['JOB DESC'].value_counts()

# # Number of unique employees who received a maternity subsidy
# unique_employees = maternity_subsidy_data['EMPLOYEE'].nunique()

# print(f'Unique job descriptions:\n{unique_job_desc}')
# print(f'Unique employees:\n{unique_employees}')

# # Filter data to only include rows with 'MATERNITY SUBSIDY' in CONCEPT
# maternity_only_data = tsv_data[tsv_data['CONCEPT'] == 'MATERNITY SUBSIDY']

# # Find unique values of JOB DESC and EMPLOYEE within this filtered data
# unique_job_desc_maternity = maternity_only_data['JOB DESC'].unique()
# unique_employees_maternity = maternity_only_data['EMPLOYEE'].unique()

# print(f'Unique job descriptions (MAT):\n{unique_job_desc_maternity}')
# print(f'Unique employees (MAT):\n{unique_employees_maternity}')
















'''
    HERE WE FIND STATS OF ELEMENTS WITHIN A VERY SPECIFIC TIME FRAME. THERE ARE TWO WAYS TO DO IT AND THE 1ST USES BUILT IN FUNCTIONS TO EXTRACT THE HOUR WHILE THE LATTER USES THE DATETIME MODULE!
'''

'''1ST APPROACH'''
# import pandas as pd

# # Load the CSV data
# file_path_paris = './CSVs//Paris_-_Paris.csv'
# paris_data = pd.read_csv(file_path_paris)

# # Display the first few rows to understand the structure and columns
# print(paris_data.head()) 
# print(paris_data.info(verbose=True))

# uniqueTimeCounts = paris_data['datetime'].value_counts()
# print(f'Unique times:\n{uniqueTimeCounts}')
# print(f'Total sum:\n{uniqueTimeCounts.sum()}')

# # Convert 'datetime' to datetime format and extract hour for filtering
# paris_data['datetime'] = pd.to_datetime(paris_data['datetime'])
# paris_data['hour'] = paris_data['datetime'].dt.hour

# uniqueTimeCounts2 = paris_data['hour'].value_counts()
# print(f'Unique times (AFTER):\n{uniqueTimeCounts2}')
# print(f'Total sum:\n{uniqueTimeCounts2.sum()}')

# # Filter data for morning (7 AM to 9 AM) and evening (5 PM to 7 PM) rush hours
# morning_rush = paris_data[(paris_data['hour'] >= 7) & (paris_data['hour'] <= 9)]
# evening_rush = paris_data[(paris_data['hour'] >= 17) & (paris_data['hour'] <= 19)]

# # Calculate average and standard deviation for morning and evening rush hours
# morning_stats = {
#     'Average Travel Time': morning_rush['TravelTimeLive'].mean(),
#     'Standard Deviation': morning_rush['TravelTimeLive'].std()
# }
# evening_stats = {
#     'Average Travel Time': evening_rush['TravelTimeLive'].mean(),
#     'Standard Deviation': evening_rush['TravelTimeLive'].std()
# }

# print(morning_stats)
# print(evening_stats)


'''2ND APPROACH'''
# from datetime import time

# # Convert `datetime` column to datetime
# paris_data['datetime'] = pd.to_datetime(paris_data['datetime'])

# # Calculate the mean `TravelTimeLive` for morning rush hours (7AM-9AM)
# morning_rush = paris_data[(paris_data['datetime'].dt.time >= time(7, 0)) & (paris_data['datetime'].dt.time <= time(9, 0))]
# mean_morning_travel_time = morning_rush['TravelTimeLive'].mean()
# std_morning_tt = morning_rush['TravelTimeLive'].std()

# # Calculate the mean `TravelTimeLive` for evening rush hours (5PM-7PM)
# evening_rush = paris_data[(paris_data['datetime'].dt.time >= time(17, 0)) & (paris_data['datetime'].dt.time <= time(19, 0))]
# mean_evening_travel_time = evening_rush['TravelTimeLive'].mean()
# std_evening_tt = evening_rush['TravelTimeLive'].std()

# # Calculate average and standard deviation for morning and evening rush hours
# morning_stats = {
#     'Average Travel Time': mean_morning_travel_time,
#     'Standard Deviation': std_morning_tt
# }
# evening_stats = {
#     'Average Travel Time': mean_evening_travel_time,
#     'Standard Deviation': std_evening_tt
# }

# print(morning_stats)
# print(evening_stats)



















'''
    HERE WE DO A SCATTER PLOT AND A BOX PLOT TO DEMONSTRATE THE NEED FOR A LOG SCALE WHEN THE RANGE OF VALUES SPAN SEVERAL ORDERS OF MAGNITUDE. THIS IS REGARDING THE BOX PLOT, SO PLAY AROUND WITH OTHER TYPES OF GRAPHS TO SEE IF YOU CAN USE THE LOG SCALE WITH THOSE AS WELL! WE ALSO DEMO THE QCUT METHOD TO SPLIT UP THE DATA ACCORDING TO THE QUARTILES. WE SHOW IT USING THE METHOD AND WE ALSO SHOW HOW TO GROUP A COLUMN INTO CATEGORIES THE MANUAL WAY.
'''
# import pandas as pd
# import matplotlib.pyplot as plt
# import seaborn as sns
# import numpy as np

# # Load the data from the CSV file with a specific encoding
# file_path = './CSVs/Real Estate Mumbai Database - Rgdcvvvh.csv'
# data = pd.read_csv(file_path, encoding='latin-1')

# # Display the first few rows of the dataset and its column names
# print(data.head())
# print(data.columns)
# print(data.info(verbose=True))

# Convert 'AMOUNT IN (INR)' to numeric to ensure proper plotting
# NOTE: THIS COL IS ALREADY NUMERIC SO THIS IS REDUNDANT!
# data['AMOUNT IN (INR)'] = pd.to_numeric(data['AMOUNT IN (INR)'])

# Drop rows with NaN values in 'CLIENT AGE' or 'AMOUNT IN (INR)' to ensure clean data for analysis
# NOTE: THERE ARE 53/53 NON-NULLS SO THIS IS REDUNDANT!
# NOTE: This line, while redundant in this specific case, could protect against future scenarios 
#       where data might have missing values, without requiring additional modifications to your script.
# cleaned_data = data.dropna(subset=['CLIENT AGE', 'AMOUNT IN (INR)'])

# # Make a subset without using dropna()
# cleaned_data = data[['CLIENT AGE', 'AMOUNT IN (INR)']]

# Plotting a scatter viz.
# plt.figure(figsize=(10, 6))
# sns.scatterplot(data = cleaned_data, 
#                 x = 'CLIENT AGE', 
#                 y = 'AMOUNT IN (INR)', 
#                 hue = 'TRANSACTION TYPE', 
#                 style = 'TRANSACTION TYPE', 
#                 palette = 'deep')
# plt.title('Trends between Client Age and Transaction Amount')
# plt.xlabel('Client Age')
# plt.ylabel('Transaction Amount (INR)')
# plt.yscale('log')  # Using log scale due to wide range in transaction amounts
# plt.grid(True)
# plt.show()


'''1ST APPROACH: NO LOG SCALE AND QCUT'''
# # Create age groups based on quantiles
# data['age_group'] = pd.qcut(cleaned_data['CLIENT AGE'], q=3, labels=['Young', 'Middle-Aged', 'Older'])

# # Calculate mean and median transaction amount by age group
# grouped_data = data.groupby('age_group')['AMOUNT IN (INR)'].agg(['mean', 'median']).round(2)

# # Sort by mean transaction amount
# grouped_data = grouped_data.sort_values(by='mean')

# # Print the results
# print(grouped_data)

# # Create boxplot
# plt.figure(figsize=(10, 6))
# data.boxplot(column='AMOUNT IN (INR)', by='age_group')

# # Add labels and title
# plt.xlabel('Age Group')
# plt.ylabel('Transaction Amount (INR)')
# plt.title('Transaction Amount Distribution by Age Group')
# #plt.yscale('log')  # NOTE: SHOULD BE USING log scale due to wide range in transaction amounts! (like below)
# # Show the plot
# plt.xticks(rotation=45)
# plt.show()



'''2ND APPROACH: LOG SCALE AND QCUT'''
# # Using qcut to categorize ages into quantiles
# cleaned_data['Age Group Qcut'] = pd.qcut(cleaned_data['CLIENT AGE'], q=3, labels=['Young', 'Middle-Aged', 'Older'])

# # Calculate the average and median transaction amounts for each age group defined by qcut
# group_stats_qcut = cleaned_data.groupby('Age Group Qcut')['AMOUNT IN (INR)'].agg(['mean', 'median']).round(2).reset_index()

# # Sort by mean transaction amount
# group_stats_qcut = group_stats_qcut.sort_values(by='mean')

# # Plotting boxplot for transaction amounts by age group using qcut
# plt.figure(figsize=(10, 6))
# sns.boxplot(data=cleaned_data, x='Age Group Qcut', y='AMOUNT IN (INR)', palette='coolwarm', hue='Age Group Qcut', legend=False)
# plt.title('Distribution of Transaction Amounts by Age Group (Qcut)')
# plt.xlabel('Age Group (Qcut)')
# plt.ylabel('Transaction Amount (INR)')
# plt.yscale('log')  # Using log scale due to wide range in transaction amounts
# plt.grid(True)
# plt.show() 
# print(group_stats_qcut)



'''3RD APPROACH: LOG SCALE AND MANUAL QUANTILE'''
# # Calculate the 33rd and 66th percentiles to define the age groups
# quantiles = cleaned_data['CLIENT AGE'].quantile([0.33, 0.66])

# # Define the age categories based on quantiles
# bins = [0, quantiles[0.33], quantiles[0.66], float('inf')]
# labels = ['Young', 'Middle-Aged', 'Older']
# cleaned_data['Age Group Quantiles'] = pd.cut(cleaned_data['CLIENT AGE'], bins=bins, labels=labels, right=False)

# # Calculate the average and median transaction amounts for each age group
# group_stats_quantiles = cleaned_data.groupby('Age Group Quantiles')['AMOUNT IN (INR)'].agg(['mean', 'median']).reset_index()

# # Plotting boxplot for transaction amounts by age group using quantiles
# plt.figure(figsize=(10, 6))
# sns.boxplot(data=cleaned_data, x='Age Group Quantiles', y='AMOUNT IN (INR)', palette='light:#5A9')
# plt.title('Distribution of Transaction Amounts by Age Group (Quantiles)')
# plt.xlabel('Age Group (Quantiles)')
# plt.ylabel('Transaction Amount (INR)')
# plt.yscale('log')  # Using log scale due to wide range in transaction amounts
# plt.grid(True)
# plt.show(), group_stats_quantiles



















'''
    HERE WE DEMO THE DIFFERENCE BETWEEN USING DROPNA AND NOT. WE ALSO DEMO THE MARKDOWN METHOD WITH PRINT.
'''
# import pandas as pd

# # Load the data from the 'last60.csv' file
# last60_data = pd.read_csv('./CSVs/last60.csv')

# # Display the first few rows of the dataset and its column names to understand its structure
# print(last60_data.head())
# print(last60_data.columns)
# print(last60_data.info(verbose=True))

# uniqueBrands = last60_data['Brand'].unique()
# print(f'Brand names:\n{uniqueBrands}')
# valCounts = last60_data['Brand'].value_counts()
# print(f'And we have this many of each:\n{valCounts}')
# print(f'Overall, there are {valCounts.sum()} of them')

# # Group by 'Brand' and calculate the required metrics
# brand_metrics = last60_data.groupby('Brand').agg(
#     Total_Cost = ('Cost', 'sum'),
#     Average_Length = ('Length', 'mean'),
#     Average_Height = ('Height', 'mean')
# )

# # Sort the results by `Total Cost` in descending order
# brand_metrics = brand_metrics.sort_values(by='Total_Cost', ascending=False)

# # Format the columns
# brand_metrics['Total Cost'] = brand_metrics['Total Cost'].apply(lambda x: f'${x:.2f}')
# brand_metrics['Average Length'] = brand_metrics['Average Length'].apply(lambda x: f'{x:.3f}')
# brand_metrics['Average Height'] = brand_metrics['Average Height'].apply(lambda x: f'{x:.3f}')

# # Print the resulting table
# print(brand_metrics.to_markdown(index=False, numalign="left", stralign="left"))



'''2ND APPROACH: DROPNA'''
# # Drop null values in `Length` and `Height` columns
# df_filtered = last60_data.dropna(subset=['Length', 'Height'])

# # Group by `Brand` and calculate the sum of `Cost`, mean of `Length`, and mean of `Height`
# df_agg = df_filtered.groupby('Brand').agg(
#     Total_Cost=('Cost', 'sum'),
#     Average_Length=('Length', 'mean'),
#     Average_Height=('Height', 'mean')
# ).reset_index()

# # Rename the columns
# df_agg = df_agg.rename(columns={'Total_Cost': 'Total Cost', 'Average_Length': 'Average Length', 'Average_Height': 'Average Height'})

# # Sort the results by `Total Cost` in descending order
# df_agg = df_agg.sort_values(by='Total Cost', ascending=False)

# # Format the columns
# df_agg['Total Cost'] = df_agg['Total Cost'].apply(lambda x: f'${x:.2f}')
# df_agg['Average Length'] = df_agg['Average Length'].apply(lambda x: f'{x:.3f}')
# df_agg['Average Height'] = df_agg['Average Height'].apply(lambda x: f'{x:.3f}')

# # Print the resulting table
# print(df_agg.to_markdown(index=False, numalign="left", stralign="left"))

















'''
    HERE WE HAVE A COMPLEX SCATTER PLOT WITH MANY CONSTRAINTS. WE PRESENT 3 APPROACHES THAT INVOKE DIFFERENT STRATEGY, LOGIC, AND SYNTAX.
'''

'''APPROACH 1: SPLIT INTO TWO TYPES OF DATA POINTS SO THAT THE LEGEND IS AUTOMATICALLY GENERATED'''
# import pandas as pd
# import matplotlib.pyplot as plt

# # Load the dataset to inspect its contents
# file_path = './CSVs/used_cars.csv'
# data = pd.read_csv(file_path)

# # Explore the data set
# print(data.head())
# print(data.info(verbose=True))

'''
Generate a scatter plot from "used_cars.csv" data with mileage (miles: int64) on the x-axis and sedan prices (price: int64) on the y-axis. 

Plot sedans with over 150 horsepower (horsepower: int64) and 130 torque (torque: int64) in red, the rest in blue. 

Include only sedans (type: object) with a 10 or less difference in city and highway mileage (city_mileage: int64, highway_mileage: int64), and show the legend. 

Please provide the Python code.
'''

# # Consider only sedans with a 10 or less difference in city and highway mileage
# df_sedans = data[data['type'] == 'sedan']
# df_sedans = df_sedans[(df_sedans['highway_mileage'] - df_sedans['city_mileage']).abs() <= 10]

# plt.figure(figsize=(10, 6))

# # Sedans with over 150 horsepower and 130 torque in red, the rest in blue
# # Do them seperate so that plt knows what each color represents regarding the legend
# df_high_perf = df_sedans[(df_sedans['horsepower'] > 150) & (df_sedans['torque'] > 130)]
# df_low_perf = df_sedans[~((df_sedans['horsepower'] > 150) & (df_sedans['torque'] > 130))]
# plt.scatter(df_low_perf['miles'], df_low_perf['price'], color='blue', label='Low Performance Sedans')
# plt.scatter(df_high_perf['miles'], df_high_perf['price'], color='red', label='High Performance Sedans')

# # Mileage on the x-axis and sedan prices on the y-axis
# plt.xlabel('Mileage')
# plt.ylabel('Price')
# plt.title('Sedan Prices vs. Mileage Based on Performance')

# # Show the legend
# plt.legend()
# plt.show()


'''APPROACH 2: THE LEGEND IS MANUALLY GENERATED BY EXPLICITLY INITIALIZING PATCHES'''
# # Filter for sedans with mileage difference <= 10
# df_filtered = data[(data['type'] == 'sedan') & (abs(data['city_mileage'] - data['highway_mileage']) <= 10)].copy()

# # Create the color column
# df_filtered['color'] = ['red' if (hp > 150) & (tq > 130) else 'blue' for hp, tq in zip(df_filtered['horsepower'], df_filtered['torque'])]

# # Create the scatter plot
# plt.figure(figsize=(10, 6))
# scatter = plt.scatter(df_filtered['miles'], df_filtered['price'], c=df_filtered['color'], alpha=0.7)

# # Add labels and title
# plt.xlabel('Mileage')
# plt.ylabel('Price ($)')
# plt.title('Price vs. Mileage for Sedans with Mileage Difference <= 10')

# # Create custom legends
# legend_labels = ['High Performance Sedans', 'Low Performance Sedans']
# legend_colors = ['red', 'blue']
# patches = [plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=color, markersize=10) for color in legend_colors]
# plt.legend(patches, legend_labels)

# # Show the plot
# plt.show()


'''APPROACH 3: USE THE PATCHES METHOD'''
# # Create the scatter plot
# plt.figure(figsize=(10, 6))
# scatter = plt.scatter(df_filtered['miles'], df_filtered['price'], c=df_filtered['color'], alpha=0.7)

# # Create custom legend
# red_patch = mpatches.Patch(color='red', label='High Performance (>150 HP & >130 Torque)')
# blue_patch = mpatches.Patch(color='blue', label='Standard Performance')
# plt.legend(handles=[red_patch, blue_patch], loc="lower right", title="Car Type")

# # Add labels and title
# plt.xlabel('Mileage')
# plt.ylabel('Price ($)')
# plt.title('Price vs. Mileage for Sedans with Mileage Difference <= 10')

# # Show the plot
# plt.show()


'''APPROACH 1: STREAMLINED'''
# import pandas as pd
# import matplotlib.pyplot as plt

# # Filter for sedans with mileage difference <= 10
# df_filtered = data[(data['type'] == 'sedan') & (abs(data['city_mileage'] - data['highway_mileage']) <= 10)].copy()

# # Create a boolean column for high power sedans
# df_filtered['high_power'] = (df_filtered['horsepower'] > 150) & (df_filtered['torque'] > 130)

# # Create the scatter plot
# plt.figure(figsize=(10, 6))
# plt.scatter(df_filtered[df_filtered['high_power']]['miles'], 
#             df_filtered[df_filtered['high_power']]['price'], 
#             color='red', 
#             label='High Power')

# plt.scatter(df_filtered[~df_filtered['high_power']]['miles'], 
#             df_filtered[~df_filtered['high_power']]['price'], 
#             color='blue', 
#             label='Other')

# # Add labels and title
# plt.xlabel('Miles')
# plt.ylabel('Price')
# plt.title('Price vs. Miles for Sedans (Mileage Difference <= 10)')

# # Add legend
# plt.legend()

# # Show the plot
# plt.show()























'''
    CREATE A NEW DATAFRAME AND SAVE IT TO A OUTPUT CSV
'''

# import pandas as pd

# # Load the dataset
# file_path = './CSVs/Real Estate Mumbai Database - Rgdcvvvh.csv'
# data = pd.read_csv(file_path, encoding='ISO-8859-1')

# # Check the dataframe structure again
# print(data.head()) 
# print(data.columns)
# print(data.info(verbose=True))

# # Calculate overall averages for the dataset
# average_values = pd.DataFrame({
#     'Overall Average Bedrooms': [data['NUMBER OF BEDROOMS'].mean()],
#     'Overall Average Amount INR': [data['AMOUNT IN (INR)'].mean()]
# })

# avg_rooms = data['NUMBER OF BEDROOMS'].mean()
# avg_amount = data['AMOUNT IN (INR)'].mean()

# # Save to CSV (idx=F to prevent pandas from writing the  
# # DataFrame's index as a separate column in the CSV file)
# output_file_path = './OutCSVs/bedrooms_to_amount.csv'
# average_values.to_csv(output_file_path, index=False)

# print(f'Average number of bedrooms:\n{avg_rooms}')
# print(f'Average values:\n{avg_amount}')
# print(f'The file is saved to this:\n{output_file_path}')



















'''
    A BAR GRAPH WHERE THE BARS ARE DIFFERENT COLORS
'''
# import pandas as pd
# import matplotlib.pyplot as plt

# # Load the newly uploaded CSV file to examine its contents and structure
# file_path_caja = './CSVs/caja-dia-a-dia-no-Pii.csv'
# caja_data = pd.read_csv(file_path_caja)

# # Explore the structure
# print(caja_data.head())
# print(caja_data.columns)
# print(caja_data.info(verbose=True))

# # Convert the 'Fecha' column to datetime format for easier date manipulation
# caja_data['Fecha'] = pd.to_datetime(caja_data['Fecha'])

# # Filter the data for the year 2022
# caja_2022 = caja_data[caja_data['Fecha'].dt.year == 2022]

# # Count records by 'Tipo Comp HABER' and 'Tipo Comp DEBE'
# haber_counts = caja_2022['Monto HABER'].value_counts().sum()
# debe_counts = caja_2022['Monto DEBE'].value_counts().sum()

# print(f'Credit:\n{haber_counts}')
# print(f'Debit:\n{debe_counts}')

# # Create two dataframes: one for HABER and one for DEBE transactions
# df_haber = caja_2022[caja_2022['Monto HABER'].notnull()].copy()
# df_debe = caja_2022[caja_2022['Monto DEBE'].notnull()].copy()

# # Aggregate the dataframes by counting the number of rows
# transactions_haber = df_haber.shape[0]
# transactions_debe = df_debe.shape[0]

# # Create a new dataframe for plotting
# df_plot = pd.DataFrame({'Tipo': ['DEBE', 'HABER'], 
#                         'Transactions': [transactions_debe, transactions_haber]})

# # Create and display a bar plot
# plt.figure(figsize=(10, 6))
# plt.bar(df_plot['Tipo'], df_plot['Transactions'], color=['skyblue', 'lightcoral'])
# plt.xlabel('Transaction Type', fontsize=12)
# plt.ylabel('Number of Transactions', fontsize=12)
# plt.title('Number of Transactions by Type in 2022', fontsize=14)
# plt.show()















'''
    METHOD TO TRY MULTIPLE ENCODINGS
'''
# encodings = ['cp1252', 'utf-16', 'ascii', 'latin-1']
# for enc in encodings:
#     try:
#         split_data = pd.read_csv(xlsx_file_path, sep='|', skiprows=[1], encoding=enc)
#         print(f"Success with encoding: {enc}")
#         break
#     except UnicodeDecodeError:
#         print(f"Failed with encoding: {enc}")
























'''
    CALCULATING PROFIT MARGIN FOR EACH REGION AND FINDING OUTLIERS
'''


# import pandas as pd
# import matplotlib.pyplot as plt

# # Load the data from the uploaded XLSX file
# xlsx_file_path = './CSVs/DynamicBiz_insight.xlsx'

# # Load the Excel file without attempting to split columns initially
# insight_data_raw = pd.read_excel(xlsx_file_path, header=None)

# # Split the single column into multiple columns by '|'
# split_data = insight_data_raw[0].str.split('|', expand=True)

# # Use the first row as header
# split_data.columns = split_data.iloc[0].apply(lambda x: x.strip())
# split_data = split_data[2:]  # Remove the header row from the data

# # Clean up the data by trimming whitespace
# split_data = split_data.map(lambda x: x.strip() if isinstance(x, str) else x)

# print(split_data.head(n=20))
# print(split_data.info(verbose=True))
# print(split_data.columns)

# # Remove unwanted characters and convert data types
# split_data['Profit Margin'] = split_data['Profit Margin'].str.replace('%', '').astype(float)

# # Calculate the mean profit margin for each region
# region_profit_margin_avg = split_data.groupby('Region')['Profit Margin'].mean().rename('Profit Margin_avg')

# # Merge the average profit margin back into the original dataframe
# data = split_data.merge(region_profit_margin_avg, on='Region')

# # Calculate the deviation of each product's profit margin from the region's average
# data['Profit Margin Deviation'] = data['Profit Margin'] - data['Profit Margin_avg']

# # For each region, find the product with the highest absolute deviation
# def get_max_deviation_product(group):
#     return group.loc[group['Profit Margin Deviation'].abs().idxmax()]

# max_deviation_products = data.groupby('Region').apply(get_max_deviation_product)

# # Display the products with the highest absolute deviation for each region
# print(max_deviation_products[['Product', 'Region', 'Profit Margin', 'Profit Margin_avg', 'Profit Margin Deviation']])
















'''
    NEED TO REVISIT AND FINISH THIS: NEED A STRATEGY TO PLOT MOONPHASE AGAINST TIME!
'''

# import pandas as pd
# import matplotlib.pyplot as plt
# import seaborn as sns

# # Load the data
# file_path = '/mnt/data/Indian Summers.csv'
# data = pd.read_csv(file_path)
# data['Date'] = pd.to_datetime(data['Date'])  # Convert 'Date' to datetime type
# data['Year'] = data['Date'].dt.year
# data['Month'] = data['Date'].dt.month

# # Filter data to include only April to June from 2007 to 2011
# filtered_data = data[(data['Year'].between(2007, 2011)) & (data['Month'].between(4, 6))]

# # Group by Year and Month, computing the average moon phase
# monthly_avg_data = filtered_data.groupby(['Year', 'Month']).agg({'moonphase': 'mean'}).reset_index()

# # Pivot to create a suitable structure for the heatmap
# heatmap_data_structured = monthly_avg_data.pivot("Month", "Year", "moonphase")

# # Create the heatmap
# plt.figure(figsize=(15, 8))
# ax = sns.heatmap(heatmap_data_structured, cmap="viridis", annot=True, fmt=".2f", cbar_kws={'label': 'Average Moon Phase (Decimal)'})
# ax.set_title('Monthly Moon Phase Averages Over Years (2007-2011)')
# ax.set_xlabel('Year')
# ax.set_ylabel('Month')
# plt.show()


























'''
    HERE WE HAVE AN INTERESTING FOR-LOOP. FIGURE OUT WHAT ITS DOING!
'''
# import pandas as pd
# from pandas.api.types import is_numeric_dtype

# # Read the CSV file into a DataFrame
# df = pd.read_excel('./CSVs/PAYROLL_MAY.xlsx')

# # Display the first 5 rows
# print(df.head())

# # Print the column names and their data types
# print(df.info(verbose=True))

# for column_name in ['YEAR']:
#   if not is_numeric_dtype(df[column_name]):
#     # Assume CSV columns can only be numeric or string.
#     df[column_name] = pd.to_numeric(
#         df[column_name].str.replace(',', 
#                                     repl='', 
#                                     regex=True),
#                                     ).fillna(0)

# print(df['YEAR'].value_counts())
# print(df['YEAR'].describe(percentiles=[.1, .25, .5, .75, .9]))

# # Combine `YEAR` and `MONTH` to create datetime column `Date`
# df['Date'] = pd.to_datetime(df[['YEAR', 'MONTH']].assign(DAY=1))

# # Print the value counts of `Date`
# print(df['Date'].value_counts())

# # Print descriptive statistics of `Date`
# print(df['Date'].describe())

# # Combine the month and year into a single datetime column
# df['Date'] = pd.to_datetime(df.YEAR.astype(str) + '-' + df.MONTH.astype(str), format='%Y-%m')

# # Show the first few rows to confirm the new column
# print(df[['MONTH', 'YEAR', 'Date']].head())

















'''
    HERE WE RENAME COLUMNS TO ADDRESS TYPOS AND WE EMPHASIZE THE DATE FORMAT WHEN CONVERTING COLUMNS TO DATETIME FORMAT.
'''

# import pandas as pd
# import seaborn as sns
# import matplotlib.pyplot as plt

# # Load the data from the provided CSV file
# patient_file_path = './CSVs/patient_list-_patient_list.csv'
# patient_data = pd.read_csv(patient_file_path)

# # Display the first few rows of the dataframe and its columns to understand its structure
# print(patient_data.head()) 
# print(patient_data.columns)
# print(patient_data.info(verbose=True))

# # Convert the '% reimbursement' column to a format that can be compared numerically
# patient_data['% reimbursement'] = patient_data['% reimbursement'].str.rstrip('%').astype('float') / 100

# # Filter the data for patients with 15% reimbursement
# patients_15_reimbursement = patient_data[patient_data['% reimbursement'] == 0.15]

# # Summary of these patients
# patients_15_reimbursement_summary = patients_15_reimbursement.describe(include='all')
# print('15 Patients:\n')
# print(patients_15_reimbursement_summary)

# # Check the uniqueness and counts of the 'release_date' column to understand its composition
# release_date_counts = patient_data['release_date'].value_counts().sum()
# print(f'Release date uniques and their counts:\n{release_date_counts}')

# # Correct the spelling mistake in column names for easier reference
# patient_data.rename(columns={'beginning_of_traetment': 'beginning_of_treatment'}, inplace=True)

# # Check for side effects
# begin_treatment_counts = patient_data['beginning_of_treatment'].value_counts().sum()
# print(f'Begin treatment (BEFORE):\n{begin_treatment_counts}')

# # Convert date columns to datetime
# patient_data['beginning_of_treatment'] = pd.to_datetime(patient_data['beginning_of_treatment'], dayfirst=True)
# patient_data['release_date'] = pd.to_datetime(patient_data['release_date'], format='%d/%m/%Y', errors='coerce')

# # Check for side effects
# release_date_counts = patient_data['release_date'].value_counts().sum()
# print(f'Release date uniques and their counts (AFTER):\n{release_date_counts}')
# begin_treatment_counts2 = patient_data['beginning_of_treatment'].value_counts().sum()
# print(f'Begin treatment (AFTER):\n{begin_treatment_counts2}')




# # Calculate treatment length in months where possible
# patient_data['treatment_length_months'] = (
#     (patient_data['release_date'] - patient_data['beginning_of_treatment']).dt.days / 30.44
#     ).round()




# # Calculate treatment length in days where possible
# patient_data['treatment_length_days'] = (patient_data['release_date'] - patient_data['beginning_of_treatment']).dt.days



# # Display the data with new treatment length column
# # print(patient_data[['name', 'age', '% reimbursement', 'beginning_of_treatment', 'release_date', 'treatment_length_months']].head())
# print(patient_data[['name', 'age', '% reimbursement', 'beginning_of_treatment', 'release_date', 'treatment_length_days']].head())

# # Scatter plot for Age vs. % Reimbursement
# plt.figure(figsize=(10, 6))
# sns.scatterplot(x=patient_data['age'], y=patient_data['% reimbursement'])
# plt.title('Age vs. % Reimbursement')
# plt.xlabel('Age')
# plt.ylabel('% Reimbursement')
# plt.grid(True)
# plt.show()

# # Calculate the Pearson correlation coefficient for age and % reimbursement
# age_reimbursement_corr = patient_data['age'].corr(patient_data['% reimbursement'])
# print(age_reimbursement_corr)

# # Filter out rows where treatment length is NaN
# filtered_data_with_treatment_length = patient_data.dropna(subset=['treatment_length_days'])

# # Scatter plot for Treatment Length vs. % Reimbursement
# plt.figure(figsize=(10, 6))
# sns.scatterplot(x=filtered_data_with_treatment_length['treatment_length_days'], y=filtered_data_with_treatment_length['% reimbursement'])
# plt.title('Treatment Length (Days) vs. % Reimbursement')
# plt.xlabel('Treatment Length (Days)')
# plt.ylabel('% Reimbursement')
# plt.grid(True)
# plt.show()

# # Calculate the Pearson correlation coefficient for treatment length and % reimbursement
# treatment_reimbursement_corr = filtered_data_with_treatment_length['treatment_length_months'].corr(filtered_data_with_treatment_length['% reimbursement'])
# print(treatment_reimbursement_corr)













'''
    USE THIS WHEN IT WONT or CANT FIND A FILE IN CSVs DIRECTORY... IT MIGHT BE HIDDEN OR HAVE A DIFF NAME BEHIND THE SCENES!
'''

# # Print current working directory
# import os
# print("Current Working Directory:", os.getcwd())

# # List files in the specific directory
# print("Files in './CSVs/':", os.listdir('./CSVs/'))





'''
    HERE WE DEMO THE IDXMAX METHOD, WHICH IS USED AFTER WE COUNT INSTANCES OF UNIQUE ENTRIES, TO GET THE MOST FREQUENT OR THE MOST COMMON STRING ENTRY. THEN WE DEMO THE ISIN METHOD, WHICH HELPS US FIND SPECIFIC STRING ENTRIES IN THEIR RESPECTIVE ROW. WE ALSO DEMO THE COUNTER MODULE AT THE END.
'''


# import pandas as pd

# df = pd.read_csv('./CSVs/WELLNESS_COST_2022_CW_V2  - Sheet1.tsv', sep='\t')
# print(df.head())
# print(df.columns)
# print(df.info(verbose=True))

# cost_center_counts = df['COST_CENTER'].value_counts()
# print(cost_center_counts.head())

# most_common_cost_center = cost_center_counts.idxmax()
# print('Most common COST_CENTER:', most_common_cost_center)

# # Filter the dataframe for rows where CONCEPT 
# # is either 'MATERNITY SUBSIDY' or 'MEDICAL REST'
# filtered_df = df[df['CONCEPT'].isin(['MATERNITY SUBSIDY', 'MEDICAL REST'])]
# print("Queried column:\n")
# print(filtered_df)

# # Filter for rows with 'MATERNITY SUBSIDY' or 'MEDICAL REST' in the CONCEPT column
# filtered_df = df[df['CONCEPT'].isin(['MATERNITY SUBSIDY', 'MEDICAL REST'])]
# print("Queried column:\n")
# # Print the filtered DataFrame
# print(filtered_df)

# # Count the number of records and check the unique years in the filtered data
# record_count = filtered_df.shape[0] # Shape returns the dimensionality, i.e., 0 = rows x cols = 1
# unique_years = filtered_df['PERIOD'].unique()
# print('Number of records:', record_count)
# print('Unique years in the data:', unique_years)


'''
    COUNT MANUALLY WITH THE COUNTER MODULE
'''
# from collections import Counter

# cost_center_counts = Counter(df['COST_CENTER'])

# # Print the results, sorted in descending order by frequency
# for cost_center, count in cost_center_counts.most_common():
#   print(f'{cost_center}: {count}')


















'''
    HERE, WE PERFORM A BUNCH OF OPS. SO ILL JUST LIST KEYWORDS:

    - COUNTER
    - MOST COMMON
    - TO PERIOD
    - LINE GRAPH
    - PLOT AND SUBPLOT
    - BAR GRAPH
    - HARDCODE/CUSTOM X-TICKS/X-AXIS LABELS
    - GRID LINES
'''

# import pandas as pd
# from collections import Counter
# import matplotlib.pyplot as plt

# # Load the Excel file
# wild_df = pd.read_excel('./CSVs/Wild_by_Aura_Final.xlsx')

# # Explore the data set
# print(wild_df.head())
# print(wild_df.columns)
# print(f"INFO STARTS HERE!\n")
# print(wild_df.info(verbose=True))

# # Get the most common entry 
# age_platform_counts = Counter(
#     zip(wild_df['Shopping Platform'], wild_df['Age'])
# )

# print("Most common:\n")
# print(age_platform_counts.most_common())

# # Extract month, quarter, and year 
# # NOTE: No need to convert since its already datetime dtype
# wild_df['Month'] = wild_df['Purchase Date'].dt.month
# wild_df['Quarter'] = wild_df['Purchase Date'].dt.to_period('Q')
# wild_df['Year'] = wild_df['Purchase Date'].dt.year

# # Remove '$' from 'Transaction Amount ' and convert to float
# wild_df['Transaction Amount'] = wild_df['Transaction Amount '].replace('[\$,]', '', regex=True).astype(float)

# # Aggregate data by quarter and month
# quarterly_counts = wild_df.groupby('Quarter').size()
# quarterly_amounts = wild_df.groupby('Quarter')['Transaction Amount'].sum()
# monthly_counts = wild_df.groupby('Month').size()
# monthly_amounts = wild_df.groupby('Month')['Transaction Amount'].sum()

# # Group data by month and year, and count the number of purchases
# monthly_purchases = wild_df.groupby(['Year', 'Month']).size()

# # Confirm
# print(monthly_purchases.head())


# # Plotting 4 line graphs
# plt.figure(figsize=(14, 10), facecolor='white')

# plt.subplot(2, 2, 1)
# plt.plot(quarterly_counts.index.astype(str), 
#          quarterly_counts.values, 
#          marker='o', 
#          color='blue', 
#          label='Transaction Count by Quarter')
# plt.title('Transaction Count by Quarter')
# plt.legend()

# plt.subplot(2, 2, 2)
# plt.plot(quarterly_amounts.index.astype(str), 
#          quarterly_amounts.values, 
#          marker='o', 
#          color='green', 
#          label='Transaction Amount by Quarter')
# plt.title('Transaction Amount by Quarter')
# plt.legend()

# plt.subplot(2, 2, 3)
# plt.plot(monthly_counts.index.astype(str), 
#          monthly_counts.values, 
#          marker='o', 
#          color='red', 
#          label='Transaction Count by Month')
# plt.title('Transaction Count by Month')
# plt.legend()

# plt.subplot(2, 2, 4)
# plt.plot(monthly_amounts.index.astype(str), 
#          monthly_amounts.values, 
#          marker='o', 
#          color='purple', 
#          label='Transaction Amount by Month')
# plt.title('Transaction Amount by Month')
# plt.legend()

# plt.tight_layout()
# plt.show()



# # Plotting a bar chart for monthly purchase trends - v1
# # V1: Hardcode the month names
# plt.figure(figsize=(10, 6), facecolor='white')
# plt.bar(monthly_counts.index, monthly_counts.values, color='skyblue')
# plt.title('Monthly Purchase Trends')
# plt.xlabel('Month')
# plt.ylabel('Number of Transactions')
# plt.xticks(monthly_counts.index, ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'])
# plt.grid(axis='y', linestyle='--', alpha=0.7)
# plt.show()



# # Plotting a bar chart for monthly purchase trends - v2
# # V2: Extract the year/month dates
# plt.figure(figsize=(12, 6))
# monthly_purchases.plot(kind='bar', color='skyblue')
# plt.title('Monthly Purchase Trends')
# plt.xlabel('Year and Month')
# plt.ylabel('Number of Purchases')
# plt.xticks(rotation=45)
# plt.grid(axis='y')
# plt.tight_layout()
# plt.show()

















'''
    HERE WE USE A NEW MODULE FOR STATS AND USE THE PROBABILITY DENSITY FUNCTION (PDF) TO GENERATE GAUSSIAN WEIGHTS.

    A Gaussian distribution, also commonly referred to as a normal distribution, is a type of continuous probability distribution that is symmetric around its mean, showing that data near the mean are more frequent in occurrence than data far from the mean. In graph form, this distribution will appear as a bell curve.

    Key Characteristics of Gaussian Distribution:
    
    ~ Symmetry: The distribution is perfectly symmetrical around its mean.
    
    ~ Mean, Median, and Mode: In a Gaussian distribution, the mean, median, and mode are all equal.
    
    ~Empirical Rule: About 68% of the data falls within one standard deviation of the mean, 95% within two standard deviations, and nearly all (99.7%) within three standard deviations.
    
    ~ Inflection Points: The curve changes concavity at points one standard deviation away from the mean. These are called the points of inflection.
'''


# import pandas as pd
# import numpy as np
# from scipy.stats import norm, skewnorm

# # Load the EQUIP-CHEMICALS.csv data
# chem_df = pd.read_csv('./CSVs/EQUIP-CHEMICALS.csv')

# # Display the head of the dataframe to understand its structure
# print(chem_df.head())
# print(chem_df.info())


'''APPLY A NORM (IS THAT RIGHT?) TRANSFORM'''
# # Calculate the weights for a Gaussian distribution
# mean_price = chem_df['Total Price'].mean()
# std_price = chem_df['Total Price'].std()

# # Generate weights using the Gaussian formula
# chem_df['Weights'] = norm.pdf(chem_df['Total Price'], 
#                               mean_price, 
#                               std_price)

# # Output the weights
# print(chem_df[['Total Price', 'Weights']].head())

# # Calculate parameters for a positively skewed distribution
# a = 4  # Positive skewness parameter
# loc = chem_df['Total Price'].min()  # Location parameter
# scale = chem_df['Total Price'].std()  # Scale parameter

# # Generate weights for the positively skewed distribution
# chem_df['Skewed_Weights'] = skewnorm.pdf(chem_df['Total Price'], 
#                                          a, 
#                                          loc, 
#                                          scale)

# # Output the updated weights
# print(chem_df[['Total Price', 'Skewed_Weights']].head())


'''APPLYING A LOG TRANSFORM (ONLY DEFINED FOR POSITIVE REALS)'''
# # Load the EQUIP-CHEMICALS.csv data
# df = pd.read_csv('./CSVs/EQUIP-CHEMICALS.csv')

# # Since log(0) is undefined, we adjust 0 values to a small positive value (e.g., 1) before transformation
# df['Total Price Adjusted'] = df['Total Price'].apply(lambda x: x if x > 0 else 1)

# # Apply log transformation to 'Total Price' to skew the distribution positively
# df['Total Price Log'] = np.log(df['Total Price Adjusted'])

# # Calculate mean and standard deviation of the log-transformed 'Total Price'
# mean_log = df['Total Price Log'].mean()
# std_dev_log = df['Total Price Log'].std()

# # Apply the Gaussian distribution formula to the log-transformed values
# weights_log = norm.pdf(df['Total Price Log'], loc=mean_log, scale=std_dev_log)

# # Print the updated weights
# print(weights_log)


 















'''
    HERE WE HAVE A LOT OF COOL NEW FUNCTIONALITY AND TWO CUSTOM BUILT METHODS USED FOR MODIFYING ENTRIES
'''

# import pandas as pd
# import matplotlib.pyplot as plt

# # Load the HOA transactions data
# hoa_df = pd.read_csv('./CSVs/hoa_transactions.csv')

# # Display the first few rows of the dataframe to understand its structure
# print(hoa_df.head())
# print(hoa_df.dtypes)
# print(hoa_df.info())


'''start tests'''
# uniq_dates = hoa_df['Date'].unique()
# print(f'Unique dates:\n{uniq_dates}')

# count_dates = hoa_df['Date'].value_counts()
# print(f'Their individual counts (BEFORE):\n{count_dates}')

# count_dates = hoa_df['Date'].value_counts().sum()
# print(f'Their TOTAL counts:\n{count_dates}')
'''end tests'''


'''converting dates with different formats'''
# # Function to parse dates with multiple formats (for this specific dataset)
# def parse_date(date_str):
#     if not isinstance(date_str, float) and len(date_str) > 2:
#         formats = ['%d-%b', '%b-%d']
#         for fmt in formats:
#             try:
#                 return pd.to_datetime(date_str, format=fmt)
#             except ValueError:
#                 pass
#     return None

# # Apply function to the date column for conversions
# hoa_df['Date'] = hoa_df['Date'].apply(parse_date)


'''start tests'''
# # Display unique non-parsed dates and the number of missing dates after conversion
# non_parsed_dates = hoa_df[hoa_df['Date'].isna()]['Date'].unique()
# print(f'Non parsed:\n{non_parsed_dates}') 

# missing_dates_count = hoa_df['Date'].isna().sum()
# print(f'Missing:\n{missing_dates_count}')

# count_dates = hoa_df['Date'].value_counts()
# print(f'Their individual counts (AFTER):\n{count_dates}')

# count_dates = hoa_df['Date'].value_counts().sum()
# print(f'Their counts (AFTER):\n{count_dates}')

# uni_total = hoa_df['Total'].value_counts().sum()
# print(f'Unique totals count (BEFORE):\n{uni_total}')
'''end tests'''


# # Clean the 'Total' column by removing commas and parentheses
# hoa_df['Total'] = hoa_df['Total'].str.replace('[(),]', '', regex=True)
# hoa_df['Total'] = pd.to_numeric(hoa_df['Total'])


'''start tests'''
# uni_total = hoa_df['Total'].value_counts().sum()
# print(f'Unique totals count (AFTER):\n{uni_total}')

# uni_units = hoa_df['Unit'].value_counts()
# print(f'Unique units:\n{uni_units}')
# total_units = uni_units.sum()
# print(f'Total number of units is:\n{total_units}')
'''end tests'''


'''spliting numeric/integer entries'''
# # Function to split the Unit column into Building and Unit
# def split_unit(value):
#      # Check if the value is fully numeric
#     if value.isnumeric(): 
#         # Split into first digit and the rest
#         return value[0], value[1:]  
#     else:
#         # Non-numeric values go entirely into Unit
#         return '', value  

# # Apply the function to the Unit column
# hoa_df[['Building', 'Unit']] = hoa_df['Unit'].apply(lambda x: split_unit(str(x)) if pd.notnull(x) else ('', '')).tolist()

# # Display the modified DataFrame to confirm changes
# print(hoa_df[['Building', 'Unit']].head(n=20))


# # Analyzing the distribution of income across different units
# income_per_unit = hoa_df.groupby('Unit')['Income'].sum().dropna()
# print(f'Income per unit:\n{income_per_unit}')
# print('\nEND')

# # Plotting the distribution of income per unit
# plt.figure(figsize=(12, 8), facecolor='white')
# income_per_unit.plot(kind='bar', color='skyblue')
# plt.title('Distribution of Income Across Units')
# plt.xlabel('Unit')
# plt.ylabel('Total Income')
# plt.xticks(rotation=45)
# plt.grid(True)
# plt.show()

# # Descriptive statistics for income distribution
# income_stats = income_per_unit.describe()
# print(income_stats)

# # Plotting the scatter plot for both Expenses and Income over Time
# plt.figure(figsize=(12, 8), facecolor='white')
# plt.scatter(hoa_df['Date'], hoa_df['Expenses'], color='blue', alpha=0.5, label='Expenses')
# plt.scatter(hoa_df['Date'], hoa_df['Income'], color='red', alpha=0.5, label='Income')
# plt.title('Scatter Plot of Expenses and Income Over Time')
# plt.xlabel('Date')
# plt.ylabel('Amount')
# plt.legend()
# plt.grid(True)
# # plt.show()

# # Calculating the correlation coefficients
# expenses_income_corr = hoa_df[['Expenses', 'Income']].corr().iloc[0, 1]
# # print('Correlation coefficient between Expenses and Income:', expenses_income_corr)

# corr1 = hoa_df['Date'].corr(hoa_df['Income'])
# print(f'Correlation coefficient between Date (as timestamp) and Income: {corr1}')

# corr2 = hoa_df['Date'].corr(hoa_df['Expenses'])
# print(f'Correlation coefficient between Date (as timestamp) and Income: {corr2}')






















# import pandas as pd

# # Load the PAYROLL_MAY.xlsx file
# payroll_df = pd.read_excel('./CSVs/PAYROLL_MAY.xlsx')

# # Display the head of the dataframe to understand its structure
# print(payroll_df.head(n=15))
# print(payroll_df.info())

# # Confirm there are only two types of expenses
# exp_counts = payroll_df['EXPENSE'].value_counts()
# print(f'Unique type of expenses:\n{exp_counts}')

# # Splitting the EXPENSE column into INDIRECT EXPENSE and GENERAL EXPENSE
# payroll_df['INDIRECT EXPENSE'] = payroll_df['EXPENSE'].apply(lambda x: x if 'INDIRECT' in x else '')
# payroll_df['GENERAL EXPENSE'] = payroll_df['EXPENSE'].apply(lambda x: x if 'GENERAL' in x else '')

# Display the modified DataFrame to confirm changes
# print(payroll_df[['EXPENSE', 'INDIRECT EXPENSE', 'GENERAL EXPENSE']].head())

# num_in_ex = payroll_df['INDIRECT EXPENSE'].value_counts()
# print(num_in_ex)

# uni_status = payroll_df['STATUS'].value_counts()
# print(f'Unique statuses:\n{uni_status}')

# # Filtering the DataFrame for records with 'CESSED' status and displaying the new columns
# cessed_records = payroll_df[payroll_df['STATUS'] == 'CESSED'][['INDIRECT EXPENSE', 'GENERAL EXPENSE']]
# print(cessed_records)



















# # Print current working directory
# import os
# print("Current Working Directory:", os.getcwd())

# # List files in the specific directory
# print("Files in './CSVs/':", os.listdir('./CSVs/'))









'''
    TWO WAYS TO FORMAT STRINGS LIKE MONEY/CURRENCY
'''

# import pandas as pd

# # Load the Excel file 
# file_path_inventory = './CSVs/dataset.xls'
# data_inventory = pd.read_excel(file_path_inventory)

# # Explore the file
# print(data_inventory.head())
# print(data_inventory.info())


# # Calculate the total cost for items in inventory as per the 'Storeroom Total' column
# total_storeroom_cost = data_inventory['Storeroom Total'].sum()
# formatted_total1 = f"${total_storeroom_cost:,.2f}"
# print(formatted_total1)

# # Calculate the total estimated value based on List Price and Storeroom Quantity
# data_inventory['Estimated Storeroom Value'] = data_inventory['List Price'] * data_inventory['Storeroom Quantity']
# total_estimated_storeroom_value = data_inventory['Estimated Storeroom Value'].sum()

# # Format the total estimated value as currency
# formatted_total2 = "${:,.2f}".format(total_estimated_storeroom_value)
# print("\nTotal est. storeroom val:")
# print(formatted_total2)









































'''
    SEGMENTATION
'''
# import pandas as pd
# import matplotlib.pyplot as plt

# # Load the Excel file to see the first few rows and understand its structure
# file_path_gas = './CSVs/Gas_Prices.xlsx'
# gas_prices_data = pd.read_excel(file_path_gas)
# # print(gas_prices_data.head())
# print(gas_prices_data.info())

# # Extract the Price Per Gallon (USD) for India
# price_per_gallon_india = gas_prices_data.loc[gas_prices_data['Country'] == 'India', 'Price Per Gallon (USD)'].values[0]
# print("\n")
# # print(price_per_gallon_india)

# # Display basic statistics and a histogram for 'Gallons GDP Per Capita Can Buy'
# gallons_stats = gas_prices_data['Gallons GDP Per Capita Can Buy'].describe()
# fig, ax = plt.subplots()
# gas_prices_data['Gallons GDP Per Capita Can Buy'].hist(ax=ax, bins=10, color='skyblue', edgecolor='black')
# ax.set_title('Distribution of Gallons GDP Per Capita Can Buy')
# ax.set_xlabel('Gallons GDP Per Capita Can Buy')
# ax.set_ylabel('Frequency')

# # print(gallons_stats) 
# print("\n")
# plt.show()


'''APPROACH 1: BY QUARTILES'''
# # Calculate percentiles for the segmentation
# percentile_25 = gas_prices_data['Gallons GDP Per Capita Can Buy'].quantile(0.25)
# percentile_50 = gas_prices_data['Gallons GDP Per Capita Can Buy'].quantile(0.50)
# percentile_75 = gas_prices_data['Gallons GDP Per Capita Can Buy'].quantile(0.75)

# # Define bins and labels based on these percentiles
# bins_percentiles = [0, percentile_25, percentile_50, percentile_75, gas_prices_data['Gallons GDP Per Capita Can Buy'].max()]
# labels_percentiles = ['25th Percentile', '50th Percentile', '75th Percentile', '100th Percentile']

# # Segment data using these new bins
# gas_prices_data['GDP Per Capita Gallons Percentile'] = pd.cut(
#     gas_prices_data['Gallons GDP Per Capita Can Buy'], 
#     bins=bins_percentiles, 
#     labels=labels_percentiles)

# # Select and display the segmented data
# segmented_data = gas_prices_data[['Country', 
#                                   'Gallons GDP Per Capita Can Buy', 
#                                   'Price Per Gallon (USD)', 
#                                   'GDP Per Capita Gallons Percentile']]
# segmented_data.groupby('GDP Per Capita Gallons Percentile', observed=True).apply(lambda x: x[['Country', 'Price Per Gallon (USD)']])









'''APPROACH 2: BY QUARTILES'''
# # Calculate 25th, 50th, and 75th percentiles and round to nearest integer
# low = gas_prices_data['Gallons GDP Per Capita Can Buy'].quantile(0.25).round()
# medium = gas_prices_data['Gallons GDP Per Capita Can Buy'].quantile(0.50).round()
# high = gas_prices_data['Gallons GDP Per Capita Can Buy'].quantile(0.75).round()

# # Create a new column for segment based on the calculated percentiles
# gas_prices_data['Gallons GDP Per Capita Can Buy Segment'] = pd.cut(
#     gas_prices_data['Gallons GDP Per Capita Can Buy'], 
#     bins=[-float('inf'), low, medium, high, float('inf')],
#     labels=['Low', 'Medium', 'High', 'Very High'])


# # Filter to relevant columns and sort by the new segment column
# df_filtered = gas_prices_data[['Price Per Gallon (USD)', 
#                                'Country', 
#                                'Gallons GDP Per Capita Can Buy Segment']].sort_values('Gallons GDP Per Capita Can Buy Segment')

# # Display the filtered and sorted dataframe
# print(df_filtered)











'''APPROACH 3: BY COUNTRY GAL. QUANT. AFFORDABILITY'''
# # Define the bins and labels for the segmentation
# bins = [0, 1000, 5000, gas_prices_data['Gallons GDP Per Capita Can Buy'].max()]
# labels = ['Low', 'Medium', 'High']

# # Segment the data
# gas_prices_data['Segment'] = pd.cut(
#     gas_prices_data['Gallons GDP Per Capita Can Buy'], 
#     bins=bins, 
#     labels=labels, 
#     right=False)

# # Calculate the number of countries in each segment
# segment_counts = gas_prices_data['Segment'].value_counts().sort_index()

# # Calculate basic statistics for each segment
# segment_stats = gas_prices_data.groupby('Segment', observed=False).agg({
#     'Gallons GDP Per Capita Can Buy': ['mean', 'median', 'std', 'min', 'max'],
#     'Price Per Gallon (USD)': ['mean', 'median', 'std', 'min', 'max'],
#     'GDP Per Capita ( USD )': ['mean', 'median', 'std', 'min', 'max'],
#     'Yearly Gallons Per Capita': ['mean', 'median', 'std', 'min', 'max'],
# })

# print(segment_counts)
# print(segment_stats)

# # Save to csv out
# output_file_path = './OutCSVs/segmentedGas.csv'
# segment_stats.to_csv(output_file_path, index=False)

# print("\n")
# print("\n")
# # print(segmented_data)
# print("\n")
# print("\n")

# # Create a formatted output to clearly show results for each percentile segment
# formatted_results = segmented_data.groupby('GDP Per Capita Gallons Percentile', observed=True) \
#                                    .apply(lambda x: x[['Country', 'Price Per Gallon (USD)']].to_dict('records')) \
#                                    .reset_index()
# formatted_results.rename(columns={0: 'Details'}, inplace=True)
# # print(formatted_results)
# print("\n")
# print("\n")















'''
    SCATTER PLOT PARAMETERS (NON-EXHAUSTIVE)

data: This specifies the DataFrame that contains the data to plot. In this case, it's heart_data_reuploaded.
x: The name of the column in the DataFrame to use for the x-coordinates of the points in the plot. Here, it's 'age'.
y: The name of the column for the y-coordinates. Here, it's 'DEATH_EVENT'.
hue: This parameter determines which column in the DataFrame should be used to color the points. By setting this to 'smoking', points are colored based on whether individuals are smokers or not.
style: It's also set to 'smoking' here. This changes the marker style based on the smoking status. You can use it to differentiate points not only by color but also by shape.
s: Size of the markers (points on the plot). You can increase or decrease this value to make points larger or smaller.
alpha: This controls the transparency of the points. A lower value makes the points more transparent, which can be useful when you have overlapping points.
palette: This defines the color palette used for different categories in hue. Here, 'coolwarm' is used, which provides a range from cool to warm colors. You can experiment with other palettes like 'viridis', 'plasma', 'inferno', 'magma', or even custom palettes like ['red', 'blue', 'green'].

    HERE WE COMPARE SMOKE, AGE, AND DEATH COLUMNS
'''


# import matplotlib.pyplot as plt
# import seaborn as sns
# import pandas as pd

# # Reload the heart_failure_clinical_records_dataset
# heart_data_path = './CSVs/heart_failure_clinical_records_dataset.csv'
# heart_data = pd.read_csv(heart_data_path)

# # Display the first few rows and the columns of the dataset
# print(heart_data.head()) 
# print(heart_data.columns)
# print(heart_data.info())


'''start testing'''
# uni_smoke = heart_data['smoking'].value_counts()
# uni_dead = heart_data['DEATH_EVENT'].value_counts()
# print(f'death:\n{uni_dead}')
# print(f'smoke:\n{uni_smoke}')

# # Count of non-smoking survivors
# non_smoking_survivors_count = heart_data[(heart_data['smoking'] == 0) & (heart_data['DEATH_EVENT'] == 0)].shape[0]

# # Count of smokers that died
# smokers_died_count = heart_data[(heart_data['smoking'] == 1) & (heart_data['DEATH_EVENT'] == 1)].shape[0]

# print("Number of non-smoking survivors:", non_smoking_survivors_count)
# print("Number of smokers that died:", smokers_died_count)
'''end testing'''


'''APPROACH 1: SIMPLE (MIGHT NOT BE THE MOST EFFICIENT)'''
# # Set style for the plots
# sns.set_theme(style="whitegrid")

# # Create a scatter plot
# plt.figure(figsize=(10, 6))
# sns.scatterplot(data=heart_data, 
#                 x='age', 
#                 y='smoking', 
#                 hue='DEATH_EVENT', 
#                 style='DEATH_EVENT', 
#                 size='DEATH_EVENT',
#                 sizes=(50,150), 
#                 alpha=1, 
#                 palette='coolwarm')

# plt.title('Impact of Age and Smoking on Mortality in Heart Failure Patients', fontsize=16)
# plt.xlabel('Age', fontsize=16)
# plt.ylabel('Smoking Status', fontsize=16)
# plt.legend(title='Smoking Status', 
#            labels=['Survived', 
#                    'Deceased'])
# plt.show()



'''APPROACH 2: MORE SOPHISTICATED (MORE COMPONENTS GIVES MORE CUSTOMIZATION)'''
# # Set the aesthetic style of the plots
# sns.set_style("whitegrid")

# # Create a figure and a set of subplots
# fig, ax = plt.subplots(figsize=(10, 6))

# # Plot the data
# sns.scatterplot(data=heart_data, 
#                 x='age', 
#                 y='smoking', 
#                 hue='DEATH_EVENT', 
#                 style='DEATH_EVENT', 
#                 size='DEATH_EVENT', 
#                 sizes=(50, 150), 
#                 alpha=0.6, 
#                 ax=ax)

# # Customizing the plot
# ax.set_title('Impact of Age and Smoking on Health Outcomes', fontsize=16)
# ax.set_xlabel('Age', fontsize=14)
# ax.set_ylabel('Smoking Status', fontsize=14)
# ax.set_yticks([0, 1])
# ax.set_yticklabels(['Non-Smoker', 'Smoker'])
# ax.legend(title='Death Event', 
#           labels=['Survived', 
#                   'Deceased'])

# plt.tight_layout()
# plt.show()

'''start testing'''

'''end testing'''




















'''find out what this does'''
# import pandas as pd

# FILEPATH = './CSVs/SOLDFOOD2023 - Winter.xlsx'
# dataframes = pd.read_excel(FILEPATH, header=3)

# # print each dataframe name
# print("Dataframe keys of dataframes:" + ", ".join(dataframes.keys()))

# for k, v in dataframes.items():
#     # strip whitespace where possible from column names; need to check if isinstance(x, str) because some column names are numbers
#     try:
#         v = v.rename(columns=lambda x: x.strip() if isinstance(x, str) else x)
#     except:
#         pass

#     # strip whitespace where possible from cells
#     try:
#         v = v.apply(lambda col: col.str.strip() if col.dtype == "object" else col)
#     except:
#         pass
#     dataframes[k] = v
#     print('dataframe: '+ k)
#     print(v.head(15))
'''find out what this ^^^^ does'''









'''
    HERE WE HAVE A MULTI-SHEET EXCEL, WITH 3 ROWS OF HEADER AND 1 ROW OF FOOTER, AND WE COMBINE THEM INTO ONE. WE THEN AGGREGATE TWO DISTINCT COLUMNS.
'''


# import pandas as pd
# import matplotlib.pyplot as plt

# # Load the Excel file
# file_path = './CSVs/SOLDFOOD2023 - Winter.xlsx'
# xls = pd.ExcelFile(file_path)

# # Sheet names
# sheet_names = xls.sheet_names

# # Load and combine data from all sheets, skip the first three rows and the last footer row
# combined_data = pd.concat(
#     [xls.parse(sheet_name, skiprows=3, skipfooter=1) for sheet_name in sheet_names],
#     ignore_index=True
# )

# print(combined_data.head(n=15))
# print(combined_data.info())

'''get some summary stats'''
# # Calculate unique products
# unique_products = combined_data['CODE'].nunique()

# # Calculate average price and standard deviation
# average_price = combined_data['PRICE'].mean()
# price_std = combined_data['PRICE'].std()

# # Calculate average quantity sold
# average_quantity = combined_data['QUANTITY'].mean()

# # Identify outliers in quantity
# q1 = combined_data['QUANTITY'].quantile(0.25)
# q3 = combined_data['QUANTITY'].quantile(0.75)
# iqr = q3 - q1
# lower_bound = q1 - 1.5 * iqr
# upper_bound = q3 + 1.5 * iqr
# outliers = combined_data[
#     (combined_data['QUANTITY'] < lower_bound) | 
#     (combined_data['QUANTITY'] > upper_bound)
#     ]['QUANTITY']

# print(f'Unique products: {unique_products}')
# print(f'Average price: {average_price:.2f}, Standard deviation of price: {price_std:.2f}')
# print(f'Average quantity sold: {average_quantity:.2f}')
# print(f'Outliers in quantity: {outliers.values}')

# # Summarize total sales by product group for all months combined
# combined_sales_summary = combined_data.groupby('GROUP').agg({'TOTAL SALE': 'sum', 'QUANTITY': 'sum'})
# print('Aggregated data by product group:\n')
# print(combined_sales_summary)

# # Group by CODE and aggregate
# aggregated_data = combined_data.groupby('CODE').agg({'QUANTITY': 'sum', 'TOTAL SALE': 'sum'}).reset_index()
# print('Aggregated data by product code:\n')
# print(aggregated_data)

# # Plotting
# plt.figure(figsize=(10, 6), facecolor='white')
# plt.bar(aggregated_data['CODE'], aggregated_data['TOTAL SALE'], color='blue')
# plt.title('Total Sales by Product Code')
# plt.xlabel('Product Code')
# plt.ylabel('Total Sales')
# plt.grid(True)
# plt.show()























'''
    HERE WE PRINT SOME STATS WITH TWO DECIMAL PLACES.

    PRINT ENTRIES TO HAVE TWO DECIMAL PLACES
'''

# import pandas as pd

# # Load the data from Sheet3 which is assumed to be the South America data
# south_america_df = pd.read_excel('./CSVs/population_and_age_1.xlsx', sheet_name='Sheet3')
# print(south_america_df.info())

# # Calculate average age and population
# average_age = south_america_df['Average Age'].mean()
# average_population = south_america_df['Population'].mean()

# print(f'Average age across all countries in South America: {average_age:.2f}')
# print(f'Average population across all countries in South America: {average_population:.2f}')



















'''
    HERE WE HAVE SEVERAL DIFF WAYS OF GENERATING MOSTLY THE SAME DATA, SOME GEN. DIFF DATA, BUT THEY ARE ALL SIMILAR. 

    NOTE: YOU NEED TO FIGURE OUT THE MOST EFFICIENT WAY OF DOING THINGS BY TRIAL, ERROR, AND ELIMINATION OF CUMBERSOME COMMANDS. E.G., WHATS THE BETTER WAY OF CLEANING THE TOTAL COLUMN? REPLACING EVERY UNWANTED CHAR OR OMITTING EVERYTHING THAT ISNT WHAT YOU WANT? BOTH APPROACHES ARE BELOW...
'''


# import pandas as pd

# # Load the data
# file_path = './CSVs/STRAWBERRY SALES 2023 - Sheet1.tsv'

'''APPROACH 1'''
# data = pd.read_csv(file_path, sep='\t', skiprows=2)

# data.columns = [
#     'Date', 'Clamshells', 'Boxes', 'Kilos', 'Price per Box',
#     'Total Sales', 'Product', 'Type of Product'
# ]

# # Convert the 'Date' column to datetime
# data['Date'] = pd.to_datetime(data['Date'], dayfirst=True)

# # Remove '$' and spaces from 'Total Sales' and convert to float
# data['Total Sales'] = data['Total Sales'].replace({',': '', '\$': '', ' ': '', '\'': ''}, regex=True).astype(float)

# # Display the first few rows of the dataframe and its column names
# print(data.head(n=20))
# print(data.columns)
# print(data.info())

# # Extract month and year from the 'Date' column for grouping
# data['Month'] = data['Date'].dt.to_period('M')
# m = data['Month']
# # print(f'The new month column:\n{m}')

# # Group data by the new 'Month' column and calculate required metrics
# monthly_sales = data.groupby('Month').agg(
#     Total_Sales=('Total Sales', 'sum'),
#     Average_Day=('Date', lambda x: (x.dt.day).mean()),
#     Median_Day=('Date', lambda x: (x.dt.day).median())
# ).reset_index()

# print(f'Monthly sales:\n{monthly_sales}')


'''APPROACH 2'''
# # Extract month and year from the `DATE` column
# data['Date'] = pd.to_datetime(data['Date'], format='%d-%b-%y')
# data['MONTH'] = data['Date'].dt.month_name()
# data['YEAR'] = data['Date'].dt.year

# # Calculate total sales for each month
# monthly_sales = data.groupby(['MONTH', 'YEAR'])['Total Sales'].sum()

# # Calculate average daily sales for each month
# average_daily_sales = data.groupby(['MONTH', 'YEAR'])['Total Sales'].mean()

# # Calculate mean day for each month
# mean_day = data.groupby(['MONTH', 'YEAR'])['Date'].apply(lambda x: int(x.dt.day.mean()))

# # Combine the three series into a DataFrame
# result_df = pd.DataFrame({'Total Sales': monthly_sales, 'Average Day': average_daily_sales, 'Mean Day': mean_day})

# # Format the `average_daily_sales` column to two decimal places
# result_df['Average Day'] = result_df['Average Day'].apply(lambda x: f'${x:.2f}')

# # Print the final DataFrame
# print(result_df.to_markdown(numalign="left", stralign="left"))

# # Save the final DataFrame to a new tsv file
# result_df.to_csv('monthly_strawberry_sales_summary.tsv', sep='\t')


'''APPROACH 3'''
# data = pd.read_csv(file_path, sep='\t')

# data.columns = [
#     'DATE', 'Clamshells', 'Boxes', 'Kilos', 'Price per Box',
#     'TOTAL', 'Product', 'Type of Product'
# ]

# # Convert the 'Date' column to datetime
# data['DATE'] = pd.to_datetime(data['DATE'], dayfirst=True)

# # Remove '$' and spaces from 'Total Sales' and convert to float
# data['TOTAL'] = data['TOTAL'].replace({',': '', '\$': '', ' ': '', '\'': ''}, regex=True).astype(float)

# Clean the data
# Removing the initial rows and setting the correct column names
# data.columns = data.iloc[1]  # Set the correct column headers
# data = data[2:]  # Remove the first two rows

# Reset index
# data.reset_index(drop=True, inplace=True)
# print(data.head(n=15))
# print(data.info())
# print(f'Cols:\n{data.columns}')
# print('\nEND')

# # Convert 'DATE' to datetime and 'TOTAL' to numeric after removing the $ sign
# data['DATE'] = pd.to_datetime(data['DATE                         '])
# data['TOTAL'] = data['TOTAL       '].replace('[\$,]', '', regex=True).astype(float)

# # Extracting month and year from DATE for grouping
# data['Month'] = data['DATE'].dt.month
# data['Year'] = data['DATE'].dt.year

# # Calculate total sales, average sales per day, and mean sales day for each month
# # Assuming "mean day" implies finding the median sale value and identifying the corresponding day(s)

# # Group by month and year
# monthly_data = data.groupby(['Year', 'Month']).agg(
#     Total_Sales=('TOTAL', 'sum'),
#     Average_Sales_Per_Day=('TOTAL', 'mean')
# ).reset_index()

# # For finding the "mean day" or median sales day, we need to compute the median sale value for each month
# # and then find the day(s) that are closest to this median value in terms of sales

# def find_median_day(group):
#     median_sales = group['TOTAL'].median()
#     closest_to_median = group.iloc[(group['TOTAL'] - median_sales).abs().argsort()[:1]]
#     return closest_to_median['DATE'].dt.day.values[0]

# monthly_data['Median_Sales_Day'] = data.groupby(['Year', 'Month']).apply(find_median_day).reset_index(level=[0,1], drop=True)


# print(monthly_data)


























'''
    HERE, WE ILLUSTRATE A TECHNIQUE TO MAKE THE GROUP BY METHOD MORE EFFICIENT IN THE CASE OF COLUMN LOOK UPS!
'''
# import pandas as pd

# # Read the CSV files into Pandas DataFrames
# df_alcohol_drug_abuse = pd.read_excel('./CSVs/Hospital_Survey_Data_Alcohol_Drug_Abuse.xlsx', skiprows=1)
# df_speticemia = pd.read_csv('./CSVs/Hospital_Survey_Data_Speticemia.csv', skiprows=1)

# # Explore the data's structure
# print("First 5 rows of Alcohol_Drug_Abuse data:")
# print(df_alcohol_drug_abuse.head())
# print("\nAlcohol_Drug_Abuse data information:")
# print(df_alcohol_drug_abuse.info())
# print("\nFirst 5 rows of Speticemia data:")
# print(df_speticemia.head())
# print("\nSpeticemia data information:")
# print(df_speticemia.info())


# # Concatenate the two dataframes
# df_combined = pd.concat([df_alcohol_drug_abuse, df_speticemia])

'''
Efficiency in Column Handling:
Code Block 1 performs the grouping on the entire DataFrame and then specifies which columns to aggregate within the agg() method. This means that the grouping considers all columns, which can be less efficient if the DataFrame has many columns that are not needed for the final aggregation.
Code Block 2 is more efficient in scenarios where the DataFrame has many irrelevant columns. By selecting only the necessary columns immediately after groupby(), it reduces the amount of data processed in the subsequent aggregation step.
'''

# # Aggregate by state (block 1)
# df_agg = df_combined.groupby('Provider State').agg(
#     Total_Discharges=('Total Discharges', 'sum'),
#     Average_Total_Payments=('Average Total Payments ($)', 'mean')
# ).reset_index()



# # Aggregate by state (block 2: more efficient)
# df_agg = df_combined.groupby('Provider State')[['Total Discharges', 'Average Total Payments ($)']].agg(
#     Total_Discharges=('Total Discharges', 'sum'),
#     Average_Total_Payments=('Average Total Payments ($)', 'mean')
# ).reset_index()



# # Rename columns
# df_agg = df_agg.rename(columns={'Total_Discharges': 
#                                 'Total Discharges', 
#                                 'Average_Total_Payments': 
#                                 'Average Total Payments ($)'})

# # Print the table
# print(f'\nSum discharges and Avg. payments per state:\n\n{df_agg}')

























'''
    CLEAN UP COLUMN NAMES THAT HAVE NEWLINE CHARS
'''

# import matplotlib.pyplot as plt
# import pandas as pd

# # Load the sushi_earns.xlsx file
# sushi_df = pd.read_excel('./CSVs/sushi_earns.xlsx')

# # Explore the dataframe to understand its structure
# print(sushi_df.head())
# print(sushi_df.info())

# # Clean up the column names newline char
# sushi_df.columns = sushi_df.columns.str.replace('\n', '', regex=True)

# # Confirm it worked
# print(sushi_df.head())
# print(sushi_df.info())

# uni_categories = sushi_df['Category'].value_counts()
# print(f'Unique categories:\n{uni_categories}')

# # Aggregate the total revenue by category
# revenue_by_category = sushi_df.groupby('Category')['revenue_from_applicable_discounts'].sum().sort_values(ascending=False)

# # Plotting the total revenue by category
# plt.figure(figsize=(10, 6), facecolor='white')
# revenue_by_category.plot(kind='bar', color='skyblue')
# plt.title('Total Revenue by Product Category')
# plt.xlabel('Product Category')
# plt.ylabel('Total Revenue')
# plt.xticks(rotation=45)
# plt.grid(axis='y', linestyle='--', alpha=0.7)
# plt.tight_layout()
# plt.show()

# # Display the aggregated data
# print(revenue_by_category)



# # Calculate total revenue (THIS SHOULD BE COLUMNS WITH "REVENUE" IN THE NAME)
# df_menu['Total_revenue'] = df_menu['Cash_revenue'] + df_menu['Pedidosya_revenue'] + df_menu['Shea_app_revenue']

# # Group by `Category` and sum up `Total_revenue`
# df_agg = df_menu.groupby('Category')['Total_revenue'].sum().reset_index()






















'''
    HERE WE SHOW HOW TO CREATE A BAR CHART WITH THE GRID BACKGROUND, HOWEVER, THE GRID'S LINES ARE DOTTED!
'''

# import pandas as pd
# import matplotlib.pyplot as plt

# # Load the SOLDFOOD2023 - Winter.xlsx file, skipping the first 3 rows
# food_df = pd.read_excel('./CSVs/SOLDFOOD2023 - Winter.xlsx', skiprows=3)

# # Display the first few rows of the dataframe to understand its structure
# print(food_df.head())

# # Aggregate the total quantity sold by group
# quantity_by_group = food_df.groupby('GROUP')['QUANTITY'].sum().sort_values(ascending=False)

# # Plotting the total quantity sold by group
# plt.figure(figsize=(10, 6), facecolor='white')
# quantity_by_group.plot(kind='bar', color='lightgreen')
# plt.title('Total Quantity Sold by Product Category')
# plt.xlabel('Product Category')
# plt.ylabel('Total Quantity Sold')
# plt.xticks(rotation=45)
# plt.grid(axis='y', linestyle='--', alpha=0.7)
# plt.tight_layout()
# plt.show()

# # Display the aggregated data
# print(quantity_by_group)






















'''
    HERE WE HAVE SOME VERY NEW FUNCTIONALITY! 
    
    1. FIRST, WE USE PANDAS TO GENERATE A PIE CHART BUT WE EXPLODE ONE OF THE SLICES TO EMPHASIZE IT... 

    2. THEN, WE INVOKE A NEW MODULE NAMED ALTAIR, WHICH HELPS WITH VISUALIZATIONS (I THINK, STILL GOTTA CONFIRM). WE USE IT TO GENERATE A PIE CHART AND SAVE THE CHART TO A HTML FILE SO THAT WE CAN RENDER IT IN THE BROWSER.
'''

'''CHART 1'''
# import matplotlib.pyplot as plt

# # Pie chart, where the slices will be ordered and plotted counter-clockwise:
# labels = 'Decluttering', 'Dusting', 'Vacuuming', 'Mopping', 'Bathroom Cleaning', 'Kitchen Cleaning'
# sizes = [15, 20, 20, 15, 15, 15]
# explode = (0.1, 0, 0, 0, 0, 0)  # only "explode" the 1st slice (i.e. 'Decluttering')

# fig1, ax1 = plt.subplots()
# ax1.pie(sizes, 
#         explode = explode, 
#         labels = labels, 
#         autopct = '%1.1f%%',
#         shadow = True, 
#         startangle = 90)
# ax1.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.

# plt.title('Effective House Cleaning Distribution')
# plt.show()


'''CHART 2'''
# import altair as alt
# import pandas as pd

# # Define the cleaning tasks and their approximate percentages
# tasks = [
#     "Decluttering",
#     "Dusting",
#     "Vacuuming & Sweeping",
#     "Mopping",
#     "Bathroom Cleaning",
#     "Kitchen Cleaning",
#     "Laundry",
#     "Windows Cleaning"
# ]

# # Approximate time investment (as percentages)
# percentages = [10, 8, 20, 15, 10, 15, 10, 12]

# # Create a DataFrame
# df = pd.DataFrame({
#     'Task': tasks,
#     'Percentage': percentages
# })

# # Create the Altair pie chart
# pie_chart = alt.Chart(df).mark_arc().encode(
#     theta = alt.Theta(field='Percentage', type='quantitative'),
#     color = alt.Color(field='Task', type='nominal'),
#     tooltip = ['Task', 'Percentage']
# ).properties(
#     title = 'Effective House Cleaning Distribution'
# )

# pie_chart.show()



'''COMPARE THIS ONE TO THE ONE ABOVE'''
# import pandas as pd
# import altair as alt

# # Create a dictionary with the cleaning tasks and their estimated times
# data = {
#     'Task': ['Declutter (All Rooms)', 'Clean High Surfaces, Walls, Baseboards, Windows, and Mirrors', 'Bedrooms', 'Bathrooms', 'Kitchen', 'Living Room', 'Empty Trash and Recycling'],
#     'Estimated Time (minutes)': [45, 40, 20, 20, 25, 15, 5]
# }

# # Create a DataFrame from the dictionary
# df = pd.DataFrame(data)

# # Calculate the percentage of each value in relation to the total sum of the `Estimated Time (minutes)` column, store it as a number between 0 and 1.
# df["percentage"] = df["Estimated Time (minutes)"] / df["Estimated Time (minutes)"].sum()

# # Create the base chart with the theta encoding for the pie slices
# base = alt.Chart(df).encode(
#     theta=alt.Theta("Estimated Time (minutes):Q", stack=True),
#     color=alt.Color("Task:N", legend=None),
#     tooltip=["Task", "Estimated Time (minutes)", alt.Tooltip("percentage", title="Percentage", format=".1%")]
# )

# # Create the pie chart with the arcs
# pie = base.mark_arc(outerRadius=120)

# # Create the text labels for the pie slices
# text = base.mark_text(radius=140, fill="white").encode(
#     text=alt.Text("Estimated Time (minutes):Q", format=".0f"),
# )

# # Combine the pie chart and text labels
# chart = pie + text

# # Configure the chart title and properties
# chart = chart.properties(
#     title="How to Effectively Clean Your House (Estimated Time per Task)"
# ).configure_title(
#     fontSize=14,
#     anchor="middle",
# ).interactive()

# # Save the chart in a JSON file
# chart.save('cleaning_tasks_pie_chart.html')


















'''
    HERE WE MAKE A REALLY COOL BAR CHART WITH THE NEW MODULE ALTAIR! THIS IS OUR FIRST TASTE OF EXPLORING WITH INTERAVTIVE BROWSER-BASED GRAPHS THAT HAVE TOOL TIPS!
'''

# import pandas as pd
# import altair as alt

# # Make a DataFrame with the columns `Variety` and `Calories per Serving (g)` using the values from the table above
# df = pd.DataFrame(
#     {
#         "Variety": [
#             "Regular Potato Chips",
#             "Baked Potato Chips",
#             "Kettle-Cooked Potato Chips",
#             "Reduced-Fat Potato Chips",
#             "Tortilla Chips",
#             "Multigrain Chips",
#         ],
#         "Calories per Serving (g)": [160, 130, 150, 140, 150, 140],
#     }
# )

# # Make bar plot with `Variety` on the x-axis and `Calories per Serving (g)` on the y-axis
# chart = (
#     alt.Chart(df, title="Calories per Serving of Different Potato Chips")
#     .mark_bar()
#     .encode(
#         x=alt.X("Variety:N", axis=alt.Axis(labelAngle=-45)),
#         y="Calories per Serving (g):Q",
#         tooltip=["Variety", "Calories per Serving (g)"],
#     )
#     .interactive()
# )

# # Save the plot
# chart.save("calories_per_serving_potato_chips_bar_chart.html")






















'''
    HERE WE DEMO THE LINESTYLE ARG
'''
# import matplotlib.pyplot as plt

# # Define some x and y data
# x = range(10)
# y1 = [i for i in x]
# y2 = [i * 2 for i in x]
# y3 = [i ** 2 for i in x]

# # Plot data with different linestyles
# plt.plot(x, y1, linestyle='-', label='Solid')
# plt.plot(x, y2, linestyle='--', label='Dashed')
# plt.plot(x, y3, linestyle=':', label='Dotted')

# plt.legend()
# plt.xlabel('X Axis')
# plt.ylabel('Y Axis')
# plt.title('Different Linestyles')
# plt.grid(True, linestyle='-.', alpha=0.7)
# plt.show()





'''
    HERE WE DEMO THE ANNOTATE METHOD
'''
# import matplotlib.pyplot as plt
# import pandas as pd

# # Define the data
# data = {
#     'body_area': ['Chest', 'Back', 'Triceps', 'Biceps', 'Shoulders', 'Abs', 'Legs', 'Legs', 'Calves', 'Glutes'],
#     'target_muscle': ['Pectoralis Major', 'Latissimus Dorsi', 'Triceps Brachii', 'Biceps Brachii', 'Deltoids',
#                       'Rectus Abdominis', 'Quadriceps', 'Hamstrings', 'Gastrocnemius', 'Gluteus Maximus'],
#     'exercise': ['Bench Press', 'Pull-Ups', 'Triceps Pushdowns', 'Bicep Curls', 'Overhead Press', 'Crunches',
#                  'Squats', 'Deadlifts', 'Calf Raises', 'Hip Thrusts'],
#     'workout_length': [20, 35, 30, 40, 30, 25, 35, 45, 20, 30],
#     'reps': [10, 10, 10, 10, 10, 20, 10, 10, 20, 10]
# }

# # Create a DataFrame
# df = pd.DataFrame(data)

# # Create the scatter plot
# plt.figure(figsize=(8, 6))
# plt.scatter(df['workout_length'], df['reps'], color='blue')
# plt.xlabel('Workout Length (minutes)')
# plt.ylabel('Reps')
# plt.title('Scatter Plot of Workout Length vs. Reps')
# plt.grid(True, linestyle='--', alpha=0.7)

# # Add annotations for each exercise
# for i, txt in enumerate(df['exercise']):
#     plt.annotate(txt, (df['workout_length'][i], df['reps'][i]), fontsize=8, ha='right')

# plt.show()



'''
    HERE WE DEMO THE ANNOTATE METHOD'S HORIZONTAL ALIGNMENT ARG (ha)
'''
# import matplotlib.pyplot as plt

# # Example data
# x = [1, 2, 3, 4, 5]
# y = [10, 15, 20, 25, 30]
# labels = ['A', 'B', 'C', 'D', 'E']

# # Plot the data
# plt.scatter(x, y)

# # Add annotations with different horizontal alignments
# for i, txt in enumerate(labels):
#     plt.annotate(txt, (x[i], y[i]), ha='left')  # Align to the left
#     plt.annotate(txt, (x[i] + 0.2, y[i]), ha='right')  # Align to the right
#     plt.annotate(txt, (x[i] + 0.4, y[i]), ha='center')  # Centered

# plt.grid(True)
# plt.xlabel('X Axis')
# plt.ylabel('Y Axis')
# plt.title('Horizontal Alignment (ha) Parameter Example')
# plt.show()



'''
This will create a legend with the markers corresponding to each exercise, providing meaningful labels. However, if all the data points use the same color or marker, you may not see all distinct labels.
'''
# import pandas as pd
# import matplotlib.pyplot as plt

# # Data
# data = {
#     "body_area": ["Chest", "Back", "Triceps", "Biceps", "Shoulders", "Abs", "Legs", "Legs", "Calves", "Glutes"],
#     "target_muscle": ["Pectoralis Major", "Latissimus Dorsi", "Triceps Brachii", "Biceps Brachii", "Deltoids", "Rectus Abdominis", "Quadriceps", "Hamstrings", "Gastrocnemius", "Gluteus Maximus"],
#     "exercise": ["Bench Press", "Pull-Ups", "Triceps Pushdowns", "Bicep Curls", "Overhead Press", "Crunches", "Squats", "Deadlifts", "Calf Raises", "Hip Thrusts"],
#     "workout_length": [30, 30, 30, 30, 30, 30, 30, 30, 30, 30],
#     "reps": [10, 10, 10, 10, 10, 20, 10, 10, 20, 10]
# }

# df = pd.DataFrame(data)

# # Plot
# plt.figure(figsize=(10, 6))
# scatter = plt.scatter(df["workout_length"], df["reps"], c='blue', label='Exercises')
# plt.title('Workout Length vs. Repetitions')
# plt.xlabel('Workout Length (minutes)')
# plt.ylabel('Repetitions')
'''HERE WE TRY TO ASSIGN THE LABELS REFERENCED ABOVE'''
# plt.legend(handles=scatter.legend_elements()[0], labels=df["exercise"])
# plt.grid(True)
# plt.show()

'''THIS WAY MIGHT BE BETTER'''
# # Create a scatter plot
# plt.figure(figsize=(10, 6))
# for i, exercise in enumerate(df['exercise']):
#     plt.scatter(df['workout_length'][i], df['reps'][i], label=exercise)

# plt.title('Workout Length vs. Repetitions')
# plt.xlabel('Workout Length (minutes)')
# plt.ylabel('Repetitions')
# plt.legend()
# plt.grid(True)
# plt.show()



















'''
    HERE, WE ILLUSTRATE THE FOLLOWING:

    1. df['column_name']: Selects a single column as a pandas Series.
    2. df[['column1', 'column2']]: Selects multiple columns as a DataFrame.
    3. Two different types of heatmaps
'''

# import pandas as pd
# import seaborn as sns
# import matplotlib.pyplot as plt

# # Define the data
# data = {
#     "Body Region": ["Chest", "Back", "Legs", "Shoulders", "Arms"],
#     "Muscle Group": ["Chest", "Back", "Quadriceps", "Shoulders", "Biceps"],
#     "Exercise": ["Bench Press", "Pull-Ups", "Squats", "Overhead Press", "Bicep Curls"],
#     "Time (in minutes)": [30, 20, 45, 25, 15],
#     "Reps": [10, 15, 20, 10, 15]
# }

# # Create a DataFrame
# df = pd.DataFrame(data)

# # Compute the correlation matrix regarding this subset
# correlation_matrix = df[["Time (in minutes)", "Reps"]].corr()

# # Plot the heatmap V1
# plt.figure(figsize=(8, 6))
# sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', vmin=-1, vmax=1)
# plt.title("Correlation Heatmap: Time vs. Reps")
# plt.show()

# # Plot the heatmap V2
# plt.figure(figsize=(8, 6))
# heatmap = sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f')
# heatmap.set_title('Correlation Heatmap between Time and Reps')
# plt.show()








'''
    HEATMAP WITH ALTAIR
'''
# import pandas as pd
# import altair as alt

# # Create a DataFrame with the data
# data = {
#     'Body Region': ['Chest', 'Back', 'Legs', 'Shoulders', 'Arms'],
#     'Muscle Group': ['Chest', 'Back', 'Quadriceps', 'Shoulders', 'Biceps'],
#     'Exercise': ['Bench Press', 'Pull-Ups', 'Squats', 'Overhead Press', 'Bicep Curls'],
#     'Time (in minutes)': [30, 20, 45, 25, 15],
#     'Reps': [10, 15, 20, 10, 15]
# }

# df = pd.DataFrame(data)

# # Calculate the correlation between `Time (in minutes)` and `Reps`
# correlation = df['Time (in minutes)'].corr(df['Reps'])

# # Create a DataFrame for the heatmap
# correlation_df = pd.DataFrame({'Correlation': [correlation]})

# # Create the heatmap
# heatmap = alt.Chart(correlation_df).mark_rect().encode(
#     x=alt.X('Correlation:Q', axis=alt.Axis(title='')),
#     y=alt.Y('Correlation:Q', axis=alt.Axis(title='')),
#     color=alt.Color('Correlation:Q', scale=alt.Scale(scheme='blueorange')),
#     tooltip=['Correlation:Q']
# ).properties(
#     title='Correlation Between Time and Reps'
# ).interactive()

# # Add text label to show the correlation value
# text = heatmap.mark_text(baseline='middle').encode(
#     text=alt.Text('Correlation:Q', format='.2f')
# )

# # Combine the heatmap and text label
# chart = heatmap + text

# # Display the chart
# chart.save('correlation_heatmap_time_reps.html')
























'''
    HERE WE CLEAN UP A FILTHY DATASET AND THEN CALCULATE THE DIFFERENT PRODUCTS PRICE DELTA AS PERCENTAGES.
'''

# import pandas as pd
# import seaborn as sns
# import matplotlib.pyplot as plt

# # Load the data from the TSV file
# strawberry_sales_path = './CSVs/STRAWBERRY SALES 2023 - Sheet1.tsv'
# strawberry_sales_data = pd.read_csv(strawberry_sales_path, sep='\t', skiprows=2)

# # Display the first few rows and recheck the data summary
# print(strawberry_sales_data.head()) 
# print(strawberry_sales_data.info(verbose=True)) 
# print(strawberry_sales_data.describe())

# # Rename the poorly formatted col names
# strawberry_sales_data.columns = ['DATE', 'CLAMSHELLS', 'NUM_BOXES', 'KILOS', 'PRICE_PER_BOX', 'TOTAL', 'PRODUCT', 'TYPE_OF_PRODUCT']

# # Convert price fields from string to float and clean them
# strawberry_sales_data['PRICE_PER_BOX'] = strawberry_sales_data['PRICE_PER_BOX'].str.replace('[\$,]', '', regex=True).str.strip().astype(float)
# strawberry_sales_data['TOTAL'] = strawberry_sales_data['TOTAL'].replace('[\$,]', '', regex=True).astype(float)

# # Convert the dates to datetime
# strawberry_sales_data['DATE'] = pd.to_datetime(strawberry_sales_data['DATE'], format='mixed')

# # Confirm the clean-up
# print("\n\nAFTER:")
# print(strawberry_sales_data.head()) 
# print(strawberry_sales_data.info(verbose=True)) 
# print(strawberry_sales_data.describe())

# # Calculate the average price per product type
# strawberry_sales_data['MONTH'] = pd.to_datetime(strawberry_sales_data['DATE']).dt.month
# average_price_per_type = strawberry_sales_data.groupby('TYPE_OF_PRODUCT')['PRICE_PER_BOX'].mean()
# print("\nAverage price per strawberry type:")
# print(average_price_per_type)

# # Calculate the month-by-month price change for each product type
# monthly_price_change = strawberry_sales_data.groupby(['TYPE_OF_PRODUCT', 'MONTH'])['PRICE_PER_BOX'].mean().groupby(level=0).pct_change()
# print("\nMonthly price percentage delta:")
# print(monthly_price_change)

# # Identify the month with the highest price for each product type
# highest_price_month = strawberry_sales_data.groupby(['TYPE_OF_PRODUCT', 'MONTH'])['PRICE_PER_BOX'].mean().groupby(level=0).idxmax()
# print("\nThe month with the highest price:")
# print(highest_price_month)





















'''
    COOL NEW MATPLOTLIB MODULES!
'''

# import pandas as pd
# import matplotlib.pyplot as plt
# import seaborn as sns
# import matplotlib.dates as mdates
# import matplotlib.colors as mcolors

# # Load the data
# file_path = './CSVs/Indian Summers.csv'
# data = pd.read_csv(file_path)

# print(data.head())
# print(data.info())

# # Convert Date to datetime type
# data['Date'] = pd.to_datetime(data['Date'])  
# data['Year'] = data['Date'].dt.year
# data['Month'] = data['Date'].dt.month

# # Define a custom color map from white to dark greyish blue
# cmap = mcolors.LinearSegmentedColormap.from_list('custom_blue', ['white', '#2c3e50'])

# # Setting up the plot
# plt.figure(figsize=(20, 5), facecolor='white')
# plt.scatter(data['Date'], 
#             data['moonphase'], 
#             c=data['moonphase'], 
#             cmap='Blues', 
#             edgecolor='none')
# plt.colorbar(label='Moonphase')
# plt.gca().xaxis.set_major_locator(mdates.YearLocator())
# plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
# plt.title('Moonphase Over Time')
# plt.xlabel('Date')
# plt.ylabel('Moonphase')
# plt.grid(True)
# plt.show()




















'''
    HERE WE PERFORM A K-CLUSTERS ANALYSIS
'''
# from sklearn.cluster import KMeans
# import seaborn as sns
# import matplotlib.pyplot as plt
# import pandas as pd

# # Load the TSV file into a DataFrame
# wellness_costs = pd.read_csv('./CSVs/WELLNESS_COST_2022_CW_V2  - Sheet1.tsv', sep='\t')

# # Explore the dataset
# print(wellness_costs.head())
# print(wellness_costs.info())

# # Adding a new column 'YEARLY' by summing the monthly columns
# wellness_costs['YEARLY'] = wellness_costs[['JANUARY', 'FEBRUARY', 'MARCH', 'APRIL', 'MAY', 'JUNE', 'JULY', 'AUGUST', 'SEPTEMBER', 'OCTOBER', 'NOVEMBER', 'DECEMBER']].sum(axis=1)

# # Display the head of the dataframe to confirm the new column
# print(wellness_costs.head())

# # Extract the YEARLY column for clustering
# X = wellness_costs[['YEARLY']].values

# # Determine the optimal number of clusters using the elbow method
# sse = []
# k_values = range(1, 11)

# for k in k_values:
#     kmeans = KMeans(n_clusters=k, random_state=0)
#     kmeans.fit(X)
#     sse.append(kmeans.inertia_)

# # Plot the elbow graph
# plt.figure(figsize=(8, 5))
# plt.plot(k_values, sse, marker='o')
# plt.xlabel('Number of Clusters')
# plt.ylabel('Sum of Squared Errors')
# plt.title('Elbow Method to Determine Optimal Number of Clusters')

# # Apply k-means clustering with 3 clusters
# kmeans_3 = KMeans(n_clusters=3, random_state=0)
# wellness_costs['Cluster_3'] = kmeans_3.fit_predict(X)

# # Analyze the clusters by showing basic statistics
# cluster_analysis_3 = wellness_costs.groupby('Cluster_3')['YEARLY'].describe()

# print("\nCluster analysis:")
# print(cluster_analysis_3)

# # Visualize the clusters
# plt.figure(figsize=(10, 6))
# sns.histplot(data=wellness_costs, x='YEARLY', hue='Cluster_3', multiple='stack', bins=30, palette='Set2')
# plt.xlabel('Yearly Expenditure')
# plt.ylabel('Count')
# plt.title('Distribution of Yearly Expenditure by Cluster (3 Clusters)')
# plt.show()


































'''
    2 DIFF HEATMAP APPROACHES
'''

# import altair as alt
# import numpy as np
# import pandas as pd
# from pandas.api.types import is_numeric_dtype

# # Read the CSV file into a DataFrame
# df = pd.read_excel('./CSVs/PAYROLL_MAY.xlsx')

# # Display the first 5 rows
# print(df.head())

# # Print the column names and their data types
# print(df.info(verbose=True))

# uni_years = df['YEAR'].value_counts()
# print(f'Unique Years:\n{uni_years}')

# for column_name in ['YEAR']:
#   if not is_numeric_dtype(df[column_name]):
#     # Assume CSV columns can only be numeric or string.
#     df[column_name] = pd.to_numeric(
#         df[column_name].str.replace(',', 
#                                     repl='', 
#                                     regex=True),
#                                     ).fillna(0)

# print(df['YEAR'].value_counts())
# print(df['YEAR'].describe(percentiles=[.1, .25, .5, .75, .9]))

# # Combine `YEAR` and `MONTH` to create datetime column `Date`
# df['Date'] = pd.to_datetime(df[['YEAR', 'MONTH']].assign(DAY=1))

# # Print the value counts of `Date`
# print(df['Date'].value_counts())

# # Print descriptive statistics of `Date`
# print(df['Date'].describe())

# # Combine the month and year into a single datetime column
# df['Date'] = pd.to_datetime(df.YEAR.astype(str) + '-' + df.MONTH.astype(str), format='%Y-%m')

# # Show the first few rows to confirm the new column
# print(df[['MONTH', 'YEAR', 'Date']].head())

# # Create a pivot table with the years extracted from the `Date` column as the index, 
# # the months extracted from the `Date` column as the columns, and the sum of `PLA_I_BASIC ASSETS` as the values.
# pivot_table = df.pivot_table(index=df['Date'].dt.year, 
#                              columns=df['Date'].dt.month, 
#                              values='PLA_I_BASIC ASSETS', 
#                              aggfunc='sum')

# # Replace missing values with 0
# pivot_table = pivot_table.fillna(0)

# # Melt the pivot table to long format for Altair
# melted_df = pivot_table.reset_index().melt(id_vars='Date', var_name='Month', value_name='PLA_I_BASIC ASSETS')

# # Convert the 'Date' and 'Month' columns to strings for Altair
# melted_df['Date'] = melted_df['Date'].astype(str)
# melted_df['Month'] = melted_df['Month'].astype(str)

# # Create the heatmap
# chart = alt.Chart(melted_df).mark_rect().encode(
#     x=alt.X('Month:O', axis=alt.Axis(title='Month', labelAngle=-45)),
#     y=alt.Y('Date:O', axis=alt.Axis(title='Year')),
#     color=alt.Color('PLA_I_BASIC ASSETS:Q', scale=alt.Scale(scheme='blues')),
#     tooltip=['Month', 'Date', 'PLA_I_BASIC ASSETS']
# ).properties(
#     title='Heatmap of PLA_I_BASIC ASSETS by Year and Month'
# ).interactive()

# # Add text labels to the heatmap cells
# text = chart.mark_text(baseline='middle').encode(
#     text=alt.Text('PLA_I_BASIC ASSETS:Q', format='.0f')
# )

# # Combine the heatmap and text layers
# final_chart = chart + text

# # Save the chart as a JSON file
# final_chart.save('basic_assets_heatmap2.html')






'''SREAMLINED AND USING MEAN INSTEAD OF SUM'''

# import altair as alt
# import pandas as pd

# # Load the dataset
# payroll_data = pd.read_excel('PAYROLL_MAY.xlsx')

# # Creating a pivot table for the heatmap
# heatmap_data = payroll_data.pivot_table(index='YEAR', 
#                                         columns='MONTH', 
#                                         values='PLA_I_BASIC ASSETS', aggfunc='mean').reset_index()

# heatmap_data = pd.melt(heatmap_data, id_vars=['YEAR'], value_vars=heatmap_data.columns[1:], var_name='MONTH', value_name='PLA_I_BASIC ASSETS')

# # Creating the heatmap using Altair
# heatmap = alt.Chart(heatmap_data).mark_rect().encode(
#     x='MONTH:O',
#     y='YEAR:O',
#     color='PLA_I_BASIC ASSETS:Q',
#     tooltip=['YEAR', 'MONTH', 'PLA_I_BASIC ASSETS']
# ).properties(
#     title='Heatmap of PLA_I_BASIC ASSETS by Month and Year',
#     width=500,
#     height=300
# )

# heatmap.save("payrollHeatmap.html")


















'''
    FIND ALL NON ZERO COLS WITHIN AN EXCEL FILE
'''

# import pandas as pd

# FILEPATH = './CSVs/outcomes_incomes_fs.xlsx'
# df = pd.read_excel(FILEPATH, header=1)

# print(df.info(verbose=True))

# # Rename columns
# columns = ['Type', 'Description', 'January', 'February', 'March', 'April', 'May', 'June',
#            'July', 'August', 'September', 'October', 'November']

# df.columns = columns

# # Filter only income-related data and exclude total rows
# income_data = df[df['Type'].str.lower() == 'incomes']

# # Remove NaN columns that don't have month data
# income_data = income_data.drop(columns=['Type'])

# # Convert all month columns to numeric values and replace non-numeric data with NaN
# months = columns[2:]
# income_data[months] = income_data[months].apply(pd.to_numeric, errors='coerce')

# # Drop NaN rows
# income_data_clean = income_data.dropna(how='all', subset=months)

# # Display non-zero incomes per month
# non_zero_incomes = {}
# for month in months:
#     non_zero_incomes[month] = income_data_clean[income_data_clean[month] != 0][['Description', month]].reset_index(drop=True)

# print(f"\nAll non-zero incomes for each month:\n{non_zero_incomes}")




















'''
    USE THIS WHEN IT WONT or CANT FIND A FILE IN CSVs DIRECTORY... IT MIGHT BE HIDDEN OR HAVE A DIFF NAME BEHIND THE SCENES!
'''

# # Print current working directory
# import os
# print("Current Working Directory:", os.getcwd())

# # List files in the specific directory
# print("Files in './CSVs/':", os.listdir('./CSVs/'))





'''
    EXTRACT COLUMNS AND SPECIFIC ENTRIES FROM THOSE COLUMNS.
'''

# import pandas as pd

# # Read the CSV file into a DataFrame
# df = pd.read_excel('./CSVs/Data_Set_TransactionReport_All_01Nov2023_30Nov2023_20231214165632.xlsx')
# print(df.head())
# print(df.info())

# # Correct the column name for 'Status' (PaymentStatus)
# columns_of_interest = [
#     'Transaction Date', 'RecordLocator', 'PaymentStatus', 'PaymentAmount'
# ]
# df_filtered = df[columns_of_interest]

# # Redefine the date range
# start_date = pd.Timestamp('2023-09-01')
# end_date = pd.Timestamp('2023-12-31')

# # Extract the payment date and time only
# payment_dates = df_filtered[
#     (df_filtered['Transaction Date'] >= start_date) &
#     (df_filtered['Transaction Date'] <= end_date) &
#     (df_filtered['PaymentStatus'].str.lower() == 'approved') &
#     (df_filtered['RecordLocator'].str.contains(r'\d'))
# ]['Transaction Date'].dt.strftime('%y-%m-%d %H:%M')

# payment_dates_list = payment_dates.tolist()

# # Display the results
# print(f'\nPayment dates:\n{payment_dates_list}')

























'''
    BARPLOT WITH LINE THAT FITS
'''

# import matplotlib.pyplot as plt
# import seaborn as sns
# import pandas as pd

# # Load the dataset
# file_path = './CSVs/Real Estate Mumbai Database - Rgdcvvvh.csv'

# # Attempt to read the dataset with different encoding
# df = pd.read_csv(file_path, encoding='ISO-8859-1')

# # Display the first few rows and check the column names
# print(df.head())
# print(df.columns)
# print(df.info())

# avg_client_age = df['CLIENT AGE'].mean()
# print(f'\nAvg. client age is {avg_client_age}')
# most_freq_addy = df['PROPERTY ADDRESS'].value_counts()
# print(f'\nCOUNT OF {most_freq_addy}')
# num_bedrooms = df['NUMBER OF BEDROOMS'].value_counts()
# print(f'\n{num_bedrooms}')
# types_of_transactions = df['TRANSACTION TYPE'].value_counts()
# print(f'\n{types_of_transactions}')
# dates_of_transactions = df['TRANSACTION DATE'].value_counts()
# print(f'\n{dates_of_transactions}')
# desc_spending_max = df['AMOUNT IN (INR)'].max()
# desc_spending_min = df['AMOUNT IN (INR)'].min()
# print(f'\nMax spent is {desc_spending_max} and the min spent is {desc_spending_min}')
# min_age = df['CLIENT AGE'].min()
# max_age = df['CLIENT AGE'].max()
# print(f'\nMax age is {max_age} and the min age is {min_age}')




# # Filter based on transaction types
# rent_transactions = df[df['TRANSACTION TYPE'] == 'RENT']
# buy_transactions = df[df['TRANSACTION TYPE'] == 'BUY']

# # Find the minimum amount for rental transactions
# cheapest_rental = rent_transactions['AMOUNT IN (INR)'].min()
# max_rental = rent_transactions['AMOUNT IN (INR)'].max()

# # Find the maximum amount for purchase transactions
# max_purchase = buy_transactions['AMOUNT IN (INR)'].max()
# min_purchase = buy_transactions['AMOUNT IN (INR)'].min()
# print(f'\nExpensive rental is {max_rental}')
# print(f'\nCheapest rental cost {cheapest_rental}')
# print(f'\nExpensive purchase is {max_purchase}')
# print(f'\nCheapest purchase cost {min_purchase}')



# # Client Age Analysis
# plt.figure(figsize=(10, 5))
# sns.histplot(df['CLIENT AGE'], bins=15, kde=True, color='skyblue')
# plt.title('Distribution of Client Age')
# plt.xlabel('Client Age')
# plt.ylabel('Frequency')
# plt.grid(True)

# # Categorize into age groups
# bins = [0, 18, 25, 35, 45, 60, 100]
# labels = ['0-18', '18-25', '25-35', '35-45', '45-60', '60+']
# df['AGE GROUP'] = pd.cut(df['CLIENT AGE'], bins=bins, labels=labels, right=False)

# # Count by Age Group
# age_group_count = df['AGE GROUP'].value_counts().sort_index()
# plt.figure(figsize=(10, 5))
# sns.barplot(x=age_group_count.index, 
#             y=age_group_count.values, 
#             palette='pastel', 
#             hue=age_group_count, 
#             legend=False)
# plt.title('Transaction Count by Age Group')
# plt.xlabel('Age Group')
# plt.ylabel('Transaction Count')
# plt.grid(True)


# # Property Address Analysis
# property_count = df['PROPERTY ADDRESS'].value_counts().head(10)
# plt.figure(figsize=(10, 5))
# sns.barplot(x=property_count.index, 
#             y=property_count.values, 
#             palette='pastel',
#             hue=property_count,
#             legend=False)
# plt.title('Top 10 Property Addresses by Transaction Count')
# plt.xlabel('Property Address')
# plt.ylabel('Transaction Count')
# plt.grid(True)


# # Correlation between Age Group and Property Address
# address_age_group = df.groupby(['PROPERTY ADDRESS', 'AGE GROUP'], observed=False).size().unstack(fill_value=0)

# plt.figure(figsize=(12, 8))
# sns.heatmap(address_age_group, annot=True, fmt='d', cmap='Blues', linewidths=.5)
# plt.title('Correlation Between Property Address and Age Group')
# plt.xlabel('Age Group')
# plt.ylabel('Property Address')
# plt.show()


























'''
    HERE WE RESOLVE THE SettingWithCopyWarning WARNING BY USING THE .copy() METHOD
'''


# import pandas as pd

# # Load the dataset with proper encoding
# laptop_df = pd.read_csv('./CSVs/Cleaned_Laptop_data.csv', encoding='ISO-8859-1')
# print(laptop_df.head())
# print(laptop_df.columns)
# print(laptop_df.info())

# # Filter laptops with a 'star_rating' of 4 or higher
# high_rating_laptops = laptop_df[laptop_df['star_rating'] >= 4].copy()

# # Calculate the mean price difference
# high_rating_laptops['price_difference'] = high_rating_laptops['old_price'] - high_rating_laptops['latest_price']
# mean_price_difference = high_rating_laptops['price_difference'].mean()

# print('Mean Price Difference:', mean_price_difference)






















'''
    GET THE TOP TEN
'''

# import pandas as pd

# # Load the dataset with ISO-8859-1 encoding
# ice_email_df = pd.read_csv('./CSVs/ice_email_replies-ice_email_replies.csv', encoding='ISO-8859-1')

# # Check the columns and head
# print(ice_email_df.head())
# print(ice_email_df.columns)
# print(ice_email_df.info())

# # Calculate the top 10 most frequent emails
# top_emails = ice_email_df['customerEmailSentFrom'].value_counts().head(10)

# # Print the top 10 emails and their frequency
# print(top_emails)





































'''
    SAVE A FILE TO OUTCSVS
'''

# import pandas as pd

# # Load the data from the Excel file
# audit_df = pd.read_excel('./CSVs/Premium Collection Audit.xlsx')
# print(audit_df.head())
# print(audit_df.columns)
# print(audit_df.info())

# # Combine the Audit ID and Policy ID Number into a new column
# audit_df['Audit/Policy ID'] = audit_df['Audit ID'].astype(str) + '/.' + audit_df['Policy ID Number'].astype(str)

# # Save the new table
# audit_df.to_csv('./OutCSVs/Audit_Policy.csv', index=False)
# print('\nSUCCESS!\nAudit_Policy.csv has been saved with the new Audit/Policy ID column.')



















































'''
    HERE, WE PERFORM A SOPHISTICATED CLEAN UP ON A MULTI-DIMENSIONAL EXCEL FILE
'''


# import seaborn as sns
# import matplotlib.pyplot as plt
# import pandas as pd

# # Read the Excel file into a DataFrame
# FILEPATH = './CSVs/outcomes_incomes_fs.xlsx'
# df = pd.read_excel(FILEPATH, header=1)

# # Rename the first two columns
# df.rename(columns={df.columns[0]: 'Categories', 
#                    df.columns[1]: 'Subcategories'}, 
#                    inplace=True)

# # Separate the dataset into two dataframes for generating two heatmaps
# # Group by Categories and drop the subcategories 
# category_group = df.groupby('Categories').sum()
# category_group = category_group.drop(category_group.columns[0], axis=1)

# # Generate the heatmap for 'Categories' vs months
# plt.figure(figsize=(12, 8))
# sns.heatmap(category_group, 
#             annot=True, 
#             cmap='coolwarm', 
#             linewidths=0.5)
# plt.title('Categories vs. Months Heatmap')
# plt.xlabel('Months')
# plt.ylabel('Categories')

# # Group by Subcategories and drop the categories
# subcategory_group = df.groupby('Subcategories').sum()
# subcategory_group = subcategory_group.drop(subcategory_group.columns[0], axis=1)

# # Generate the heatmap for 'Subcategories' vs months
# plt.figure(figsize=(20, 30))
# sns.heatmap(subcategory_group, 
#             annot=True, 
#             cmap='coolwarm', 
#             linewidths=0.5)
# plt.title('Subcategories vs. Months Heatmap')
# plt.xlabel('Months')
# plt.ylabel('Subcategories')
# plt.show()







































































# import seaborn as sns
# import matplotlib.pyplot as plt

# # Cleaning and restructuring the data for visualization
# # Extracting month headers
# months = df.iloc[0, 2:].values

# # Dropping unnecessary rows and columns for clarity
# cleaned_data = df.drop([0]).reset_index(drop=True)

# # Setting the first row as column names and dropping the first column
# cleaned_data.columns = ['Category', 'Source'] + list(months)
# cleaned_data = cleaned_data.drop(columns=['Category'])

# # Melting the dataframe to have a long format for easier plotting
# melted_data = cleaned_data.melt(id_vars='Source', var_name='Month', value_name='Amount')

# # Converting 'Amount' to numeric, errors='coerce' will convert non-convertible values to NaNs
# melted_data['Amount'] = pd.to_numeric(melted_data['Amount'])

# # Dropping NaN values if any
# melted_data.dropna(inplace=True)

# # Pivot table for heatmap
# pivot_table = melted_data.pivot("Source", "Month", "Amount")

# # Generating the heatmap
# plt.figure(figsize=(12, 8))
# sns.heatmap(pivot_table, annot=True, fmt=".0f", cmap="YlGnBu", linewidths=.5)
# plt.title('Monthly Income and Outgoings for Each Category (2022)')
# plt.xticks(rotation=45)
# plt.ylabel('Source/Category')
# plt.xlabel('Month')
# plt.tight_layout()

# # Display the heatmap
# plt.show()





























'''
    THIS IS THE MOST STRAIGHTFORWARD WAY
'''

# import pandas as pd
# import matplotlib.pyplot as plt

# # Step 1: Load the CSV file into a DataFrame
# file_path = './CSVs/Bancos-Bodega.xlsxcsv-Global.csv'
# data = pd.read_csv(file_path)

# # First, ensure we handle negative values correctly by trimming spaces
# data['Monto'] = data['Monto'].str.replace('$', '').str.replace(',', '').str.replace(' ', '').astype(float)

# # Now, let's try calculating the average value of 'Monto' again
# average_monto = data['Monto'].mean()

# # Display the result
# print(f'The average value of Monto is: {average_monto}')
























'''
    SPLITTING STRINGS AND JOINING STRINGS
'''


# import pandas as pd

# # Load the top_200_youtubers.csv data
# youtubers_df = pd.read_csv('./CSVs/top_200_youtubers.csv')
# uni_more_topics = youtubers_df['More topics'].value_counts()
# print(f'\nUnique {uni_more_topics}')

# # Split the 'More topics' column's comma-separated values into three new columns
# youtubers_df[['topic one', 'topic two', 'topic three']] = youtubers_df['More topics'].str.split(',', expand=True, n=2)

# # Print the head of the dataframe to verify the new columns
# print(youtubers_df[['More topics', 'topic one', 'topic two', 'topic three']].head(n=20))

# # Consolidate 'topic one', 'topic two', 'topic three' into one column named 'topic'
# youtubers_df['topic'] = youtubers_df[['topic one', 'topic two', 'topic three']].apply(lambda x: ', '.join(x.dropna()), axis=1)

# # Print the head of the dataframe to verify the new 'topic' column
# print(youtubers_df[['topic one', 'topic two', 'topic three', 'topic']].head(n=20))






















# import matplotlib.pyplot as plt
# import pandas as pd 

# # Load the hoa_transactions.csv data
# hoa_df = pd.read_csv('./CSVs/hoa_transactions.csv')

# print(hoa_df.head())
# print(hoa_df.columns)
# print(hoa_df.info())

# # Inspect its format
# print("\n")
# print(hoa_df['Date'].head(n=20))


# '''converting dates with different formats'''
# # Function to parse dates with multiple formats (for this specific dataset)
# def parse_date(date_str):
#     if not isinstance(date_str, float) and len(date_str) > 2:
#         formats = ['%d-%b', '%b-%d']
#         for fmt in formats:
#             try:
#                 return pd.to_datetime(date_str, format=fmt)
#             except ValueError:
#                 pass
#     return None

# # Apply function to the date column for conversions
# hoa_df['Date'] = hoa_df['Date'].apply(parse_date)

# # Reinspect the format conversion
# print("\n")
# print(hoa_df['Date'].head(n=20))
# print(hoa_df.info())

# # Create a scatter plot between `Date` and `Income`.
# plt.figure(figsize=(10, 6))
# plt.scatter(hoa_df['Date'], hoa_df['Income'])

# # Calculate correlations after converting Date to timestamp
# corr = hoa_df['Date'].astype('int64').corr(hoa_df['Expenses'])
# print(f'Correlation coefficient between Date (as timestamp) and Expenses: {corr}')

# # Calculate correlations after converting Date to timestamp
# corr = hoa_df['Date'].astype('int64').corr(hoa_df['Income'])
# print(f'Correlation coefficient between Date (as timestamp) and Income: {corr}')

# # Add title, x and y axis labels.
# plt.title('Date vs Income')
# plt.xlabel('Date')
# plt.ylabel('Income')
# plt.xticks(rotation=45, ha='right')
# plt.locator_params(axis='y', nbins=10)
# plt.tight_layout()
# plt.show()

# # Set up the plot figure
# plt.figure(figsize=(14, 7))

# # Plotting Expenses
# plt.subplot(1, 2, 1)  # 1 row, 2 columns, 1st subplot
# plt.scatter(hoa_df['Date'], hoa_df['Expenses'], color='red', alpha=0.5)
# plt.title('Expenses over Time')
# plt.xlabel('Date')
# plt.ylabel('Expenses')
# plt.xticks(rotation=45, ha='right')

# # Plotting Income
# plt.subplot(1, 2, 2)  # 1 row, 2 columns, 2nd subplot
# plt.scatter(hoa_df['Date'], hoa_df['Income'], color='blue', alpha=0.5)
# plt.title('Income over Time')
# plt.xlabel('Date')
# plt.ylabel('Income')
# plt.xticks(rotation=45, ha='right')

# # Show the plots
# plt.tight_layout()
# plt.show()



































'''
    Take th series with averages for specific intervals, and calculate the daily averages for missing days by interpolating the data
'''

# import matplotlib.pyplot as plt
# import pandas as pd

# # Load the top_200_youtubers.csv data
# top_youtubers_df = pd.read_csv('./CSVs/top_200_youtubers.csv')

# # Count the number of rows with a null value in the 'Category' column
# null_categories = top_youtubers_df['Category'].isnull().sum()
# print('Number of rows with a null value in the Category column:', null_categories)

# # Filter for Like Nastya and select relevant columns
# nastya_data = top_youtubers_df[
#     top_youtubers_df['Channel Name'] == 'Like Nastya'][[
#         'Avg. 1 Day', 
#         'Avg. 3 Day', 
#         'Avg. 7 Day', 
#         'Avg. 14 Day', 
#         'Avg. 30 day', 
#         'Avg. 60 day']].iloc[0]
# print('\n')
# print(nastya_data) 

# # Mapping intervals to days
# interval_to_day = {
#     'Avg. 1 Day': 1,
#     'Avg. 3 Day': 3,
#     'Avg. 7 Day': 7,
#     'Avg. 14 Day': 14,
#     'Avg. 30 day': 30,
#     'Avg. 60 day': 60
# }

# # Create a DataFrame with the known daily averages
# known_averages = pd.Series({interval_to_day[k]: v for k, v in nastya_data.items()})
# df = pd.DataFrame({'Daily Avg Views': known_averages})

# # Reindex to include all days from 1 to 60 and interpolate missing values
# df = df.reindex(range(1, 61)).interpolate(method='linear').fillna(method='bfill').fillna(method='ffill')

# # Plot the data
# plt.figure(figsize=(12, 6))
# plt.plot(df.index, df['Daily Avg Views'], marker='o', linestyle='-')
# plt.title('Daily Average Views for Like Nastya')
# plt.xlabel('Day')
# plt.ylabel('Average Views')
# plt.grid(axis='y', linestyle='--')
# plt.show()


'''TASK: FIND OUT WHAT THE TWO APPROACHES BELOW ARE MISSING. ARE THEY ON TRACK TO GENERATE THE PLOT ABOVE?'''
# # Divide each column by the corresponding number of days and fill NaN with 0
# daily_avg_views = nastya_data.div([1, 3, 7, 14, 30, 60]).fillna(0)

# # Create a list of days from 1 to 60
# days = list(range(1, 61))

# # Extend the daily_avg_views to match the length of days
# extended_daily_avg_views = daily_avg_views.tolist() + [daily_avg_views[-1]] * (len(days) - len(daily_avg_views))

# # Create the line chart
# plt.figure(figsize=(12, 6))
# plt.plot(days, extended_daily_avg_views, marker='o', linestyle='-')
# plt.title('Daily Average Views for Like Nastya')
# plt.xlabel('Day')
# plt.ylabel('Average Views')
# plt.grid(axis='y', linestyle='--')



# # Filter the data for the channel 'Like Nastya'
# nastya_data = top_youtubers_df[top_youtubers_df['Channel Name'] == 'Like Nastya']

# # Extract the relevant average viewers columns
# day_columns = ['Avg. 1 Day', 'Avg. 3 Day', 'Avg. 7 Day', 'Avg. 14 Day', 'Avg. 30 day', 'Avg. 60 day']

# # Calculate the daily averages for each specified period
# averages = {}
# for col in day_columns:
#     period = int(col.split(' ')[1])  # Extract the number of days from the column name
#     averages[period] = [nastya_data[col].iloc[0] / period for _ in range(period)]

# # Flatten the list of averages and create a continuous day range
# all_averages = [avg for sublist in averages.values() for avg in sublist]
# days = list(range(1, sum(len(v) for v in averages.values()) + 1))

# # Plot the daily averages
# plt.figure(figsize=(12, 6), facecolor='white')
# plt.plot(days, all_averages, marker='o')
# plt.title('Daily Average Viewers for Like Nastya')
# plt.xlabel('Day')
# plt.ylabel('Average Viewers')
# plt.grid(True)
# plt.xticks(rotation=45)
# plt.tight_layout()
# plt.show()





















'''
    SPLITTING STRINGS
'''

# import matplotlib.pyplot as plt
# import pandas as pd

# # Load the uk_universities.csv data
# uk_universities_df = pd.read_csv('./CSVs/uk_universities.csv')

# # Check the first few rows to understand the format of 'Student_enrollment' column
# print(uk_universities_df.head())
# print(uk_universities_df.info())
# stu_enroll = uk_universities_df['Student_enrollment'].head(n=15)
# print(stu_enroll)

# # Split the 'Student_enrollment' column into two new columns
# uk_universities_df[['Student_enrollment_Lower_Bound', 
#                     'Student_enrollment_Upper_Bound']] = uk_universities_df['Student_enrollment'].str.split('-', expand=True)

# # Convert the new columns to numeric type
# uk_universities_df['Student_enrollment_Lower_Bound'] = pd.to_numeric(
#     uk_universities_df['Student_enrollment_Lower_Bound'].str.replace(',', '')
# )
# uk_universities_df['Student_enrollment_Upper_Bound'] = pd.to_numeric(
#     uk_universities_df['Student_enrollment_Upper_Bound'].str.replace(',', '')
# )

# # Display the updated dataframe to confirm the changes
# print(uk_universities_df.head())

















'''
    CLEAN UP AND STANDARDIZE COLUMN NAMES TO MAKE THE CONCATINATION OF THE TWO FILES EASIER AND TO NOT LOSE ANY DATA

    WE ALSO PERFORM AN IQR ANALYSIS TO FIND OUTLIERS AND THEN SAVE OUR FINDINGS TO AN OUTPUT TSV FILE.
'''


# import pandas as pd

# # Read the TSV file into a DataFrame
# df_tsv = pd.read_csv('./CSVs/FAL Projects NY - West SM.tsv', sep='\t', skiprows=9)

# # Read the CSV file into a DataFrame
# df_csv = pd.read_excel('./CSVs/FAL Projects NY - office NY - FAL Proyectos.xlsx', skiprows=9)

# Standardize the column names
# df_csv.rename(columns={df_csv.columns[5]: 'Shipping/ Handling:'}, inplace=True)

# print(df_tsv.head())
# print(df_tsv.columns)
# print(df_tsv.info())
# print(df_csv.head())
# print(df_csv.columns)
# print(df_csv.info())

# Combine the dataframes
# combined_df = pd.concat([df_tsv, df_csv])

# # Display the first 5 rows
# print(combined_df.head())
# print(combined_df.info())

# # Filter to keep only the columns `Purchased By`, `Order Subtotal`, `Shipping/Handling`, `Vendor`, and `Created`
# filtered_df = combined_df[['Purchased By:', 
#                            'Order Sub Total:', 
#                            'Shipping/ Handling:', 
#                            'Vendor Name:', 
#                            'Create Date:']]

# # Drop rows where any of the columns `Purchased By`, `Order Subtotal`, `Shipping/Handling`, `Vendor`, and `Created` have null values
# filtered_df = filtered_df.dropna(subset=['Purchased By:', 
#                            'Order Sub Total:', 
#                            'Shipping/ Handling:', 
#                            'Vendor Name:', 
#                            'Create Date:'])

# # Remove '$' and ',' from the columns `Order Subtotal` and `Shipping/Handling` and convert to numeric
# for column_name in ['Order Sub Total:', 'Shipping/ Handling:']:
#     filtered_df[column_name] = filtered_df[column_name].astype(str).str.replace(r'[$,]', '', regex=True)
#     filtered_df[column_name] = pd.to_numeric(filtered_df[column_name])

# # Calculate a new column `Shipping/Handling Proportion` as `Shipping/Handling` divided by `Order Subtotal`
# filtered_df['Shipping/Handling Proportion'] = filtered_df['Shipping/ Handling:'] / filtered_df['Order Sub Total:']

# # Calculate the first quartile (Q1), third quartile (Q3), and interquartile range (IQR) for the `Shipping/Handling Proportion` column
# Q1 = filtered_df['Shipping/Handling Proportion'].quantile(0.25)
# Q3 = filtered_df['Shipping/Handling Proportion'].quantile(0.75)
# IQR = Q3 - Q1

# # Define the lower and upper bounds for outliers as Q1 - 1.5 * IQR and Q3 + 1.5 * IQR, respectively
# lower_bound = Q1 - 1.5 * IQR
# upper_bound = Q3 + 1.5 * IQR

# # Filter to keep only the rows where the `Shipping/Handling Proportion` is outside the lower and upper bounds
# outlier_df = filtered_df.loc[(filtered_df['Shipping/Handling Proportion'] < lower_bound) | 
#                              (filtered_df['Shipping/Handling Proportion'] > upper_bound)]

# # Sort the filtered dataframe by `Shipping/Handling Proportion` in descending order
# outlier_df = outlier_df.sort_values(by='Shipping/Handling Proportion', ascending=False)
# print(f'Outliers:\n{outlier_df}')

# Write the sorted dataframe to a new TSV file called "outlier_purchases.tsv"
# outlier_df.to_csv('./OutCSVs/outlier_purchases.tsv', sep='\t', index=False)















 



'''what are the significant aspects of the Hospital Survey Data on alcohol and drug abuse in regards to the number of entries, data completeness, data type diversity, and any apparent outliers'''

'''
    IQR ANALYSIS ON ONE FILE. I STARTED OUT WITH TWO BUT THEN AFTER RE-READING THE PROMPT, I REALIZED THAT ITS ASKING FOR A SPECIFIC FILE'S STATISTICS.
'''


# import pandas as pd

# # Load the files
# df_alcohol_drug_abuse = pd.read_excel('./CSVs/Hospital_Survey_Data_Alcohol_Drug_Abuse.xlsx', header=1)
# df_septicemia = pd.read_csv('./CSVs/Hospital_Survey_Data_Speticemia.csv', header=1)

# print(df_alcohol_drug_abuse.head())
# print(df_alcohol_drug_abuse.columns)
# print(df_alcohol_drug_abuse.info())

# print(df_septicemia.head())
# print(df_septicemia.columns)
# print(df_septicemia.info())

# max_discharges = df_alcohol_drug_abuse['Total Discharges'].max()
# min_discharges = df_alcohol_drug_abuse['Total Discharges'].min()
# print(f'\nThe range of discharges goes from {min_discharges} to {max_discharges}\n')

# max_discharges = df_septicemia['Total Discharges'].max()
# min_discharges = df_septicemia['Total Discharges'].min()
# print(f'\nThe range of discharges goes from {min_discharges} to {max_discharges}\n')

# # Combine the dataframes
# combined_df = pd.concat([df_alcohol_drug_abuse, df_septicemia])

# print(combined_df.info())

# max_discharges = combined_df['Total Discharges'].max()
# min_discharges = combined_df['Total Discharges'].min()
# print(f'\nThe range of discharges goes from {min_discharges} to {max_discharges}\n')

# # Filter columns containing '$'
# filtered_columns = [col for col in combined_df.columns if '$' in col]
# filtered_df = combined_df[filtered_columns]
# print(f'The filtered money columns:\n{filtered_df}')
# filt_count = filtered_df.value_counts().sum()
# print(f'\nIts length:\n{filt_count}')

# # Convert all columns in filtered_df to numeric, coercing errors to NaN
# # for col in filtered_df.columns:
# #     filtered_df[col] = pd.to_numeric(filtered_df[col])

# # Descriptive statistics for filtered columns
# print("Descriptive Statistics for columns with '$':")
# print(filtered_df.describe())

# # Calculate and print IQR for each filtered column
# print("\nInterquartile Range (IQR) for columns with '$':")
# print((filtered_df.quantile(0.75) - filtered_df.quantile(0.25)))

# # Calculate and print outlier bounds for each filtered column
# print("\nOutlier Bounds for columns with '$':")
# Q1 = filtered_df.quantile(0.25)
# Q3 = filtered_df.quantile(0.75)
# IQR = Q3 - Q1
# upper_bound = Q3 + 1.5 * IQR
# lower_bound = Q1 - 1.5 * IQR
# bounds_df = pd.DataFrame({'Lower Bound': lower_bound, 'Upper Bound': upper_bound})
# print(bounds_df)

# # Identify and print the number of outliers in each filtered column
# print("\nNumber of Outliers for columns with '$':")
# for col in filtered_columns:
#     outliers = filtered_df[(filtered_df[col] < lower_bound[col]) | (filtered_df[col] > upper_bound[col])]
#     print(f"{col}: {len(outliers)} outliers")


'''SAME ANALYSIS BUT WE PRINT IT IN MARKDOWN FORM WHICH IS MORE ORGANIZED'''
# # Filter columns containing '$'
# filtered_columns = [col for col in df_alcohol_drug_abuse.columns if '$' in col]
# filtered_df = df_alcohol_drug_abuse[filtered_columns]

# # Descriptive statistics for filtered columns
# print("Descriptive Statistics for columns with '$':")
# print(filtered_df.describe().to_markdown(numalign="left", stralign="left"))

# # Calculate and print IQR for each filtered column
# print("\nInterquartile Range (IQR) for columns with '$':")
# print((filtered_df.quantile(0.75) - filtered_df.quantile(0.25)).to_markdown(numalign="left", stralign="left"))

# # Calculate and print outlier bounds for each filtered column
# print("\nOutlier Bounds for columns with '$':")
# Q1 = filtered_df.quantile(0.25)
# Q3 = filtered_df.quantile(0.75)
# IQR = Q3 - Q1
# upper_bound = Q3 + 1.5 * IQR
# lower_bound = Q1 - 1.5 * IQR
# bounds_df = pd.DataFrame({'Lower Bound': lower_bound, 'Upper Bound': upper_bound})
# print(bounds_df.to_markdown(numalign="left", stralign="left"))

# # Identify and print the number of outliers in each filtered column
# print("\nNumber of Outliers for columns with '$':")
# for col in filtered_columns:
#     outliers = filtered_df[(filtered_df[col] < lower_bound[col]) | (filtered_df[col] > upper_bound[col])]
#     print(f"{col}: {len(outliers)} outliers")





















'''
    HERE WE CREATE A VERY COOL BAR GRAPH USING THE NEON AND BLACK COLORWAY
'''

# import pandas as pd
# import matplotlib.pyplot as plt

# # Load the ByrdBiteAdData.csv
# df_byrdbite = pd.read_csv('./CSVs/ByrdBiteAdData.csv')
# print(df_byrdbite.head())
# print(df_byrdbite.info())

# # Count the occurrences of each age
# df_age_counts = df_byrdbite['Age'].value_counts().sort_index()

# # Define neon colors for the bars
# neon_colors = ['#FFD700', 
#                '#FF6347', 
#                '#FF1493', 
#                '#00FF00', 
#                '#00BFFF', 
#                '#8A2BE2', # Repeat the color list to cover all bars
#                '#FF4500'] * (len(df_age_counts) // 7 + 1)  


# # Adjusting the plot to have a black background behind the bars as well
# plt.figure(figsize=(10, 6), facecolor='black')
# ax = plt.subplot(111, facecolor='black')
# ax.bar(df_age_counts.index, df_age_counts.values, color=neon_colors[:len(df_age_counts)])
# ax.set_title('Number of Times Each Age Population Appears', color='white')
# ax.set_xlabel('Age', color='white')
# ax.set_ylabel('Frequency', color='white')
# ax.tick_params(axis='x', colors='white')
# ax.tick_params(axis='y', colors='white')
# ax.grid(axis='y', linestyle='--', alpha=0.7, color='white')
# plt.show()


'''2nd approach with altair'''
# import altair as alt

# # Drop null values in `Age`
# df_byrdbite.dropna(subset = ['Age'], inplace=True)

# # Calculate the frequency of each unique value in `Age`
# age_counts = df_byrdbite['Age'].value_counts()

# # Convert the series `age_counts` to a DataFrame
# age_counts_df = age_counts.reset_index()

# # Rename the columns
# age_counts_df.columns = ['Age', 'Frequency']

# # Sort the DataFrame by `Age` in ascending order
# age_counts_df = age_counts_df.sort_values('Age')

# # Create a bar chart with `Age` on the x-axis and `Frequency` on the y-axis
# chart = alt.Chart(age_counts_df).mark_bar().encode(
#     x='Age',
#     y='Frequency',
#     tooltip = ['Age', 'Frequency']
# ).properties(
#     title = 'Frequency of Age Groups'
# ).interactive()

# Save the chart
# chart.save('./OutCSVs/age_groups_frequency_bar_chart.html')

















# import pandas as pd

# # Load the current_accounts.csv
# df_current_accounts = pd.read_csv('./CSVs/current_accounts.csv')

# # Display the first few rows of the dataframe to understand its structure and contents
# print(df_current_accounts.head())
# print(df_current_accounts.columns)
# print(df_current_accounts.info())

# uni_statuses = df_current_accounts['Status'].unique()
# print(f'The unique statuses are the following:\n{uni_statuses}')
# val_counts = df_current_accounts['Status'].value_counts()
# print(f'And there are this many:\n{val_counts}')

# # Convert Balance to numeric, handling commas as decimal points 
# df_current_accounts['Balance'] = pd.to_numeric(df_current_accounts['Balance'].str.replace(',', '.'))

# # Filter data by 'PENDIENTE' and 'PARCIAL' statuses
# df_pendiente = df_current_accounts[df_current_accounts['Status'] == 'PENDIENTE']
# df_parcial = df_current_accounts[df_current_accounts['Status'] == 'PARCIAL']

# # Calculate mean and median for 'PENDIENTE' status
# mean_pendiente = df_pendiente['Balance'].mean()
# median_pendiente = df_pendiente['Balance'].median()

# # Calculate mean and median for 'PARCIAL' status
# mean_parcial = df_parcial['Balance'].mean()
# median_parcial = df_parcial['Balance'].median()

# print('Mean Balance for PENDIENTE status:', mean_pendiente)
# print('Median Balance for PENDIENTE status:', median_pendiente)
# print('Mean Balance for PARCIAL status:', mean_parcial)
# print('Median Balance for PARCIAL status:', median_parcial)




















'''
    COHORT ANALYSIS AND A FUNCTION TO CONVERT DATES
'''



# # Function to handle non-empty date strings
# def safe_to_datetime(date_str, date_format='%Y-%m-%d'):
#     try:
#         return pd.to_datetime(date_str, format=date_format)
#     except (ValueError, TypeError):
#         return pd.NaT

# # Apply the conversion safely to each value
# df['date_column'] = df['date_column'].apply(lambda x: safe_to_datetime(x) if x else pd.NaT)






# import pandas as pd

# # Load the dataset
# file_path = './CSVs/data_offers_orders_joined.csv'
# data = pd.read_csv(file_path, sep=';')

# print(data.info())



# # Replace missing manager_id values with "HQ"
# data['manager_id'].fillna('HQ', inplace=True)
# print("AFTER")
# print(data.info())
# uni_ids = data['manager_id'].value_counts()
# print(f'Unique manager ids (BEFORE):\n{uni_ids}')

# # Convert net_offer_sum and invoice_sum to numeric values (they're currently strings with commas for decimals)
# data['net_offer_sum'] = data['net_offer_sum'].str.replace(',', '.').astype(float)
# data['invoice_sum'] = data['invoice_sum'].str.replace(',', '.').astype(float)

# # Group the data by manager_id and calculate performance metrics
# manager_performance = data.groupby('manager_id').agg(
#     projects_count=pd.NamedAgg(column='id', aggfunc='count'),
#     avg_net_offer_sum=pd.NamedAgg(column='net_offer_sum', aggfunc='mean'),
#     avg_invoice_sum=pd.NamedAgg(column='invoice_sum', aggfunc='mean')
# ).reset_index()

# # Display the summarized performance metrics by manager
# print(manager_performance)
    





# # Display the first few rows and column names to understand the structure
# print(df.head()) 
# print(df.columns)
# print(df.info())

# uni_ids = df['manager_id'].value_counts()
# print(f'Unique manager ids (BEFORE):\n{uni_ids}')

# # Replace missing manager_id with "HQ"
# df['manager_id'] = df['manager_id'].fillna('HQ')

# uni_ids = df['manager_id'].value_counts()
# print(f'Unique manager ids (AFTER):\n{uni_ids}')

# # Convert necessary columns to numeric after replacing commas with dots
# df['net_offer_sum'] = df['net_offer_sum'].str.replace(',', '.').astype(float)
# df['net_order_sum'] = df['net_order_sum'].str.replace(',', '.').astype(float)
# df['invoice_sum'] = df['invoice_sum'].str.replace(',', '.').astype(float)

# # Convert relevant date columns to datetime 
# # NOTE: here, the coerce arg has a positive effect and the format wasnt needed
# date_cols = ['offer_submission_date', 'offer_reject_date', 'order_confirmation_date',
#              'expected_order_start_date', 'expected_order_done_date', 'actual_order_done_date', 'invoice_date']
# for col in date_cols:
#     df[col] = pd.to_datetime(df[col], errors='coerce')

# print('AFTER date time convert')
# print(df.info())

# # Group by manager_id and calculate relevant performance metrics
# cohort_analysis = df.groupby('manager_id').agg(
#     offers_count=('id', 'count'),
#     confirmed_offers=('offer_status', lambda x: (x == 'Confirmed').sum()),
#     rejected_offers=('offer_status', lambda x: (x == 'Rejected').sum()),
#     total_offer_value=('net_offer_sum', 'sum'),
#     total_order_value=('net_order_sum', 'sum'),
#     total_invoice_value=('invoice_sum', 'sum'),
#     avg_offer_value=('net_offer_sum', 'mean'),
#     avg_order_value=('net_order_sum', 'mean'),
#     avg_invoice_value=('invoice_sum', 'mean')
# ).reset_index()

# print(cohort_analysis)


























'''
    HERE WE HAVE THREE DIFFERENT APPROACHES TO GENERATING BAR GRAPHS WHERE EACH DATE HAS TWO BARS REPRESENTING TWO DIFFERENT COLUMNS. I LIKE THE 3RD APPROACH. ITS THE CLEANEST.

    NOTE: FIND OUT HOW THE LAST APPROACH IS SO CLEAN, I.E., HOW IS THE DATE BEING FORMATTED AUTOMATICALLY???? ALSO, FIGURE OUT HOW TO MAKE A GROUPED BAR CHART USING THE ALTAIR MODULE. 
'''


# import pandas as pd
# import matplotlib.pyplot as plt

# # Load the dataset
# trapiche_df = pd.read_csv('./CSVs/trapiche_ingenio_nv.csv')
# print('BEFORE')
# print(trapiche_df.info())

# # Convert 'Fecha' to datetime
# trapiche_df['Fecha'] = pd.to_datetime(trapiche_df['Fecha'], format='%Y-%m-%d')

# print('AFTER')
# print(trapiche_df.info())

# # Group by 'Fecha' and sum 'Bruto' and 'Neto'
# sums_df = trapiche_df.groupby('Fecha').agg({'Bruto': 'sum', 'Neto': 'sum'}).reset_index()
# grouped_data = trapiche_df.groupby('Fecha')[['Bruto', 'Neto']].sum().reset_index()


# # Plotting
# plt.figure(figsize=(10, 6), facecolor='white')
# sums_df.plot(x='Fecha', y=['Bruto', 'Neto'], kind='bar', color=['blue', 'green'], alpha=0.7)
# plt.title('Sum of Bruto and Neto by Date')
# plt.xlabel('Date')
# plt.ylabel('Sum')
# plt.xticks(rotation=45)
# plt.legend(title='Column')
# plt.tight_layout()
# plt.show()

# import altair as alt

# melted_data = grouped_data.melt(id_vars='Fecha', 
#                                 value_vars=['Bruto', 'Neto'], 
#                                 var_name='Metric', 
#                                 value_name='Sum')
# chart = alt.Chart(melted_data).mark_bar().encode(
#     x=alt.X('date:T', title='Date'),
#     y=alt.Y('Sum:Q', title='Sum'),
#     color='Metric:N',
#     column='Metric:N'
# ).properties(
#     title='Sum of Bruto and Neto per Date'
# ).resolve_scale(
#     y='independent'
# )

# chart = alt.Chart(melted_data).mark_bar().encode(
#     x=alt.X('date:T', title='Date'),
#     y=alt.Y('Sum:Q', title='Sum'),
#     color='Metric:N',
#     tooltip=['date:T', 'Metric:N', 'Sum:Q']
# ).properties(
#     width=800,
#     height=400,
#     title='Sum of Bruto and Neto per Date'
# )


# # Display the chart
# chart.save('./OutCSVs/bruto.html')




# import altair as alt

# # Load the dataset
# df = pd.read_csv('./CSVs/trapiche_ingenio_nv.csv')

# # Convert `Fecha` to datetime
# df['Fecha'] = pd.to_datetime(df['Fecha'], format='%Y-%m-%d')

# # Aggregate by `Fecha`, summing `Bruto` and `Neto`
# df_agg = df.groupby('Fecha')[['Bruto', 'Neto']].sum().reset_index()

# # Reshape to long format
# df_long = df_agg.melt(id_vars='Fecha', var_name='Variable', value_name='Value')

# # Create the bar chart
# chart = alt.Chart(df_long).mark_bar().encode(
#     x=alt.X('Fecha:T', axis=alt.Axis(title='Date', labelAngle=-45)),
#     y=alt.Y('Value:Q', axis=alt.Axis(title='Total')),
#     color='Variable:N',
#     tooltip=['Fecha', 'Variable', 'Value']
# ).properties(
#     title='Total Bruto and Neto by Date'
# ).interactive()

# # Save the chart
# chart.save('./OutCSVs/bruto_neto_by_date_bar_chart.html')


# # Now let's create the bar chart
# import matplotlib.pyplot as plt
# import pandas as pd

# # Load the dataset
# df2 = pd.read_csv('./CSVs/trapiche_ingenio_nv.csv')

# grouped_data = df2.groupby('Fecha')[['Bruto', 'Neto']].sum().reset_index()

# # Set the figure size for better readability
# plt.figure(figsize=(14, 7))

# # Plotting the data
# plt.bar(grouped_data['Fecha'], grouped_data['Bruto'], label='Bruto', alpha=0.6)
# plt.bar(grouped_data['Fecha'], grouped_data['Neto'], label='Neto', alpha=0.6)

# # Adding labels and title
# plt.xlabel('Fecha')
# plt.ylabel('Suma')
# plt.title('Suma de Bruto y Neto por Fecha')
# plt.xticks(rotation=45)
# plt.legend()

# # Show the plot
# plt.tight_layout()
# plt.show()







'''
    CHECK TO SEE IF ANY OF THE FOLLOWING EXAMPLES CAN BE USED TO SATISFY THE VISUALIZATION ABOVE. WE RAN INTO MANY ISSUES ABOVE, TRYING TO GENERATE A GROUPED BAR GRAPH USING ALTAIR 
'''

# import altair as alt
# import pandas as pd

# # Create a DataFrame
# data = {
#     'Year': ['2022', '2022', '2022', '2023', '2023', '2023', '2024', '2024', '2024'],
#     'Product': ['Product A', 'Product B', 'Product C', 'Product A', 'Product B', 'Product C', 'Product A', 'Product B', 'Product C'],
#     'Sales': [25, 22, 20, 32, 30, 28, 34, 35, 32]
# }
# df = pd.DataFrame(data)

# # Create the grouped bar graph
# bar_chart = alt.Chart(df).mark_bar().encode(
#     x=alt.X('Year:N', title='Year'),
#     y=alt.Y('Sales:Q', title='Sales'),
#     color='Product:N',
#     column='Product:N'
# ).properties(
#     width=100,
#     title='Sales per Product Over Time'
# )

# bar_chart.show()






# import altair as alt
# import pandas as pd

# # Create a DataFrame
# data = {
#     'Year': ['2022', '2022', '2022', '2023', '2023', '2023', '2024', '2024', '2024'],
#     'Product': ['Product A', 'Product B', 'Product C', 'Product A', 'Product B', 'Product C', 'Product A', 'Product B', 'Product C'],
#     'Sales': [25, 22, 20, 32, 30, 28, 34, 35, 32]
# }
# df = pd.DataFrame(data)

# # Create the grouped bar graph
# bar_chart = alt.Chart(df).mark_bar().encode(
#     x=alt.X('Year:O', title='Year', axis=alt.Axis(labelAngle=0)),
#     y=alt.Y('Sales:Q', title='Sales'),
#     color='Product:N',
#     column='Product:N'
# ).properties(
#     width=100,
#     title='Sales per Product Over Time'
# )

# bar_chart.show()

















'''
    USE THIS WHEN IT WONT or CANT FIND A FILE IN CSVs DIRECTORY... IT MIGHT BE HIDDEN OR HAVE A DIFF NAME BEHIND THE SCENES!
'''

# # Print current working directory
# import os
# print("Current Working Directory:", os.getcwd())

# # List files in the specific directory
# print("Files in './CSVs/':", os.listdir('./CSVs/'))


'''
    LEARN HOW TO USE THE STRING METHOD 'CONTAINS' WHICH JUST ASKS THE RELEVANT ENTRIES IF IT CONTAINS SOME STRING OR SUBSTRING. 
'''

# import pandas as pd

# # Load the Excel file
# hospital_df = pd.read_excel('./CSVs/Hospital_Survey_Data_Alcohol_Drug_Abuse.xlsx', skiprows=1)

# # Display the first few rows to understand the structure of the data
# print(hospital_df.head())
# print(hospital_df.columns)
# print(hospital_df.info())

# uni_drg = hospital_df['DRG Definition'].value_counts()
# print(f'unique DRG:\n{uni_drg}')

# # Filter the data for cases with and without rehabilitation therapy
# cases_with_rehab = hospital_df[hospital_df['DRG Definition'].str.contains('W REHABILITATION THERAPY')]
# cases_without_rehab = hospital_df[hospital_df['DRG Definition'].str.contains('W/O REHABILITATION THERAPY')]

# # Calculate the average discharge rates for both groups
# avg_discharge_with_rehab = cases_with_rehab['Total Discharges'].mean()
# avg_discharge_without_rehab = cases_without_rehab['Total Discharges'].mean()

# print('Average Discharges with Rehabilitation Therapy:', avg_discharge_with_rehab)
# print('Average Discharges without Rehabilitation Therapy:', avg_discharge_without_rehab)






































# import pandas as pd
# import matplotlib.pyplot as plt

# # Creating the dataframe
# data = {
#     'Information_Availability': ['High', 'Medium', 'Low', 'High', 'Medium', 'Low', 'High', 'Medium', 'Low', 'High', 'Medium'],
#     'House_Cost': ['Low', 'Medium', 'High', 'Medium', 'Low', 'Medium', 'High', 'Medium', 'Low', 'Medium', 'High'],
#     'School_Quality': ['Good', 'Average', 'Bad', 'Excellent', 'Good', 'Bad', 'Excellent', 'Good', 'Bad', 'Excellent', 'Good'],
#     'Trust_in_Police': ['High', 'Medium', 'Low', 'High', 'Medium', 'Low', 'High', 'Medium', 'Low', 'High', 'Medium'],
#     'Street_Quality': ['Good', 'Average', 'Bad', 'Excellent', 'Good', 'Bad', 'Excellent', 'Good', 'Bad', 'Excellent', 'Good'],
#     'Events': ['Many', 'Some', 'Few', 'Many', 'Some', 'Few', 'Many', 'Some', 'Few', 'Many', 'Some'],
#     'Happiness': ['Happy', 'Content', 'Sad', 'Very Happy', 'Happy', 'Sad', 'Very Happy', 'Happy', 'Sad', 'Very Happy', 'Happy']
# }

# df = pd.DataFrame(data)

# # Mapping categorical data to numeric values
# house_cost_map = {'Low': 1, 'Medium': 2, 'High': 3}
# school_quality_map = {'Bad': 1, 'Average': 2, 'Good': 3, 'Excellent': 4}

# df['House_Cost'] = df['House_Cost'].map(house_cost_map)
# df['School_Quality'] = df['School_Quality'].map(school_quality_map)

# # Plotting the scatter plot
# plt.figure(figsize=(10, 6))
# plt.scatter(df['House_Cost'], df['School_Quality'], c='blue', marker='o')
# plt.title('Relationship between House Cost and School Quality')
# plt.xlabel('House Cost (1: Low, 2: Medium, 3: High)')
# plt.ylabel('School Quality (1: Bad, 2: Average, 3: Good, 4: Excellent)')
# plt.grid(True)
# plt.show()



























# import pandas as pd
# import matplotlib.pyplot as plt

# # Creating, then initializing the dataframe
# data = {
#     'Data_Availability': ['Good', 'Fair', 'Excellent', 'Poor', 'Good', 'Fair', 'Excellent', 'Poor', 'Good'],
#     'Affordable_Housing': ['High', 'Low', 'High', 'Low', 'Medium', 'Low', 'High', 'Low', 'Medium'],
#     'Educational_Quality': ['Excellent', 'Good', 'Poor', 'Excellent', 'Good', 'Excellent', 'Poor', 'Excellent', 'Good'],
#     'Trust_in_Law_Enforcement': ['High', 'Medium', 'Low', 'High', 'Medium', 'High', 'Low', 'High', 'Medium'],
#     'Roads_and_Infrastructure': ['Good', 'Fair', 'Excellent', 'Poor', 'Good', 'Fair', 'Excellent', 'Poor', 'Good'],
#     'Community_Events': ['Many', 'Few', 'Many', 'Few', 'Many', 'Few', 'Many', 'Few', 'Many'],
#     'Mental_Wellness': ['Good', 'Fair', 'Excellent', 'Poor', 'Good', 'Fair', 'Excellent', 'Poor', 'Good']
# }

# df = pd.DataFrame(data)

# # Plotting the bar chart
# plt.figure(figsize=(12, 8))
# df.groupby('Community_Events')['Mental_Wellness'].value_counts().unstack().plot(kind='bar', 
#                                                                                 stacked=True, 
#                                                                                 color=['skyblue', 'lightgreen', 'salmon', 'orange'], 
#                                                                                 ax=plt.gca()
# )
# plt.title('Distribution of Mental Wellness for Cities with Different Levels of Community Events')
# plt.xlabel('Community Events')
# plt.ylabel('Count of Mental Wellness')
# plt.xticks(rotation=0)
# plt.legend(title='Mental Wellness')
# plt.grid(axis='y')
# plt.show()








''' MA  '''
# import pandas as pd
# import altair as alt

# # Create a DataFrame with the columns `Community Events` and `Mental Wellness` using the data provided in the query.
# data = {
#     'Community Events': ['Many', 'Few', 'Many', 'Few', 'Many', 'Few', 'Many', 'Few', 'Many'],
#     'Mental Wellness': ['Good', 'Fair', 'Excellent', 'Poor', 'Good', 'Fair', 'Excellent', 'Poor', 'Good']
# }
# df = pd.DataFrame(data)

# # Count the occurrences of each unique combination of `Community Events` and `Mental Wellness` and store the result in `grouped_data`.
# grouped_data = df.groupby(['Community Events', 'Mental Wellness']).size().reset_index(name='Count')

# # Create a bar chart using the data in `grouped_data`, with `Community Events` on the x-axis, the count on the y-axis, and separate bars for each `Mental Wellness` category.
# chart = alt.Chart(grouped_data).mark_bar().encode(
#     x='Community Events',
#     y='Count',
#     color='Mental Wellness',
#     tooltip=['Community Events', 'Mental Wellness', 'Count']
# ).properties(
#     title='Distribution of Mental Wellness by Community Events'
# ).interactive()

# # Save the chart in json file
# chart.save('./OutPlots/mental_wellness_by_community_events_bar_chart.html')









''' MB  '''
# import pandas as pd
# import matplotlib.pyplot as plt

# # Create the data frame
# data = {
#     "Data Availability": ["Good", "Fair", "Excellent", "Poor", "Good", "Fair", "Excellent", "Poor", "Good"],
#     "Affordable Housing": ["High", "Low", "High", "Low", "Medium", "Low", "High", "Low", "Medium"],
#     "Educational Quality": ["Excellent", "Good", "Poor", "Excellent", "Good", "Excellent", "Poor", "Excellent", "Good"],
#     "Trust in Law Enforcement": ["High", "Medium", "Low", "High", "Medium", "High", "Low", "High", "Medium"],
#     "Roads and Infrastructure": ["Good", "Fair", "Excellent", "Poor", "Good", "Fair", "Excellent", "Poor", "Good"],
#     "Community Events": ["Many", "Few", "Many", "Few", "Many", "Few", "Many", "Few", "Many"],
#     "Mental Wellness": ["Good", "Fair", "Excellent", "Poor", "Good", "Fair", "Excellent", "Poor", "Good"]
# }

# df = pd.DataFrame(data)

# # Pivot table to aggregate Mental Wellness by Community Events
# pivot_table = df.pivot_table(index="Community Events", columns="Mental Wellness", aggfunc="size", fill_value=0)

# # Plot
# pivot_table.plot(kind="bar", stacked=True, figsize=(10, 6))
# plt.title("Distribution of Mental Wellness by Community Events")
# plt.xlabel("Community Events")
# plt.ylabel("Number of Cities")
# plt.xticks(rotation=45)
# plt.legend(title="Mental Wellness")
# plt.tight_layout()

# plt.show()
























'''
    I CANT GET THE DATE COLUMN TO BE READ IN ON THIS ONE, WHY? THE GOAL IS TO SIMPLY SPLIT THE DATE COLUMN INTO TWO COLUMNS, THE MONTH AND DAY.
'''
# import pandas as pd

# # Load the data from the TSV file
# strawberry_sales_path = './CSVs/STRAWBERRY SALES 2023 - Sheet1.tsv'
# strawberry_sales_data = pd.read_csv(strawberry_sales_path, sep='\t', skiprows=2)

# # Display the first few rows and recheck the data summary
# print(strawberry_sales_data.head()) 
# print(strawberry_sales_data.info(verbose=True)) 
# print(strawberry_sales_data.describe())

# # First, let's clean the dataset and remove any rows with NaN values and unnecessary headers
# cleaned_data = strawberry_sales_data.dropna().reset_index(drop=True)

# # The relevant data starts from the 2nd row (index 1)
# cleaned_data.columns = cleaned_data.iloc[0]
# cleaned_data = cleaned_data.drop(0).reset_index(drop=True)

# # Rename the poorly formatted col names
# cleaned_data.columns = ['DATE', 'CLAMSHELLS', 'NUM_BOXES', 'KILOS', 'PRICE_PER_BOX', 'TOTAL', 'PRODUCT', 'TYPE_OF_PRODUCT']

# # Convert the DATE column to datetime format
# cleaned_data['DATE'] = pd.to_datetime(cleaned_data['DATE'], format='%d-%b-%y')

# # Extract month and day of the week from the DATE column
# cleaned_data['MONTH'] = cleaned_data['DATE'].dt.month_name()
# cleaned_data['DAY_OF_WEEK'] = cleaned_data['DATE'].dt.day_name()

# print(cleaned_data.head())








'''ATTEMPT 2'''
# import pandas as pd

# # Read the TSV file into a DataFrame
# df = pd.read_csv('./CSVs/STRAWBERRY SALES 2023 - Sheet1.tsv', delimiter='\t')

# # Display the first 5 rows
# print(df.head())

# # Print the column names and their data types
# print(df.info())

# # Set the column names to the values in the 2nd row
# df.columns = df.iloc[1]

# # Drop the first 2 rows
# df = df.iloc[2:].copy()

# # Convert the `DATE` column to datetime
# df['DATE'] = pd.to_datetime(df['DATE'])

# # Extract month name and day of week from `DATE`
# df['MONTH'] = df['DATE'].dt.strftime('%B')
# df['DAY_OF_WEEK'] = df['DATE'].dt.strftime('%A')

# # Show the first 5 rows of the columns `DATE`, `MONTH`, and `DAY_OF_WEEK`
# print(df[['DATE', 'MONTH', 'DAY_OF_WEEK']].head())
























# import pandas as pd

# # Load the dataset
# file_path = './CSVs/Data Set #13 report.csv'
# df = pd.read_csv(file_path, encoding='ascii')

# print(df.head())
# print(df.info(verbose=True))
# print('\n')
# print(df['inventory_type'].unique())
# print('\n')

# # Convert COST and sell_price columns to numeric after removing the dollar sign and commas
# df['COST'] = df['COST'].replace('[\$,]', '', regex=True).astype(float)
# df['sell_price'] = df['sell_price'].replace('[\$,]', '', regex=True).astype(float)

# # Clean up the 'qty_in_transit units' and 'qty_on_hand units' columns by removing commas and converting to numeric
# cols_to_clean = ['qty_in_transit units', 'qty_on_hand units']
# for col in cols_to_clean:
#     df[col] = df[col].replace('[,"]', '', regex=True).astype(float)

# # Group by inventory_type and calculate the required sums
# summary_df = df.groupby('inventory_type').agg({
#     'COST': 'sum',
#     'sell_price': 'sum',
#     'qty_in_transit units': 'sum',
#     'qty_on_hand units': 'sum'
# }).reset_index()

# print(summary_df.info(verbose=True))

# # Rename columns for clarity
# summary_df.columns = ['Inventory Type', 
#                       'Total Cost', 
#                       'Total Sell Price', 
#                       'Total Quantity in Transit', 
#                       'Total Quantity on Hand'
# ]

# # Display the summary table
# print(summary_df)








# import pandas as pd

# # Read the CSV file into a DataFrame
# df = pd.read_csv('./CSVs/Data Set #13 report.csv')

# # Display the first 5 rows
# print(df.head())

# # Print the column names and their data types
# print(df.info())

# # Remove '$' and ',' from columns and convert to numeric
# for col in ['COST', 'sell_price', 'qty_in_transit units', 'qty_on_hand units']:
#   df[col] = df[col].astype(str).str.replace(r'[\$,]', '', regex=True)
#   df[col] = pd.to_numeric(df[col], errors='coerce')

# print(df.info())

# # Group by `inventory_type` and sum up `COST`, `sell_price`, `qty_in_transit units`, and `qty_on_hand units` columns
# summary_df = (
#     df.groupby('inventory_type')[['COST', 'sell_price', 'qty_in_transit units', 'qty_on_hand units']]
#     .sum()
#     .round(2)
# )

# # Print the resulting dataframe
# print("Summary Statistics by Inventory Type:\n")
# print(summary_df)























'''
    BINARY LIST OF NON-ZERO INCOME ENTRIES
'''
# # Load the Excel file, skipping the first row and setting the header
# import pandas as pd

# df_outcomes_incomes = pd.read_excel('./CSVs/outcomes_incomes_fs.xlsx', skiprows=1)

# print(df_outcomes_incomes.head())
# print(df_outcomes_incomes.info(verbose=True))

# # Set the correct header and drop the first column
# new_header = df_outcomes_incomes.iloc[0] # first row as header

# df_outcomes_incomes = df_outcomes_incomes[1:10] # take the data rows

# df_outcomes_incomes.columns = new_header # set the header

# df_outcomes_incomes = df_outcomes_incomes.reset_index(drop=True)

# # Display the cleaned dataframe
# print(df_outcomes_incomes.head())

# # Rename the columns to month names
# month_names = ['January', 'February', 'March', 'April', 'May', 'June', 'July', 'August', 'September', 'October', 'November']
# df_outcomes_incomes.columns = [''] + ['income_type'] + month_names

# # Create a binary dataframe where 1 indicates non-zero income and 0 indicates zero income
# binary_incomes = df_outcomes_incomes.map(lambda x: 1 if x != 0 else 0)

# # Add the income types back to the dataframe
# binary_incomes['income_type'] = df_outcomes_incomes['income_type']

# # Display the binary dataframe
# print(binary_incomes)
























# import pandas as pd

# # Remove the first row and set the second row as the header
# strawberry_sales_df = pd.read_csv('./CSVs/STRAWBERRY SALES 2023 - Sheet1.tsv', sep='\	', header=1)

# # Display the first few rows to confirm the changes
# print(strawberry_sales_df.head())

# # Rename the columns to remove 'Unnamed' and make them more readable
# strawberry_sales_df.columns = ['DATE', 'CLAMSHELLS', 'BOXES', 'KILOS', 'PRICE_PER_BOX', 'TOTAL', 'PRODUCT', 'TYPE_OF_PRODUCT']

# # The unique values show that there is an entry '$/BOX' which is causing the conversion issue. We need to remove or handle this entry.
# # Let's filter out rows where PRICE_PER_BOX is '$/BOX' and then proceed with the conversion.

# # Filter out rows with '$/BOX'
# strawberry_sales_df = strawberry_sales_df[strawberry_sales_df['PRICE_PER_BOX'] != '$/BOX']

# # Convert relevant columns to numeric values for analysis
# strawberry_sales_df['CLAMSHELLS'] = pd.to_numeric(strawberry_sales_df['CLAMSHELLS'])
# strawberry_sales_df['BOXES'] = pd.to_numeric(strawberry_sales_df['BOXES'])
# strawberry_sales_df['KILOS'] = pd.to_numeric(strawberry_sales_df['KILOS'])
# strawberry_sales_df['PRICE_PER_BOX'] = strawberry_sales_df['PRICE_PER_BOX'].replace('[\$,]', '', regex=True).astype(float)
# strawberry_sales_df['TOTAL'] = strawberry_sales_df['TOTAL'].replace('[\$,]', '', regex=True).astype(float)

# # Display the cleaned dataframe
# print(strawberry_sales_df.head())

# # Calculate the average revenue per sale for organic and conventional strawberries
# # Group by TYPE_OF_PRODUCT and calculate the mean of the TOTAL column
# average_revenue_per_sale = strawberry_sales_df.groupby('TYPE_OF_PRODUCT')['TOTAL'].mean()
# print(average_revenue_per_sale)























'''
    CREATE A TSV from a csv
'''
# import pandas as pd

# # Load the CSV file
# file_path = './CSVs/Data Set #13 report.csv'
# data = pd.read_csv(file_path)

# # Display the first few rows to understand its structure
# print(data.head())

# # Extract the required columns: item_num, COST, sell_price, and AGE OF INVENTORY DAYS
# inventory_data_df = data[['item_num', 'COST', 'sell_price', 'AGE OF INVENTORY DAYS']]

# # Save the extracted data to a TSV file
# tsv_file_path = './OutCSVs/inventory_data.tsv'
# inventory_data_df.to_csv(tsv_file_path, sep='\t', index=False)

# print(tsv_file_path)


























'''
    TWO APPROACHES TO AGGREGATING DATA
'''
# # Load the grades.xlsx file and inspect the sheets
# import pandas as pd

# # Load the 'Period 2' sheet to inspect the data
# period_2_df = pd.read_excel('./CSVs/grades.xlsx', sheet_name='Period 2')
# print(period_2_df.head())

# # Create a new table with students and their aggregated scores across homework, midterm, and final exam for Period 2
# period_2_df['Aggregated Score'] = period_2_df[['Homework', 'Midterm', 'Final Exam']].sum(axis=1)

# # Select the required columns: Student and Aggregated Score
# aggregated_scores_df = period_2_df[['Student', 'Aggregated Score']]
# print(aggregated_scores_df)


'''APPROACH 2'''
# # Calculate the total score for each student by summing up their scores across Homework, Midterm, and Final Exam
# period_2_df['Total Score'] = period_2_df['Homework'] + period_2_df['Midterm'] + period_2_df['Final Exam']

# # Create a new table with students and their aggregated scores
# aggregated_scores_df = period_2_df[['Student', 'Total Score']]


























# import pandas as pd

# # Loading with ISO-8859-1 encoding
# real_estate_df = pd.read_csv('./CSVs/Real Estate Mumbai Database - Rgdcvvvh.csv', encoding='ISO-8859-1')
# print(real_estate_df.head())

# # Convert the TRANSACTION DATE column to datetime format
# real_estate_df['TRANSACTION DATE'] = pd.to_datetime(real_estate_df['TRANSACTION DATE'], format='%d/%m/%Y')

# # Create 'Year' and 'Month and Day' columns
# real_estate_df['Year'] = real_estate_df['TRANSACTION DATE'].dt.year
# real_estate_df['Month and Day'] = real_estate_df['TRANSACTION DATE'].dt.strftime('%m/%d')

# # Drop the original TRANSACTION DATE column
# real_estate_df = real_estate_df.drop(columns=['TRANSACTION DATE'])

# # Display the updated dataframe
# print(real_estate_df.head())































# import pandas as pd

# # Load the last60.csv file
# last60_df = pd.read_csv('./CSVs/last60.csv')

# # Inspect the first few rows to understand the structure of the data
# print(last60_df.head())

# # Calculate the average of 'Cost' for each unique 'Brand'
# avg_cost_per_brand = last60_df.groupby('Brand')['Cost'].mean().reset_index()
# print(avg_cost_per_brand)

# # Find out which 'Brand' has the highest 'QtyAvail'
# highest_qty_brand = last60_df.loc[last60_df['QtyAvail'].idxmax(), 'Brand']
# print('Brand with highest QtyAvail:', highest_qty_brand)

# # Check and print those entries where 'AcquiredDate' is '11/29/2023'
# entries_acquired_date = last60_df[last60_df['AcquiredDate'] == '11/29/2023']
# print(entries_acquired_date)


























# import pandas as pd

# # Load the TSV file
# tsv_file = './CSVs/FAL Projects NY - West SM.tsv'
# df_tsv = pd.read_csv(tsv_file, sep='\t', skiprows=9)

# # Load the Excel file
# excel_file = './CSVs/FAL Projects NY - office NY - FAL Proyectos.xlsx'
# df_excel = pd.read_excel(excel_file, sheet_name=None, skiprows=9)

# # Extract the relevant sheet from the Excel file
# df_excel_sheet = df_excel['NYC Office']

# # Display the first few rows of each dataframe to understand their structure
# print("TSV File DataFrame:")
# print(df_tsv.head())
# print(df_tsv.info(verbose=True))

# print("\nExcel File DataFrame (NYC Office):")
# print(df_excel_sheet.head())
# print(df_excel_sheet.info(verbose=True))

# Split the 'Create Date:' column in both datasets into separate 'Year', 'Month', and 'Day' columns

# # For the TSV file
# df_tsv['Year'] = pd.to_datetime(df_tsv['Create Date:'], errors='coerce').dt.year
# df_tsv['Month'] = pd.to_datetime(df_tsv['Create Date:'], errors='coerce').dt.month
# df_tsv['Day'] = pd.to_datetime(df_tsv['Create Date:'], errors='coerce').dt.day

# # For the TSV file
# df_tsv['Year'] = pd.to_datetime(df_tsv['Create Date:'], format='mixed', dayfirst=False).dt.year
# df_tsv['Month'] = pd.to_datetime(df_tsv['Create Date:'], format='mixed', dayfirst=0).dt.month
# df_tsv['Day'] = pd.to_datetime(df_tsv['Create Date:'], format='mixed', dayfirst=0).dt.day

# # THIS LINE REVEALS THAT USING THE "coerce" FLAG OMITS SIGNIFICANT INFO!
# print("TSV File DataFrame (AFTER):")
# print(df_tsv.info(verbose=True))

# # For the Excel file
# df_excel_sheet['Year'] = pd.to_datetime(df_excel_sheet['Create Date:']).dt.year
# df_excel_sheet['Month'] = pd.to_datetime(df_excel_sheet['Create Date:']).dt.month
# df_excel_sheet['Day'] = pd.to_datetime(df_excel_sheet['Create Date:']).dt.day

# # Display the first 5 rows of each updated dataframe
# print('TSV File DataFrame with Year, Month, Day columns:')
# print(df_tsv.head())

# print('\
# Excel File DataFrame (NYC Office) with Year, Month, Day columns:')
# print(df_excel_sheet.head())




'''THIS IS A COOL FUNCTION TO SPLIT A DATES COLUMN. THEN, WE DEMO THE MARKDOWN METHOD TO PRETTY PRINT'''
# from pandas.errors import ParserError

# # Function to split the date and handle potential errors
# def split_date(df, date_column):
#   try:
#     df[date_column] = pd.to_datetime(df[date_column], format='mixed')
#     df['Year'] = df[date_column].dt.year
#     df['Month'] = df[date_column].dt.month
#     df['Day'] = df[date_column].dt.day
#   except (ParserError, ValueError):
#     df['Year'] = 'NaN'
#     df['Month'] = 'NaN'
#     df['Day'] = 'NaN'
#   return df

# # Apply the function to both DataFrames
# df_nyc_office = split_date(df_excel_sheet, 'Create Date:')
# df_west_sm = split_date(df_tsv, 'Create Date:')

# # Display the first 5 rows of each DataFrame
# print("First 5 rows of NYC Office data with split date:")
# print(df_nyc_office[['Create Date:', 'Year', 'Month', 'Day']].head().to_markdown(index=False, numalign="left", stralign="left"))

# print("\nFirst 5 rows of West SM data with split date:")
# print(df_west_sm[['Create Date:', 'Year', 'Month', 'Day']].head().to_markdown(index=False, numalign="left", stralign="left"))

































# # Load the CSV file and calculate the number of times money was returned to a customer
# import pandas as pd

# # Load the CSV file
# df_sales = pd.read_csv('./CSVs/Ventas_Julio-Octubre-wines.xlsxcsv-Julio-Octubre.csv')

# # Display the first few rows to understand the structure of the data
# print(df_sales.head())
# print(df_sales.info(verbose=1))

# # Calculate the number of returns
# returns_count = df_sales[df_sales['\u00cdtem - Impte. Neto mon. Local'] < 0].shape[0]

# # Calculate total spent and total returned for each customer
# # total_spent = df_sales.groupby('Cliente - Pa\u00eds - C\u00f3d.')['\u00cdtem - Impte. Neto mon. Local'].sum()

# total_spent = df_sales[df_sales['\u00cdtem - Impte. Neto mon. Local'] > 0].groupby('Cliente - Pa\u00eds - C\u00f3d.')['\u00cdtem - Impte. Neto mon. Local'].sum()

# total_returned = df_sales[df_sales['\u00cdtem - Impte. Neto mon. Local'] < 0].groupby('Cliente - Pa\u00eds - C\u00f3d.')['\u00cdtem - Impte. Neto mon. Local'].sum().abs()

# # Calculate the relationship score for each customer
# relationship_score = (total_spent - total_returned) / total_spent * 100

# # Combine the results into a single DataFrame
# customer_relationship = pd.DataFrame({'Total Spent': total_spent, 'Total Returned': total_returned, 'Relationship Score (%)': relationship_score})

# # Display the number of returns and the first few rows of the customer relationship DataFrame
# print('Number of returns:', returns_count)
# # print(customer_relationship.head())


# # Sort the results in descending order by the relationship score
# relationship_df = customer_relationship.sort_values(by='Relationship Score (%)', ascending=False)

# # Print the results
# print(relationship_df.to_markdown(index=False, numalign="left", stralign="left"))


























'''
    MULTI-SHEET EXCEL
'''
# import pandas as pd

# # Display the sheet names first to identify the correct sheet
# excel_file = pd.ExcelFile('./CSVs/population_and_age_1.xlsx')
# print(excel_file.sheet_names)

# # Load the third sheet
# df_south_america = pd.read_excel('./CSVs/population_and_age_1.xlsx', sheet_name=2)

# # Display the first few rows 
# print(df_south_america.head())
# print(df_south_america.info(verbose=1))

# # Calculate the average age and population for South American countries in the third sheet
# average_age_south_america = df_south_america['Average Age'].mean()
# average_population_south_america = df_south_america['Population'].mean()

# # Display the results
# print('Average Age in South America:', average_age_south_america)
# print('Average Population in South America:', average_population_south_america)











































'''
    HEATMAP USING AN EXCEL DATASET WITH MULTIPLE SHEETS, WHICH WE COMBINE HERE
'''
# import pandas as pd
# import seaborn as sns
# import matplotlib.pyplot as plt

# # Load the first sheet
# df_month1 = pd.read_excel('./CSVs/SOLDFOOD2023 - Winter.xlsx', sheet_name=0, header=3)
# df_month1['Month'] = 'Month1'

# # Load the second sheet
# df_month2 = pd.read_excel('./CSVs/SOLDFOOD2023 - Winter.xlsx', sheet_name=1, header=3)
# df_month2['Month'] = 'Month2'

# # Combine both sheets into a single dataframe
# df_combined = pd.concat([df_month1, df_month2])

# # Display the first few rows of the combined dataframe to confirm the changes
# print(df_combined.head())
# print(df_combined.info(verbose=1))

# # Create a pivot table to prepare the data for the heatmap
# pivot_table = df_combined.pivot_table(index='GROUP', columns='Month', values='QUANTITY', aggfunc='sum')

# # Create the heatmap
# plt.figure(figsize=(10, 6))
# sns.heatmap(pivot_table, annot=True, fmt='g', cmap='viridis')
# plt.title('Quantity Sold by Group and Month')
# plt.xlabel('Month')
# plt.ylabel('Group')
# plt.tight_layout()
# plt.show()























'''
    CREATE AGE GROUPS FROM AN AGE COLUMN, I.E., PARSE THE AGE COLUMN INTO SPECIFIC SIZED BINS.
'''
# import pandas as pd

# # Load the Excel file
# df_life_ins = pd.read_excel('./CSVs/LIFE INS ISSUE AGE AUDIT.xlsx')

# print(df_life_ins.head())
# print(df_life_ins.info(verbose=1))

# # Create age groups in increments of 10 years
# bins = list(range(0, df_life_ins['Issue Age'].max() + 10, 10))

# labels = [str(i) + '-' + str(i + 9) for i in bins[:-1]]

# df_life_ins['Age Group'] = pd.cut(
#     df_life_ins['Issue Age'], 
#     bins=bins, 
#     labels=labels, 
#     right=False
# )

# # Calculate the average premium for each age group
# average_premium_by_age_group = df_life_ins.groupby('Age Group', observed=False)['Mode Premium'].mean().reset_index()

# # Display the result
# print(average_premium_by_age_group)




'''2nd APPROACH'''
# age_bins = list(range(0, 101, 10))
# age_labels = [f"{i}-{i+9}" for i in range(0, 100, 10)]

# # Categorize each policy into an age group
# df_life_ins['Age Group'] = pd.cut(
#     df_life_ins['Issue Age'], 
#     bins=age_bins, 
#     labels=age_labels, 
#     right=False
# )

# # Calculate the average 'Mode Premium' for each age group
# average_premium_by_age_group = df_life_ins.groupby('Age Group')['Mode Premium'].mean().reset_index()



























'''
    HERE I HAD TO PREPROCESS THE GRAND TOTAL COL SUCH THAT BOTH POSITIVE AND NEGATIVE (REFUNDS) VALUES ARE ACCOUNTED FOR AND HANDLED ELEGANTLY. I NEEDED TO FIND THE TRANSACTION WITH THE LOWEST AMOUNT AND AT FIRST I SIMPLY GOT THE MIN, WHICH TURNED OUT TO BE A NEGATIVE, BUT THEN I REALIZED WHAT A NEGATIVE INT MEANS IN THE CONTEXT OF SALES, WHICH IS A REFUND. THEREFORE, I HAD TO UPDATE THE CODE TO IGNORE THESE.
'''
# import pandas as pd
# import re

# header_row = 9
# df_fal_projects = pd.read_excel('./CSVs/FAL Projects NY - office NY - FAL Proyectos.xlsx', header=header_row)

# # Display the first few rows to confirm the changes
# print(df_fal_projects.head())
# print(df_fal_projects.info(verbose=1))

# # Clean up the col names
# df_fal_projects.columns = ['PO/ Order #', 'Create Date', 'Vendor Name', 'Card #', 'Order Sub Total', 'Shipping/Handling', 'Delivery', 'Tax', 'Grand Total', 'Trade Adjustment', 'Purchased By', 'Item', 'Tracking Information', 'Notes', 'Arrived', 'Return/Change Order', 'Change Comments']

# # Preprocess the 'Grand Total' column to handle negative values represented by parentheses
# def preprocess_grand_total(value):
#     if isinstance(value, str):
#         # Remove dollar sign and commas
#         value = value.replace('$', '').replace(',', '')
#         # Check for negative values represented by parentheses
#         if re.match(r'\(.*\)', value):
#             value = '-' + value.strip('()')
#     return float(value)

# # Apply the preprocessing function to the 'Grand Total' column
# df_fal_projects['Grand Total'] = df_fal_projects['Grand Total'].apply(preprocess_grand_total)

# # Filter out rows with missing vendor or date
# filtered_df = df_fal_projects.dropna(subset=['Vendor Name', 'Create Date'])

# print('AFTER')
# print(df_fal_projects.info(verbose=1))

# # Filter out negative values from the 'Grand Total' column
# positive_df = filtered_df[filtered_df['Grand Total'] > 0]

# # Find the row with the smallest 'Grand Total' in the filtered dataframe
# min_transaction_filtered = positive_df.loc[positive_df['Grand Total'].idxmin()]

# # Extract relevant details
# vendor_filtered = min_transaction_filtered['Vendor Name']
# date_filtered = min_transaction_filtered['Create Date']
# amount_filtered = min_transaction_filtered['Grand Total']

# # Display the result
# print('Vendor:', vendor_filtered)
# print('Date:', date_filtered)
# print('Amount:', amount_filtered)

























'''
    HERE IS A COOL WAY OF GENERATING SUMMARIES AND THERES ALSO A FOR-LOOP THAT TRAVERSES ALL THE COLUMNS AND CALCULATES THE UNIQUE VALUES WITHIN EACH.
'''
# import pandas as pd

# # Read the CSV file into a DataFrame
# df = pd.read_excel('./CSVs/phone_buying_preference1.xlsx')
# print(df.head())
# print(df.info())

# # Get a summary of the data
# summary = {
#     'Categories': list(df.columns),
#     'Total Entries': len(df),
#     'Missing Values': df.isnull().sum(),
#     'Strange Values': df.apply(lambda x: x.str.strip().isin(['', 'NA', 'N/A', 'none', 'None']).sum() if x.dtype == "object" else 0)
# }

# summary_df = pd.DataFrame(summary)
# print(summary_df)



'''CREATES A COOL MARKDOWN TABLE IN THE TERMINAL'''
# # Get all columns that are of type Object
# object_columns = df.select_dtypes(include=['object']).columns

# # Iterate through each object column
# for col in object_columns:
#   # Calculate the frequency of each unique value
#   value_counts = df[col].value_counts(dropna=False)

#   print(f'\nUnique values and their frequencies for column: {col}')
#   if (len(value_counts) > 20):
#     # Sample 20 of them if there are too many unique values
#     print(value_counts.sample(20).to_markdown(numalign="left", stralign="left"))
#   else:
#     # Otherwise print all
#     print(value_counts.to_markdown(numalign="left", stralign="left"))

































# import pandas as pd

# # Load the CSV file
# last60_df = pd.read_csv('./CSVs/last60.csv')

# # Count the records with missing values in the 'MapPrice' column
# missing_mapprice_count = last60_df['MapPrice'].isnull().sum()

# # Get the part numbers of the records with missing 'MapPrice'
# missing_mapprice_part_numbers = last60_df[last60_df['MapPrice'].isnull()]['PartNumber']

# # Display the count and several part numbers
# print('Count of records with missing MapPrice:', missing_mapprice_count)
# print('First 5 part numbers with missing MapPrice:\n', missing_mapprice_part_numbers.head(n=25))







# # Get all unique rows where `MapPrice` has null values
# null_map_price_rows = last60_df[last60_df['MapPrice'].isnull()].drop_duplicates()

# if (len(null_map_price_rows) > 20):
#   # Sample 20 of them if there are too many unique rows
#   print(null_map_price_rows.sample(20))
# else:
#   # Otherwise print all unique rows
#   print(null_map_price_rows)


























'''
    IS THE 'equal' FLAG PASSED INTO THE AXIS METHOD REALLY NECISSARY HERE?
'''

# import pandas as pd
# import matplotlib.pyplot as plt

# df = pd.read_csv('./CSVs/Loan_Default.csv')

# # Filter the dataset
# filtered_df = df[(df['loan_type'] == 'type1') & 
#                  (df['loan_purpose'] == 'p1') & 
#                  (df['Region'] == 'south')]

# # Group by gender and count the occurrences
# gender_counts = filtered_df['Gender'].value_counts()

# # Create a pie chart
# plt.figure(figsize=(12, 10))
# plt.pie(gender_counts, 
#         labels=gender_counts.index, 
#         autopct='%1.1f%%', 
#         startangle=140)
# plt.title('Percentage of Each Gender Type with Loan Type1, Purpose P1, in South Region')
# plt.axis('equal')  # Circular pie chart
# plt.show()































# import matplotlib.pyplot as plt
# import pandas as pd

# # Load the data
# vendor_df = pd.read_csv('./CSVs/VendorDownload_4_1f4f8b55-1614-476d-8f6d-7ece6d7c793c.xlsx - Líneas de artículos.tsv', delimiter='\t')
# print(vendor_df.head())
# print(vendor_df.info(verbose=1))

# # Convert the date columns to datetime format
# vendor_df['PO Date'] = pd.to_datetime(vendor_df['PO Date'])
# vendor_df['Delivery beguin date'] = pd.to_datetime(vendor_df['Delivery beguin date'])
# vendor_df['End Delivery date'] = pd.to_datetime(vendor_df['End Delivery date'])
# vendor_df['Max Delivery Date'] = pd.to_datetime(vendor_df['Max Delivery Date'])

# # Clean the 'Discount' column
# vendor_df['Discount'] = pd.to_numeric(vendor_df['Discount'].str.replace('%', ''))

# print("AFTER")
# print(vendor_df.info(verbose=1))

# # Print unique 'PO Date' values
# print('Unique PO Dates:', vendor_df['PO Date'].unique())
# print('Unique beguin dates:', vendor_df['Delivery beguin date'].unique())
# print('Unique end Dates:', vendor_df['End Delivery date'].unique())
# print('Unique max Dates:', vendor_df['Max Delivery Date'].unique())

# # Plot the 'Discount' values
# plt.figure(figsize=(10, 6))
# plt.plot(vendor_df['PO Date'], vendor_df['Discount'], marker='o')
# plt.title('Discount Values')
# plt.xlabel('PO Date')
# plt.ylabel('Discount')
# plt.grid(True)
# plt.xticks(rotation=45)
# plt.tight_layout()
# plt.show()


























'''
    HERE, I CAREFULLY CLEAN THE TWO DATASETS AND CALCULATE THEIR RESPECTIVE TAX PERCENTAGE USING MULTIPLE COLUMNS
'''
# # Load the FAL Projects NY - West SM.tsv file and display the column names to identify the correct columns for the tax calculation
# import pandas as pd

# # Load the TSV file, skipping the first 9 rows to get to the header
# fal_west_df = pd.read_csv('./CSVs/FAL Projects NY - West SM.tsv', sep='\t', skiprows=9)

# # Load the Excel file, skipping the first 9 rows to get to the header
# fal_office_df = pd.read_excel('./CSVs/FAL Projects NY - office NY - FAL Proyectos.xlsx', skiprows=9)

# # print(fal_west_df.head())
# # print(fal_west_df.info(verbose=1))
# # print(fal_west_df.columns)
# # print(fal_west_df['Tax:'].unique())
# # print(fal_west_df['Order Sub Total:'].unique())
# # print(fal_west_df['Shipping/ Handling:'].unique())

# print(fal_office_df.head())
# print(fal_office_df.columns)
# print(fal_office_df.info(verbose=1))

# # Correct the column names and calculate the tax percentage
# fal_office_df.rename(columns={'Shipping/\nHandling:': 'Shipping and Handling Fees', 'Tax:': 'Tax'}, inplace=True)
# print(fal_office_df['Tax'].unique())
# print(fal_office_df['Order Sub Total:'].unique())
# print(fal_office_df['Shipping and Handling Fees'].unique())

# # Convert the 'Tax:', 'Order Sub Total:', and 'Shipping/ Handling:' columns to numeric types
# fal_west_df['Tax:'] = pd.to_numeric(fal_west_df['Tax:'].str.replace('$', '').str.replace(',', ''), errors='coerce')
# fal_west_df['Order Sub Total:'] = pd.to_numeric(fal_west_df['Order Sub Total:'].str.replace('$', '').str.replace(',', '').str.replace('€', ''), errors='coerce')
# fal_west_df['Shipping/ Handling:'] = pd.to_numeric(fal_west_df['Shipping/ Handling:'].str.replace('$', '').str.replace(',', ''), errors='coerce')

# # No need to convert the fal_office relevant columns to numeric types, they already are numeric.

# # # Check if any information was omittted as a result of using the coerce option
# # print("AFTER")
# # print(fal_west_df.info(verbose=1))
# # print(fal_west_df['Tax:'].unique())
# # print(fal_west_df['Order Sub Total:'].unique())
# # print(fal_west_df['Shipping/ Handling:'].unique())

# # Calculate the tax as a percentage of the Order Sub Total plus shipping and handling fees
# fal_west_df['Total Before Tax'] = fal_west_df['Order Sub Total:'] + fal_west_df['Shipping/ Handling:']
# fal_west_df['Tax Percentage'] = (fal_west_df['Tax:'] / fal_west_df['Total Before Tax']) * 100

# # Calculate the tax as a percentage of the Order Sub Total plus shipping and handling fees
# fal_office_df['Total Before Tax'] = fal_office_df['Order Sub Total:'] + fal_office_df['Shipping and Handling Fees']
# fal_office_df['Tax Percentage'] = (fal_office_df['Tax'] / fal_office_df['Total Before Tax']) * 100


# # Display the first few rows to verify the new column
# print(fal_west_df.head())
# print(fal_west_df['Tax Percentage'].unique())

# # Display the first few rows to verify the new column
# print(fal_office_df.head())
# print(fal_office_df['Tax Percentage'].unique())































# import pandas as pd

# vendor_df = pd.read_csv('./CSVs/VendorDownload_4_1f4f8b55-1614-476d-8f6d-7ece6d7c793c.xlsx - Líneas de artículos.tsv', sep='\t')

# print(vendor_df.head())
# print(vendor_df.info(verbose=1))

# # Display the rows where 'Qty Ordered' is non-zero and 'Qty Confirmed' is zero
# non_zero_ordered_zero_confirmed = vendor_df[(vendor_df['Qty Ordered'] != 0) & 
#                                             (vendor_df['Qty Confirmed'] == 0)]
# print(non_zero_ordered_zero_confirmed)

# # Get unique values from `Availability/Comment`
# unique_availability_comments = non_zero_ordered_zero_confirmed['Availability/Comment'].unique()
# print(unique_availability_comments)

# # Create the subset 's1' where 'Qty Ordered' is non-zero and 'Qty Confirmed' is zero
# s1 = vendor_df[(vendor_df['Qty Ordered'] != 0) & (vendor_df['Qty Confirmed'] == 0)]

# # Find rows that are not in 's1' but have 'R2...' or 'CP...' in the 'Availability/Comment' column
# other_rows = vendor_df[~vendor_df.index.isin(s1.index) & vendor_df['Availability/Comment'].str.contains('R2|CP', na=False)]

# # Display the rows
# print(other_rows)





























# import pandas as pd

# # Load the cars_raw.csv file
# cars_df = pd.read_csv('./CSVs/cars_raw.csv')

# print(cars_df.head())
# print(cars_df.info(verbose=1))
# print(cars_df['FuelType'].unique())

# # Filter the data to include only electric cars
# electric_cars_df = cars_df[cars_df['FuelType'].str.contains('Electric', case=False, na=False)]
# print("Num electric cars:", electric_cars_df.shape[0])

# # Filter the data on `FuelType` column
# electric_cars_df2 = cars_df[cars_df['FuelType'] == 'Electric'].copy()
# print("Num electric cars2:", electric_cars_df2.shape[0])

# # Save the filtered data to a new CSV file
# electric_cars_df.to_csv('./OutCSVs/electric_cars.csv', index=False)
# print('Electric cars data saved to electric_cars.csv')




























'''
    HERE IS AN INTERESTING SCENARIO WHERE A NEWLY CREATED COLUMN IS PLACED AT A VERY SPECIFIC LOCATION WITHIN A DATASET.
'''
# import pandas as pd

# # Load the TSV file
# retail_store_df = pd.read_csv('./CSVs/Retail Store Performance and Capacity Metrics - EXO2E Crypto - cccvvv.tsv', sep='\t')

# print(retail_store_df.head())
# print(retail_store_df.info(verbose=1))

# # Determine the position to insert the new column
# position = retail_store_df.columns.get_loc('Category')

# # Insert a new column after 'Installed Capacity 30' and initialize with zeros
# retail_store_df.insert(position, 'Installed Capacity Difference', 0)

# # Create the new requested column
# retail_store_df['Installed Capacity Difference'] = retail_store_df['Installed Capacity 30'] - retail_store_df['Installed Capacity 20']

# # Save the new table to a CSV file
# retail_store_df.to_csv('./OutCSVs/retail_store_performance_installed_difference.csv', index=False)
# print('Table with Installed Capacity Difference saved to retail_store_performance_installed_difference.csv')































'''
    HERE IS AN EXAMPLE OF ME WAISTING TIME BY NOT EXPLORING THE DATASET FIRST. I TRY TO FILTER FOR DATES WITHIN OCTOBOR AND NOVEMBER BUT I DO  NOT REALIZE THAT THE ENTIRE DATASET IS WITHIN THESE DATES SO FILTERING IS REDUNDANT!
'''

# import pandas as pd

# # Load the dataset
# vas_df = pd.read_csv('./CSVs/DATA_ECOM_VAS_v1-.xlsx-Grossreport.csv')
# print(vas_df.head())
# print(vas_df.info(verbose=1))

# Filter the data for October and November
# vas_df['Creation Date'] = pd.to_datetime(vas_df['Creation Date'])
# print("AFTER")
# print(vas_df.info(verbose=1))
# oct_nov_df = vas_df[(vas_df['Creation Date'].dt.month == 10) | (vas_df['Creation Date'].dt.month == 11)]

# # Group by product and sum the units sold
# best_selling = oct_nov_df.groupby('SKU Name')['Quantity_SKU'].sum().reset_index()

# # Find the best-selling product
# best_selling_product = best_selling.loc[best_selling['Quantity_SKU'].idxmax()]

# # Display the best-selling product and the number of units sold
# print('Best-selling product of October and November:', best_selling_product['SKU Name'])
# print('Units sold:', best_selling_product['Quantity_SKU'])


# # Group by product and sum the units sold
# best_selling = vas_df.groupby('SKU Name')['Quantity_SKU'].sum().reset_index()

# # Find the best-selling product
# best_selling_product = best_selling.loc[best_selling['Quantity_SKU'].idxmax()]

# # Display the best-selling product and the number of units sold
# print('Best-selling product of October and November:', best_selling_product['SKU Name'])
# print('Units sold:', best_selling_product['Quantity_SKU'])





























# import pandas as pd

# # Load the Excel file
# file_path = './CSVs/population_and_age.xlsx'
# xlsx = pd.ExcelFile(file_path)

# # Get the names of all sheets in the Excel file
# sheet_names = xlsx.sheet_names
# print(sheet_names)

# # Load the South America sheet
# south_america_df = pd.read_excel(file_path, sheet_name=sheet_names[2])

# # Display the first few rows to understand the structure
# print(south_america_df.head())
# print(south_america_df.info(verbose=1))

# # Calculate the average age and population
# average_age = south_america_df['Average Age'].mean()
# total_population = south_america_df['Population'].mean()

# # Display the results
# print('Average Age across all countries in South America:', average_age)
# print('Total Population across all countries in South America:', total_population)























































# import pandas as pd

# # Load the Excel file
# file_path = './CSVs/Gycology_Service_2020.xlsx'
# df = pd.read_excel(file_path, skiprows=6)

# # Display the first few rows of the DataFrame
# print('First few rows of the DataFrame:')
# print(df.head())

# # Display the summary of the DataFrame
# print('\
# Summary of the DataFrame:')
# print(df.info())

# # Check for missing values
# print('\
# Missing values in the DataFrame:')
# print(df.isnull().sum())

# # Describe the DataFrame to get an overview of the data
# print('\
# Description of the DataFrame:')
# print(df.describe())

# # Check for possible outliers using IQR method
# Q1 = df.quantile(0.25)
# Q3 = df.quantile(0.75)
# IQR = Q3 - Q1
# outliers = ((df < (Q1 - 1.5 * IQR)) | (df > (Q3 + 1.5 * IQR))).sum()
# print('\
# Possible outliers in the DataFrame:')
# print(outliers)






























'''
    HOW WOULD I FIX THE PLOT BELOW? IT IS A PIE CHART, BUT THERE ARE MANY SLICES WHICH CAUSE LABEL OVERLAP. HOW CAN I INCREASE READABILITY? 
'''

# import pandas as pd
# import matplotlib.pyplot as plt

# # Load the CSV file
# file_path = './CSVs/MX_Pharmacy_inventory2022-dataset-Dataset.csv'
# df = pd.read_csv(file_path, encoding='MacRoman')

# print(df.head())
# print(df.info(verbose=1))
# print(df['Medication Description'].unique())

# # Group by 'Lab ' and count the number of products
# lab_product_counts = df['Lab '].value_counts()

# # Plot a pie chart
# plt.figure(figsize=(10, 8))
# plt.pie(lab_product_counts, 
#         labels=lab_product_counts.index, 
#         autopct='%1.1f%%', 
#         startangle=140)
# plt.title('Distribution of Products by Lab')
# plt.axis('equal')  
# plt.show()

'''
    THE PLOTS BELOW INCREASE READABILITY BY CREATING A LEGEND. APPLY THE SAME TECHNIQUE TO THE PLOT ABOVE TO SEE IF IT WORKS!
'''

# import altair as alt

# products_by_lab = df.groupby('Lab ')['Medication Description'].nunique()

# # Sort in descending order of counts
# products_by_lab = products_by_lab.sort_values(ascending=False)

# # Print the result
# print("Number of products by Lab:\n")
# print(products_by_lab.to_markdown(numalign="left", stralign="left"))

# # Create a DataFrame for plotting
# df_plot = products_by_lab.reset_index().rename(columns={'Lab ': 'Lab', 'Medication Description': 'Num_Products'})

# # Create a pie chart of the number of products by Lab
# chart1 = alt.Chart(df_plot).mark_arc().encode(
#     theta=alt.Theta(field="Num_Products", type="quantitative"),
#     color=alt.Color(field="Lab", type="nominal"),
#     tooltip = ["Lab", "Num_Products"]
# ).properties(
#     title = "Distribution of Products by Lab"
# ).interactive()

# # Save the chart in json file
# chart1.save('./OutPlots/products_by_lab_pie_chart.html')


'''
    ALSO, APPLY THE LEGEND TO THE TOP 10 PLOT TO INCREASE ITS READABILITY!
'''


# # Keep only top 10 labs and create an 'Others' category
# top_10_labs = products_by_lab.head(10)
# other_labs = pd.Series({'Others': products_by_lab.iloc[10:].sum()})
# combined_data = pd.concat([top_10_labs, other_labs])

# # Create a DataFrame for plotting
# df_plot = combined_data.reset_index().rename(columns={'index': 'Lab', 0: 'Num_Products'})

# # Calculate percentages for each lab
# total_products = df_plot['Num_Products'].sum()
# df_plot['Percentage'] = ((df_plot['Num_Products'] / total_products) * 100).round(2)

# # Create a pie chart of the number of products by Lab for the top 10 labs and 'Others' category
# base = alt.Chart(df_plot).encode(
#     theta=alt.Theta("Num_Products:Q"),
#     color=alt.Color("Lab:N", legend=None),
#     tooltip = ["Lab", "Num_Products", "Percentage"]
# )

# pie = base.mark_arc(outerRadius=120, innerRadius=0)
# text = base.mark_text(radius=125, fill="black").encode(alt.Text(field="Percentage", type="quantitative", format=".1f"))
# text_label = base.mark_text(radius=140, fill="black").encode(alt.Text(field="Lab", type="nominal"))

# # Combine pie chart, text, and legend
# chart2 = alt.layer(pie, text, text_label, data=df_plot).resolve_scale(theta="independent").properties(
#     title='Distribution of Products by Top 10 Labs and Others'
# ).interactive()

# # Save the chart in json file
# chart2.save('./OutPlots/top_10_labs_and_others_pie_chart.html')






























'''
    WHATS THE DIFFERENCE BETWEEN USING THE .pivot_table() AND NOT USING IT WHEN INVOKING THE .heatmap() or .corr()?
'''


# import pandas as pd
# import seaborn as sns
# import matplotlib.pyplot as plt

# # Load the CSV file
# file_path = './CSVs/projects.csv'
# df = pd.read_csv(file_path)
# print(df.head())
# print(df.info(verbose=1))

# # There is 3, 20+, and 9 unique entries, respectively
# print(df['DISCOUNT'].unique())
# print(df['TOTAL'].unique())
# print(df['PROJECT_TYPE'].unique())

# # Function to clean and convert TOTAL column
# def clean_total(total):
#     # Remove quotes
#     total = total.replace('"', '')
#     # Replace comma with period
#     total = total.replace(',', '.')
#     # Convert to float
#     return float(total)

# # Apply the cleaning function to the TOTAL column
# df['TOTAL'] = df['TOTAL'].apply(clean_total)

# # To see if it omits entries
# df['TOTAL'] = pd.to_numeric(df['TOTAL'], errors='coerce')

# print("AFTER")
# print(df.info(verbose=1))

# # Create a pivot table for the heatmap
# pivot_table = df.pivot_table(index='PROJECT_TYPE', 
#                              values='TOTAL', 
#                              columns='DISCOUNT',
#                              aggfunc='sum')

# # Plot the heatmap
# plt.figure(figsize=(12, 8))
# sns.heatmap(pivot_table, annot=True, cmap='coolwarm', fmt='.2f')
# plt.title('Heatmap of PROJECT_TYPE, DISCOUNT, and TOTAL')
# plt.show()






























'''
    COHORT ANALYSIS ON A MULTI-SHEET EXCEL FILE. I LOAD EACH SHEET INDIVIDUALLY AND THEN CREATE A NEW COLOUMN TO STORE THEIR RESPECTIVE SHEET AND USE THAT TO CREATE COHORTS BEFORE FINALLY COMBINING ALL THE SHEETS. 
'''


# import pandas as pd
# import matplotlib.pyplot as plt
# import seaborn as sns

# # Load the Excel file
# file_path = './CSVs/SOLDFOOD2023 - Fall.xlsx'

# # Read all sheets from the Excel file
# xls = pd.ExcelFile(file_path)
# sheet_names = xls.sheet_names
# print(sheet_names)

# # Read the first sheet into a DataFrame
# sep = pd.read_excel(file_path, sheet_name=sheet_names[0], skiprows=3)
# october = pd.read_excel(file_path, sheet_name=sheet_names[1], skiprows=3)
# nov = pd.read_excel(file_path, sheet_name=sheet_names[2], skiprows=3)

# # Add a cohort identifier
# sep['COHORT'] = '2023-09'
# october['COHORT'] = '2023-10'
# nov['COHORT'] = '2023-11'

# # Combine DataFrames
# df_combined = pd.concat([sep, october, nov])
# df_combined['COHORT'] = pd.to_datetime(df_combined['COHORT'])

# # Display the first few rows of the combined DataFrame
# print(df_combined.head())
# print(df_combined.info())

# # Aggregate the data to calculate total revenue for each product within each cohort
# cohort_analysis = df_combined.groupby(['COHORT', 'DESCRIPTION']).agg({'TOTAL SALE': 'sum'}).reset_index()

# # # Display the cohort analysis DataFrame
# # print(cohort_analysis)

# # Create a pivot table to visualize the data
# pivot_table = cohort_analysis.pivot(index='DESCRIPTION', 
#                                     columns='COHORT', 
#                                     values='TOTAL SALE')

# # Display the pivot table
# print(pivot_table)

# # Plot the heatmap
# plt.figure(figsize=(12, 8))
# sns.heatmap(pivot_table, 
#             annot=True, 
#             cmap='coolwarm', 
#             fmt='.0f')
# plt.title('Revenue by Product and Cohort (Month)')
# plt.xlabel('Cohort (Month)')
# plt.ylabel('Product ID')
# plt.show()




































'''
    HERE I HAVE A FUNCTION THAT CYCLES THRU COLUMNS AND CREATES PLOTS OUT OF THE CORRESPONDING COL.
'''

# import pandas as pd
# import altair as alt

# # Load the Excel file
# file_path = './CSVs/Hospital_Survey_Data_Alcohol_Drug_Abuse.xlsx'
# df = pd.read_excel(file_path, skiprows=1)

# print(df.head())
# print(df.info())

# # Check for missing values
# missing_values = df.isnull().sum()
# print('Missing values in each column:')
# print(missing_values)

# # Check for data type diversity
# data_types = df.dtypes
# print('Data types of each column:')
# print(data_types)

# # Check for apparent outliers using describe
# describe_df = df.describe()
# print('Statistical summary of the DataFrame:')
# print(describe_df)


# # Filter the dataframe to only include the columns `Total Discharges`, `Average Covered Charges ($)`, `Average Total Payments ($)`, `Average Medicare Payments ($)`, and `Hospital Rating`
# columns_to_keep = ['Total Discharges', 'Average Covered Charges ($)', 'Average Total Payments ($)', 'Average Medicare Payments ($)', 'Hospital Rating']
# filtered_df = df[columns_to_keep]

# # Display descriptive statistics for the filtered dataframe
# print("Descriptive Statistics of the Filtered Data:\n")
# print(filtered_df.describe().to_markdown(numalign="left", stralign="left"))

# # Create histograms for each of the columns in the filtered dataframe
# for col in filtered_df.columns:
#   chart = alt.Chart(filtered_df).mark_bar().encode(
#       x=alt.X(col, bin=True),
#       y=alt.Y('count()', title='Count of Hospitals'),
#       tooltip=[col, 'count()']
#   ).properties(
#       title=f'Histogram of {col}'
#   ).interactive()

#   chart.save(f'./OutPlots/{col}_histogram.html')















































'''
    EXTRACT SPECIFIC COLS AND THEN USE THREE MONEY COLS TO CREATE A TOTAL INCOME COL THAT IS ESSENTIALLY THE SUM OF THREE SUMMATIONS.
'''

# import pandas as pd
# import matplotlib.pyplot as plt
# import seaborn as sns

# file_path_alcohol_drug = './CSVs/Hospital_Survey_Data_Alcohol_Drug_Abuse.xlsx'
# df_alcohol_drug = pd.read_excel(file_path_alcohol_drug, skiprows=1)
# file_path_septicemia = './CSVs/Hospital_Survey_Data_Speticemia.csv'
# df_septicemia = pd.read_csv(file_path_septicemia, skiprows=1)
# print(df_alcohol_drug.head())
# print(df_septicemia.head())
# print(df_alcohol_drug.info(verbose=1))
# print(df_septicemia.info(verbose=1))

# # Combine the relevant columns from both datasets
# combined_df = pd.concat([
#     df_alcohol_drug[['Provider Zip Code', 'Average Covered Charges ($)', 'Average Total Payments ($)', 'Average Medicare Payments ($)']],
#     df_septicemia[['Provider Zip Code', 'Average Covered Charges ($)', 'Average Total Payments ($)', 'Average Medicare Payments ($)']]
# ])

# print("AFTER")
# print(combined_df.info(verbose=1))
# print(combined_df['Provider Zip Code'].unique())

# # Group by 'Provider Zip Code' and sum the specified columns
# grouped_df = combined_df.groupby('Provider Zip Code').agg({
#     'Average Covered Charges ($)': 'sum',
#     'Average Total Payments ($)': 'sum',
#     'Average Medicare Payments ($)': 'sum'
# }).reset_index()

# # Combine the three summations to get a total income value
# grouped_df['Total Income'] = (
#     grouped_df['Average Covered Charges ($)'] +
#     grouped_df['Average Total Payments ($)'] +
#     grouped_df['Average Medicare Payments ($)']
# )

# grouped_df = grouped_df.sort_values(by='Total Income', ascending=0)
# print(grouped_df.head(n=20))

# # Find the best-selling product
# largest_income = grouped_df.loc[grouped_df['Total Income'].idxmax()]
# print('Largest income:', largest_income)


# # Plot the total income value for each zip code
# plt.figure(figsize=(10, 6))
# plt.bar(grouped_df['Provider Zip Code'], grouped_df['Total Income'], color='skyblue')
# plt.xlabel('Provider Zip Code')
# plt.ylabel('Total Income ($)')
# plt.title('Total Income by Provider Zip Code')
# plt.xticks(rotation=45)
# plt.tight_layout()
# plt.show()



# # Group by 'Provider Zip Code' and sum the payment columns
# income_by_zip = combined_df.groupby('Provider Zip Code').sum()

# # Sort by 'Average Total Payments ($)' to find the zip codes with the highest income
# income_by_zip_sorted = income_by_zip.sort_values(by='Average Total Payments ($)', ascending=False)

# # Display the top zip codes by income
# print('Top zip codes by income:')
# print(income_by_zip_sorted.head(10))

# # Plot the top zip codes by income
# plt.figure(figsize=(12, 8))
# sns.barplot(x=income_by_zip_sorted['Average Total Payments ($)'].head(10), y=income_by_zip_sorted.index[:10], palette='viridis')
# plt.title('Top Zip Codes by Income')
# plt.xlabel('Total Payments ($)')
# plt.ylabel('Zip Code')
# plt.show()




'''
    2ND APPROACH: USE LOOPS TO CLEAN AND PREPROCESS THE DATA THEN PRINT IT IN HUMAN FRIENDLY WAY WITH THE MARKDOWN METHOD.
'''

# # Convert column names to lowercase and replace whitespace with underscores
# df_alcohol_drug.columns = df_alcohol_drug.columns.str.lower().str.replace(' ', '_')
# df_septicemia.columns = df_septicemia.columns.str.lower().str.replace(' ', '_')

# # Remove ',' from the columns `average_covered_charges`, `average_total_payments`, and `average_medicare_payments`
# for col in ['average_covered_charges_($)', 'average_total_payments_($)', 'average_medicare_payments_($)']:
#     df_alcohol_drug[col] = df_alcohol_drug[col].astype(str).str.replace(',', '', regex=False)
#     df_septicemia[col] = df_septicemia[col].astype(str).str.replace(',', '', regex=False)

# # Convert the columns `average_covered_charges`, `average_total_payments`, and `average_medicare_payments` to numeric
# for col in ['average_covered_charges_($)', 'average_total_payments_($)', 'average_medicare_payments_($)']:
#     df_alcohol_drug[col] = pd.to_numeric(df_alcohol_drug[col], errors='coerce')
#     df_septicemia[col] = pd.to_numeric(df_septicemia[col], errors='coerce')

# print("AFTER")
# print(df_alcohol_drug.head())
# print(df_septicemia.head())

# print(df_alcohol_drug.info(verbose=1))
# print(df_septicemia.info(verbose=1))

# # Replace missing values in specified columns with 0
# for col in ['average_covered_charges_($)', 'average_total_payments_($)', 'average_medicare_payments_($)']:
#     df_alcohol_drug[col] = df_alcohol_drug[col].fillna(0)
#     df_septicemia[col] = df_septicemia[col].fillna(0)

# # Filter the DataFrames to only contain the columns `provider_zip_code`, `average_covered_charges`, `average_total_payments`, and `average_medicare_payments`
# df_alcohol_drug_abuse = df_alcohol_drug[['provider_zip_code', 'average_covered_charges_($)', 'average_total_payments_($)', 'average_medicare_payments_($)']]
# df_speticemia = df_septicemia[['provider_zip_code', 'average_covered_charges_($)', 'average_total_payments_($)', 'average_medicare_payments_($)']]

# # Group by `provider_zip_code` and sum over `average_covered_charges_($)`, `average_total_payments_($)`, and `average_medicare_payments_($)` columns
# grouped_df_alcohol_drug_abuse = df_alcohol_drug_abuse.groupby('provider_zip_code')[['average_covered_charges_($)', 'average_total_payments_($)', 'average_medicare_payments_($)']].sum()
# grouped_df_speticemia = df_speticemia.groupby('provider_zip_code')[['average_covered_charges_($)', 'average_total_payments_($)', 'average_medicare_payments_($)']].sum()

# # Sort grouped DataFrames in descending order of `average_covered_charges_($)`, `average_total_payments_($)`, and `average_medicare_payments_($)` columns
# grouped_df_alcohol_drug_abuse = grouped_df_alcohol_drug_abuse.sort_values(by=['average_covered_charges_($)', 'average_total_payments_($)', 'average_medicare_payments_($)'], ascending=False)
# grouped_df_speticemia = grouped_df_speticemia.sort_values(by=['average_covered_charges_($)', 'average_total_payments_($)', 'average_medicare_payments_($)'], ascending=False)

# # Display the first 3 rows
# print("Alcohol and Drug Abuse DataFrame - Zip codes with the highest income:")
# print(grouped_df_alcohol_drug_abuse.head(3).to_markdown(numalign="left", stralign="left"))

# print("\nSepticemia DataFrame - Zip codes with the highest income:")
# print(grouped_df_speticemia.head(3).to_markdown(numalign="left", stralign="left"))




























'''
    CREATE A PLOT THAT USES DAYS OF THE WEEK, SO I HAVE TO CONVERT THE DATES THEN EXTRACT THE DAY AND ITS NAME.
'''

# import pandas as pd
# import matplotlib.pyplot as plt
# import seaborn as sns

# # Load the TSV file again with correct header settings
# df_strawberry_sales = pd.read_csv('./CSVs/STRAWBERRY SALES 2023 - Sheet1.tsv', sep='\t', skiprows=2)
# print(df_strawberry_sales.columns)
# print(df_strawberry_sales.head())
# print(df_strawberry_sales.info(verbose=1))

# # Fix the column names
# df_strawberry_sales.columns = ['DATE', 'NUM_CLAMSHELLS', 'NUM_BOXES', 'NUM_KILOS',
#        'PRICE_PER_BOX', 'TOTAL', 'PRODUCT', 'TYPE_OF_PRODUCT']

# # Convert the 'DATE ' column to datetime
# df_strawberry_sales['DATE'] = pd.to_datetime(df_strawberry_sales['DATE'], format='mixed')

# # Extract the day of the week from the 'DATE ' column
# df_strawberry_sales['DAY_OF_WEEK'] = df_strawberry_sales['DATE'].dt.day_name()

# # Remove the dollar sign and commas from the 'TOTAL ' column and convert to float
# df_strawberry_sales['TOTAL'] = df_strawberry_sales['TOTAL'].replace('[\$,]', '', regex=True).astype(float)

# # Group by 'TYPE OF PRODUCT' and 'DAY_OF_WEEK' and calculate the average 'TOTAL '
# avg_sales_by_day = df_strawberry_sales.groupby(['TYPE_OF_PRODUCT', 'DAY_OF_WEEK'])['TOTAL'].mean().unstack()

# # Define the order of the days of the week
# ordered_days = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']

# # Reindex the columns to follow the correct order
# avg_sales_by_day = avg_sales_by_day[ordered_days]

# # Display the average sales by day
# print('Average sales by day:')
# print(avg_sales_by_day)

# # Plot the heatmap
# plt.figure(figsize=(12, 8))
# sns.heatmap(avg_sales_by_day, 
#             annot=True, 
#             fmt='.2f', 
#             cmap='viridis')
# plt.title('Average Daily Total Sales Value for Each Type of Product by Day of the Week')
# plt.xlabel('Day of the Week')
# plt.ylabel('Type of Product')
# plt.show()



























'''
    IF YOU PLOT DATES AND THE TIMESTAMP IS INCLUDED IN THE DATES, BUT YOU WANT TO OMIT THEM, THEN USE THE CODE BELOW
'''


# import pandas as pd
# import matplotlib.pyplot as plt

# # Load the trapiche_ingenio_nv.csv dataset
# df_trapiche = pd.read_csv('./CSVs/trapiche_ingenio_nv.csv', encoding='ascii')

# # Strip leading and trailing spaces from column names
# df_trapiche.columns = df_trapiche.columns.str.strip()

# print(df_trapiche.info(verbose=1))
# print(df_trapiche['Fecha'].unique())

# # Convert the 'Fecha' column to datetime, but ONLY use the date, not the timestamp
# df_trapiche['Fecha'] = pd.to_datetime(df_trapiche['Fecha']).dt.date

# print("AFTER")
# print(df_trapiche['Fecha'].unique())

# # Group by 'Fecha' and calculate the sum of 'Bruto' and 'Neto' columns
# sum_bruto_neto_by_date = df_trapiche.groupby('Fecha')[['Bruto', 'Neto']].sum()

# # Plot the bar chart
# plt.figure(figsize=(14, 8))
# sum_bruto_neto_by_date.plot(kind='bar', stacked=False)
# plt.title('Sum of Bruto and Neto by Date')
# plt.xlabel('Date')
# plt.ylabel('Sum')
# plt.xticks(rotation=45)
# plt.legend(['Bruto', 'Neto'])
# plt.show()






























'''
    FIND OUT HOW THE MERGE FUNCTION WORKS AND WHAT IT DOES.
'''

# import pandas as pd

# # Load the dataset
# file_path = './CSVs/Ventas_Julio-Octubre-wines.xlsxcsv-Julio-Octubre.csv'
# df = pd.read_csv(file_path)

# print(df.head())
# print(df.columns)
# print(df.info(verbose=1))

# # df['Ítem - Impte. Fact. Loc.'] = pd.to_numeric(df['Ítem - Impte. Fact. Loc.'], errors='coerce')

# # Group by customer and calculate total spending and total refunds
# customer_spending = df.groupby('Cliente - País - Cód.')['Ítem - Impte. Fact. Loc.'].sum().reset_index()
# customer_spending.columns = ['Cliente - País - Cód.', 'Total_Spending']

# # Calculate total refunds (negative values)
# customer_refunds = df[df['Ítem - Impte. Fact. Loc.'] < 0].groupby('Cliente - País - Cód.')['Ítem - Impte. Fact. Loc.'].sum().reset_index()
# customer_refunds.columns = ['Cliente - País - Cód.', 'Total_Refunds']

# # Merge spending and refunds dataframes
# customer_summary = pd.merge(customer_spending, customer_refunds, on='Cliente - País - Cód.', how='left')
# customer_summary['Total_Refunds'] = customer_summary['Total_Refunds'].fillna(0)

# # Calculate the score as a percentage
# customer_summary['Score'] = (customer_summary['Total_Spending'] + customer_summary['Total_Refunds']) / customer_summary['Total_Spending'] * 100

# # Display the customer summary
# print(customer_summary.head())

# # Count the number of refunds
# num_refunds = df[df['Ítem - Impte. Fact. Loc.'] < 0].shape[0]
# refunds_group = df[df['Ítem - Impte. Fact. Loc.'] < 0]
# print('Number of refunds:', num_refunds)
# print(refunds_group['Ítem - Impte. Fact. Loc.'].unique())
# print(refunds_group['Cliente - País - Cód.'].unique())
































"""
    Function to print the head, column names, and info of a DataFrame.
    
    Parameters:
    df (pd.DataFrame): The DataFrame to analyze.
"""

import pandas as pd

def analyze_dataframe(df):

    print("\nANALYZING THE DATA...\n")
    
    # Print the first few rows of the DataFrame
    print("Head of the DataFrame:")
    print(df.head(), "\n")
    
    # Print the column names of the DataFrame
    print("Column names:")
    print(df.columns, "\n")
    
    # Print the info of the DataFrame
    print("Info:")
    print(df.info(verbose=1), "\n")














# NOTE: ADD FUNCTIONALITY FOR DELIMITERS, ENCODING FLAGS, AND FINISH THE MULTI-SHEET EXCEL FUNCTION THAT EXTRACTS SHEETS!

'''
    METHOD TO TRY MULTIPLE ENCODINGS
'''
# encodings = ['cp1252', 'utf-16', 'ascii', 'latin-1', 'MacRoman', 'Windows-1252']
# for enc in encodings:
#     try:
#         split_data = pd.read_csv(xlsx_file_path, sep='|', skiprows=[1], encoding=enc)
#         print(f"Success with encoding: {enc}")
#         break
#     except UnicodeDecodeError:
#         print(f"Failed with encoding: {enc}")

"""
    Function to load a file from the "./CSVs/" directory and return a DataFrame.

    Parameters:
    file_name (str): The name of the file to load.
    file_type (str): The type of the file ('csv' or 'excel'). Default is 'csv'.

    Returns:
    pd.DataFrame: The loaded DataFrame.
"""


# import pandas as pd

# def load_file(file_name, file_type='csv'):
    
#     # Construct the full file path
#     file_path = f"./CSVs/{file_name}"
    
#     # Load the file based on the file type
#     if file_type == 'csv':
#         df = pd.read_csv(file_path)
#     elif file_type == 'excel':
#         df = pd.read_excel(file_path)
#     elif file_type == 'multi':
#         df = pd.ExcelFile(file_path)
#         sheets = df.sheet_names
#         print("The file has the following sheets:", sheets)
#     else:
#         raise ValueError("Unsupported file type. Use 'csv' or 'excel'.")
    
#     print("\nLOADING THE DATA...\n")
    
#     return df

# Example usage:
# df_csv = load_file('example.csv', 'csv')
# df_excel = load_file('example.xlsx', 'excel')




import pandas as pd

def load_file(file_name, file_type='csv', skiprows=0):
    """
    Load a file from the "./CSVs/" directory and return a DataFrame.

    Parameters:
    file_name (str): The name of the file to load.
    file_type (str): The type of the file ('csv', 'excel', 'multi'). Default is 'csv'.

    Returns:
    pd.DataFrame: The loaded DataFrame.
    """
    # Construct the full file path
    file_path = f"./CSVs/{file_name}"

    # List of common encodings to try
    encodings = ['utf-8', 'latin-1', 'utf-16', 'cp1252', 'iso-8859-1']

    def try_loading_file(encoding=None):
        if file_type == 'csv':
            return pd.read_csv(file_path, encoding=encoding, skiprows=skiprows)
        elif file_type == 'tsv':
            return pd.read_csv(file_path, delimiter='\t', encoding=encoding, skiprows=skiprows)
        elif file_type == 'excel':
            return pd.read_excel(file_path, skiprows=skiprows)
        elif file_type == 'multi':
            return pd.ExcelFile(file_path, skiprows=skiprows)
        else:
            raise ValueError("Unsupported file type. Use 'csv', 'tsv', 'excel', or 'multi'.")

    # Try to load the file without specifying an encoding first
    try:
        df = try_loading_file()
    # If the file doesnt exist
    # NOTE: NEEDS A MORE ELEGANT SOLUTE... IS THIS EVEN NEEDED? BC THE CODE KEEPS GOING SINCE IT RETURNS NONE
    except FileNotFoundError as e:
        print(f"Error: {e}")
        print("The specified file was not found. Please check the file path and try again.")
        return None
    # If decoding was the issue
    except Exception as e:
        print(f"Failed to load file without specifying encoding: {e}")
        for encoding in encodings:
            try:
                print(f"Trying to load file with encoding: {encoding}")
                df = try_loading_file(encoding=encoding)
                print("File loaded successfully with encoding:", encoding)
                break
            except Exception as e:
                print(f"Failed to load file with encoding {encoding}: {e}")
        else:
            raise ValueError("Failed to load file with all attempted encodings.")

    if file_type == 'multi':
        sheets = df.sheet_names
        print("The file has the following sheets:", sheets)
    
    print("\nLOADING THE DATA...\n")

    return df

# Example usage:
# df_csv = load_file('example.csv', 'csv')
# df_excel = load_file('example.xlsx', 'excel')
# df_multi = load_file('example.xlsx', 'multi')











# import pandas as pd

# file_path = './CSVs/paid-ads-top-campaigns-table_2023-11-30_2023-12-29.csv'
# df_ads = pd.read_csv(file_path)

# analyze_dataframe(df_ads)

# # Drop null values in `Last update`
# df_ads.dropna(subset=["Last update"], inplace=True)
# df_ads['Last update'] = pd.to_datetime(df_ads['Last update'])
# print(df_ads["Last update"].unique())

# print("AFTER")
# analyze_dataframe(df_ads)

# # Display rows where 'Last update' is NaT to identify errors
# errors_last_update = df_ads[df_ads['Last update'].isna()]
# print(errors_last_update)

# Display the corrected dataframe
# print(df_ads.head())





























# import altair as alt
# import seaborn as sns
# import matplotlib.pyplot as plt

# # Load the dataset from the Excel file
# file_path = './CSVs/phone_buying_preference1.xlsx'
# df_phone = pd.read_excel(file_path)

# Display the first few rows, column names, and info 
# of the dataframe to understand its structure
# analyze_dataframe(df_phone)

# # Convert relevant columns to categorical data type for better analysis
# categorical_columns = [
#     'sex', 'Age bracket', 'Is phone colour a bother when purchasing a phone',
#     'Phone internal storage increased, decreased or remain the same compared to previous phone',
#     'Is previous phone still fuctional or faulty and unfunctional',
#     'Was the price to acquire new phone higher, lower or unchanged ',
#     'Did You transfer your data from the previous phone',
#     'Did you change the phone manufacturer',
#     'Did you consider the improvement of the camera quality',
#     'Do you consider the year of manufacture of the phone when buying?'
# ]

# df_phone[categorical_columns] = df_phone[categorical_columns].astype('category')

# # Plot the distribution of phone storage changes
# plt.figure(figsize=(10, 6))
# sns.countplot(data=df_phone, x='Phone internal storage increased, decreased or remain the same compared to previous phone')
# plt.title('Distribution of Phone Storage Changes')
# plt.xlabel('Phone Storage Change')
# plt.ylabel('Count')
# plt.show()

# # Plot the relationship between phone storage change and other factors
# plt.figure(figsize=(14, 8))
# sns.countplot(data=df_phone, x='Phone internal storage increased, decreased or remain the same compared to previous phone', hue='sex')
# plt.title('Phone Storage Change by Sex')
# plt.xlabel('Phone Storage Change')
# plt.ylabel('Count')
# plt.show()

# plt.figure(figsize=(14, 8))
# sns.countplot(data=df_phone, x='Phone internal storage increased, decreased or remain the same compared to previous phone', hue='Age bracket')
# plt.title('Phone Storage Change by Age Bracket')
# plt.xlabel('Phone Storage Change')
# plt.ylabel('Count')
# plt.show()

# plt.figure(figsize=(14, 8))
# sns.countplot(data=df_phone, x='Phone internal storage increased, decreased or remain the same compared to previous phone', hue='Is phone colour a bother when purchasing a phone')
# plt.title('Phone Storage Change by Phone Colour Preference')
# plt.xlabel('Phone Storage Change')
# plt.ylabel('Count')
# plt.show()

# plt.figure(figsize=(14, 8))
# sns.countplot(data=df_phone, x='Phone internal storage increased, decreased or remain the same compared to previous phone', hue='Is previous phone still fuctional or faulty and unfunctional')
# plt.title('Phone Storage Change by Previous Phone Functionality')
# plt.xlabel('Phone Storage Change')
# plt.ylabel('Count')
# plt.show()

# plt.figure(figsize=(14, 8))
# sns.countplot(data=df_phone, x='Phone internal storage increased, decreased or remain the same compared to previous phone', hue='Was the price to acquire new phone higher, lower or unchanged ')
# plt.title('Phone Storage Change by Price of New Phone')
# plt.xlabel('Phone Storage Change')
# plt.ylabel('Count')
# plt.show()

# plt.figure(figsize=(14, 8))
# sns.countplot(data=df_phone, x='Phone internal storage increased, decreased or remain the same compared to previous phone', hue='Did You transfer your data from the previous phone')
# plt.title('Phone Storage Change by Data Transfer')
# plt.xlabel('Phone Storage Change')
# plt.ylabel('Count')
# plt.show()

# plt.figure(figsize=(14, 8))
# sns.countplot(data=df_phone, x='Phone internal storage increased, decreased or remain the same compared to previous phone', hue='Did you change the phone manufacturer')
# plt.title('Phone Storage Change by Change of Phone Manufacturer')
# plt.xlabel('Phone Storage Change')
# plt.ylabel('Count')
# plt.show()

# plt.figure(figsize=(14, 8))
# sns.countplot(data=df_phone, x='Phone internal storage increased, decreased or remain the same compared to previous phone', hue='Did you consider the improvement of the camera quality')
# plt.title('Phone Storage Change by Camera Quality Consideration')
# plt.xlabel('Phone Storage Change')
# plt.ylabel('Count')
# plt.show()

# plt.figure(figsize=(14, 8))
# sns.countplot(data=df_phone, x='Phone internal storage increased, decreased or remain the same compared to previous phone', hue='Do you consider the year of manufacture of the phone when buying?')
# plt.title('Phone Storage Change by Year of Manufacture Consideration')
# plt.xlabel('Phone Storage Change')
# plt.ylabel('Count')
# plt.show()

# print('Exploratory Data Analysis Completed')




# # Calculate the proportion of 'yes' values for the specified columns
# for col in ['Is phone colour a bother when purchasing a phone', 'Did You transfer your data from the previous phone', 'Did you change the phone manufacturer', 'Did you consider the improvement of the camera quality', 'Do you consider the year of manufacture of the phone when buying?']:
#   proportion_yes = df_phone[col].value_counts(normalize=True).get('yes', 0)
#   print(f"Proportion of 'yes' values for '{col}': {proportion_yes:.1%}")

# # Count the occurrences of each unique value in the 'Phone internal storage increased, decreased or remain the same compared to previous phone' column
# storage_counts = df_phone['Phone internal storage increased, decreased or remain the same compared to previous phone'].value_counts()
# print("\nCounts for 'Phone internal storage increased, decreased or remain the same compared to previous phone':")
# print(storage_counts.to_markdown(numalign="left", stralign="left"))



# # Create a histogram for the column `How often do you usually change phone in months`
# chart1 = alt.Chart(df_phone).mark_bar().encode(
#     alt.X('How often do you usually change phone in months', bin=True, title='Months'),
#     y='count()',
#     tooltip=[alt.Tooltip('How often do you usually change phone in months', bin=True, title='Months'), 'count()']
# ).properties(title='Histogram of How Often People Change Phones').interactive()
# chart1.save('./OutPlots/phone_change_frequency_histogram.html')

# # Create a histogram for the column `on a scale of 1 to 5 rate your current phone compared to the previous`
# chart2 = alt.Chart(df_phone).mark_bar().encode(
#     alt.X('on a scale of 1 to 5 rate your current phone compared to the previous', bin=True, title='Rating'),
#     y='count()',
#     tooltip=[alt.Tooltip('on a scale of 1 to 5 rate your current phone compared to the previous', bin=True, title='Rating'), 'count()']
# ).properties(title='Histogram of Phone Ratings Compared to Previous').interactive()
# chart2.save('./OutPlots/phone_rating_comparison_histogram.html')



# # Filter the DataFrame to only include rows where storage increased
# df_filtered = df_phone[df_phone['Phone internal storage increased, decreased or remain the same compared to previous phone'] == 'Increase']

# # Group by age bracket and previous phone status, and count the occurrences
# df_grouped = df_filtered.groupby(['Age bracket', 'Is previous phone still fuctional or faulty and unfunctional']).size().to_frame(name='count')

# # Create a pivot table with age bracket as the index, previous phone status as the columns, and count as the values
# pivot_table = df_grouped.pivot_table(index='Age bracket', 
#                                      columns='Is previous phone still fuctional or faulty and unfunctional', 
#                                      values='count', 
#                                      fill_value=0)

# # Print the pivot table
# print(pivot_table.to_markdown(numalign="left", stralign="left"))


# # Filter the DataFrame to only include rows where storage increased
# df_filtered = df_phone[df_phone['Phone internal storage increased, decreased or remain the same compared to previous phone'] == 'Increase']

# # Group by age bracket and previous phone status, and count the occurrences
# df_grouped = df_filtered.groupby(['Age bracket', 'Was the price to acquire new phone higher, lower or unchanged ']).size().to_frame(name='count')

# # Create a pivot table with age bracket as the index, previous phone status as the columns, and count as the values
# pivot_table = df_grouped.pivot_table(index='Age bracket', 
#                                      columns='Was the price to acquire new phone higher, lower or unchanged ', 
#                                      values='count', 
#                                      fill_value=0)

# # Print the pivot table
# print(pivot_table.to_markdown(numalign="left", stralign="left"))



























'''
    FIND THE PERSON THAT GAVE THE MOST FEEDBACK
'''

# # Load the dataset from the CSV file
# file_path = './CSVs/ACS_Ecom-purchase_Q2-Naman_Paliwal.csv'
# df_ecom = pd.read_csv(file_path)

# analyze_dataframe(df_ecom)

# print(df_ecom["Customer Name"].unique())

# # Find the customer who provided the most feedback
# most_feedback_customer = df_ecom['Customer Name'].value_counts().idxmax()
# feedback_count = df_ecom['Customer Name'].value_counts().max()
# print('Customer who provided the most feedback:', most_feedback_customer)
# print('Number of feedbacks provided:', feedback_count)

'''
    A MORE STREAMLINED APPROACH: 1-LINER
'''

# # Summarize the amount of feedback by customer
# feedback_summary = df_ecom.groupby('Customer Name')['Customer Feedback'].count().sort_values(ascending=False)

# print(feedback_summary.head())
































'''
    AN ATTEMPT TO CALCULATE THE DISTANCE COVARIANCE
'''

# # Load the dataset from the TSV file
# file_path = './CSVs/Retail Store Performance and Capacity Metrics - EXO2E Crypto - cccvvv.tsv'
# df_retail = pd.read_csv(file_path, sep='\t')

# mean_gcs_ngk = df_retail['GCS NGK'].mean()
# mean_gcs_counter = df_retail['GCS Counter '].mean()
# print("MEANS:\n")
# print(mean_gcs_ngk)
# print(mean_gcs_counter)

# df_retail['distance_from_mean'] = ((df_retail['GCS NGK'] - mean_gcs_ngk)**2 + (df_retail['GCS Counter '] - mean_gcs_counter)**2)**0.5

# # Calculate the "country value" as the average of 'distance_from_mean' for each country
# country_value = df_retail.groupby('Country')['distance_from_mean'].mean().reset_index(name='Country Value')

# print(country_value.head())

# # analyze_dataframe(df_retail)
# # print(df_retail['Country'].unique())

# # Calculate the covariance between 'GCS NGK' and 'GCS Counter'
# covariance = df_retail[['GCS NGK', 'GCS Counter ']].cov().iloc[0, 1]
# print('Covariance between GCS NGK and GCS Counter:', covariance)





























'''
    Add a new column to the dataset calculating the total monthly cost per patient based on session price, frequency, and reimbursement.

    THIS INVOLVED SOME PRETTY COOL FUNCTIONS THAT MAP CATEGORICAL STATS TO NUMERICAL ONES FOR COMPUTE.
'''

# import matplotlib.pyplot as plt
# import seaborn as sns

# # Load the patient list dataset
# file_path_patient = './CSVs/patient_list-_patient_list.csv'
# df_patient = pd.read_csv(file_path_patient)

# analyze_dataframe(df_patient)
# print(df_patient['frequency'].unique())
# print(df_patient['% reimbursement'].unique())

# # Function to convert frequency to the number of sessions per month
# def frequency_to_sessions_per_month(frequency):
#     if frequency == 'weekly':
#         return 4
#     elif frequency == 'bi-weekly':
#         return 2
#     elif frequency == 'monthly':
#         return 1
#     else:
#         return 0  # Handle any unexpected values
    
# # Function to convert frequency to the number of sessions per month
# def reimbursement_conversion(percentage):
#     if percentage == '10%':
#         return 0.1
#     elif percentage == '5%':
#         return 0.05
#     elif percentage == '15%':
#         return 0.15
#     elif percentage == '33%':
#         return 0.33
#     else:
#         return 0  # Handle any unexpected values

# # Apply the function to create a new column for the number of sessions per month
# df_patient['sessions_per_month'] = df_patient['frequency'].apply(frequency_to_sessions_per_month)

# # Apply the function to create a new column for the percentages
# df_patient['reimbursement_perc'] = df_patient['% reimbursement'].apply(reimbursement_conversion)
# df_patient['net'] = df_patient['session_price'] - (df_patient['session_price'] * df_patient['reimbursement_perc'])

# # Calculate the total monthly cost per patient
# df_patient['total_monthly_cost'] = df_patient['net'] * df_patient['sessions_per_month'] 

# # Display the first few rows of the updated dataframe
# print(df_patient.head(n=20))

# # Create a histogram to show the distribution of total monthly cost
# plt.figure(figsize=(10, 6))
# sns.histplot(df_patient['total_monthly_cost'].dropna(), bins=20, kde=True)
# plt.title('Distribution of Total Monthly Cost')
# plt.xlabel('Total Monthly Cost')
# plt.ylabel('Frequency')
# plt.grid(True)
# plt.show()































# import matplotlib.pyplot as plt
# import seaborn as sns

# # Load the inventory snapshot dataset
# file_path_inventory = './CSVs/inventory-snapshot-table_2024-01-15_ 01 - sheet1.tsv'
# df_inventory = pd.read_csv(file_path_inventory, sep='\t')

# # analyze_dataframe(df_inventory)
# # print(df_inventory['Product category'].unique())

# # Function to clean the "Total value" column
# def clean_total_value(value):
#     # Remove the euro sign and any extra spaces
#     value = value.replace('€', '').strip()
#     # Convert to float
#     return float(value)

# # Apply the cleaning function to the "Total value" column
# df_inventory['Total value'] = df_inventory['Total value'].apply(clean_total_value)

# # Filter the dataset for the 'Earrings' category
# df_earrings = df_inventory[df_inventory['Product category'].str.contains('Earrings', case=False, na=False)]
# print("FIRST")
# print(df_earrings['Product category'].unique())

# # Display the first few rows of the filtered dataframe
# print(df_earrings.head())

# # Display summary statistics for the filtered dataframe
# print(df_earrings.describe(include='all'))

# # Create visualizations for the 'Earrings' category
# plt.figure(figsize=(14, 10))

# # Histogram of Inventory levels
# plt.subplot(2, 2, 1)
# sns.histplot(df_earrings['Inventory'].dropna(), bins=10, kde=True)
# plt.title('Distribution of Inventory Levels')
# plt.xlabel('Inventory')
# plt.ylabel('Frequency')
# plt.grid(True)

# # Histogram of Total value
# plt.subplot(2, 2, 2)
# df_earrings['Total value'] = df_earrings['Total value'].str.replace(' €', '').astype(float)
# sns.histplot(df_earrings['Total value'].dropna(), bins=10, kde=True)
# plt.title('Distribution of Total Value')
# plt.xlabel('Total Value (€)')
# plt.ylabel('Frequency')
# plt.grid(True)

# # Count plot of Last sold on dates
# plt.subplot(2, 2, 3)
# df_earrings['Last sold on'] = pd.to_datetime(df_earrings['Last sold on'], errors='coerce')
# sns.countplot(y=df_earrings['Last sold on'].dt.date)
# plt.title('Last Sold On Dates')
# plt.xlabel('Count')
# plt.ylabel('Last Sold On')
# plt.grid(True)

# plt.tight_layout()
# plt.show()


# # Create a pie chart to visualize the Inventory and Total value columns for the 'Earrings' product category
# plt.figure(figsize=(14, 7))

# # Pie chart for Inventory
# plt.subplot(1, 2, 1)
# plt.pie(df_earrings['Inventory'], labels=df_earrings['Product category'], autopct='%1.1f%%', startangle=140)
# plt.title('Inventory Distribution for Earrings')

# # Pie chart for Total value
# plt.subplot(1, 2, 2)
# plt.pie(df_earrings['Total value'], labels=df_earrings['Product category'], autopct='%1.1f%%', startangle=140)
# plt.title('Total Value Distribution for Earrings')

# plt.tight_layout()
# plt.show()








# df = pd.read_csv(file_path_inventory, sep='\t')

# # Remove '€' from `Total value` column and convert it to numeric datatype
# df['Total value'] = df['Total value'].astype(str).str.replace('€', '', regex=False)
# df['Total value'] = pd.to_numeric(df['Total value'])

# # Fill in missing values of `Total value` with '0'
# df['Total value'] = df['Total value'].fillna(0)

# # Filter the data for the product category 'Earrings'
# earrings_df = df[df['Product category'].str.contains('Earrings')]
# print(earrings_df['Product category'].unique())

# # Group the data on `Product category` and sum the `Inventory` and `Total value` columns
# grouped_df = earrings_df.groupby('Product category')[['Inventory', 'Total value']].sum()

# # Display the table, sorted by `Total value` in descending order
# print(grouped_df.sort_values('Total value', ascending=False))

# import altair as alt

# # Create a pie chart of the `Inventory` and `Total value` columns
# chart1 = alt.Chart(grouped_df.reset_index()).mark_arc(outerRadius=120).encode(
#     theta=alt.Theta(field="Inventory", type="quantitative", stack=True),
#     color=alt.Color(field="Product category", type="nominal"),
#     order=alt.Order(field="Inventory", type="quantitative", sort="descending"),
#     tooltip=[alt.Tooltip(field="Product category", type="nominal"),
#              alt.Tooltip(field="Inventory", type="quantitative", title='Inventory', format = '.2f'),
#              alt.Tooltip(field="Total value", type="quantitative", title='Total value', format = '.2f')]
# ).properties(
#     title='Distribution of Inventory for Earrings'
# ).interactive()

# chart2 = alt.Chart(grouped_df.reset_index()).mark_arc(outerRadius=120).encode(
#     theta=alt.Theta(field="Total value", type="quantitative", stack=True),
#     color=alt.Color(field="Product category", type="nominal"),
#     order=alt.Order(field="Total value", type="quantitative", sort="descending"),
#     tooltip=[alt.Tooltip(field="Product category", type="nominal"),
#              alt.Tooltip(field="Inventory", type="quantitative", title='Inventory', format = '.2f'),
#              alt.Tooltip(field="Total value", type="quantitative", title='Total value', format = '.2f')]
# ).properties(
#     title='Distribution of Total Value for Earrings'
# ).interactive()

# # Save the charts in JSON files
# chart1.save('./OutPlots/inventory_pie_chart.html')
# chart2.save('./OutPlots/total_value_pie_chart.html')









































# import pandas as pd
# import matplotlib.pyplot as plt

# # Load the dataset
# # df = pd.read_csv('caja-dia-a-dia-no-Pii.csv', encoding='utf-8')
# file_path = 'caja-dia-a-dia-no-Pii.csv'
# df = load_file(file_path)

# # analyze_dataframe(df)

# # Convert the 'Fecha' column to datetime format
# df['Fecha'] = pd.to_datetime(df['Fecha'])
# print(df['Fecha'].value_counts())

# # Count the number of occurrences of each unique value in `Fecha`
# date_counts = df['Fecha'].value_counts().reset_index(name='counts').rename(columns={'index': 'Fecha'})
# print(date_counts)

# # Plot the histogram
# plt.figure(figsize=(10, 6))
# plt.hist(df['Fecha'], bins=30, edgecolor='black')
# plt.title('Distribution of Dates')
# plt.xlabel('Date')
# plt.ylabel('Frequency')
# plt.grid(True)
# plt.show()






# import altair as alt

# # Convert `Fecha` to datetime
# df['Fecha'] = pd.to_datetime(df['Fecha'])

# # Create the histogram
# chart = alt.Chart(df).mark_bar().encode(
#     x=alt.X('Fecha', bin=alt.Bin(maxbins=30), title='Date'),
#     y=alt.Y('count()', title='Count'),
#     tooltip=[alt.Tooltip('Fecha', bin=alt.Bin(maxbins=30), title='Date'), 'count()']
# ).properties(
#     title='Distribution of Dates'
# ).interactive()

# # Save the chart
# chart.save('./OutPlots/distribution_of_dates_histogram.html')



# import altair as alt

# # Convert `Fecha` to datetime
# df['Fecha'] = pd.to_datetime(df['Fecha'])

# # Count the number of occurrences of each unique value in `Fecha`
# date_counts = df['Fecha'].value_counts().reset_index(name='counts').rename(columns={'index': 'Fecha'})

# # Create a chart
# chart = alt.Chart(date_counts).mark_bar().encode(
#     x=alt.X('Fecha:T', axis=alt.Axis(title='Date', format='%Y-%m')),
#     y=alt.Y('counts:Q', axis=alt.Axis(title='Count')),
#     tooltip=['Fecha', 'counts']
# ).properties(
#     title='Distribution of Dates'
# ).interactive()

# chart.save('./OutPlots/distribution_of_dates_histogram2.html')

































'''
    GET THE MOST COMMON "MUESTRA" USING TWO APPROACHES. APPROACH 1. USE THE .value_counts() METHOD, APPROACH 2. USE THE .mode()[0] METHOD
'''

# import pandas as pd

# # Load the dataset
# fPath = 'trapiche_ingenio_nv.csv'
# df = load_file(fPath)
# analyze_dataframe(df)

# # Find the most common 'Muestra'
# most_common_muestra = df['Muestra'].mode()

# print('The most common Muestra is:', most_common_muestra)

# # Find the row with the most common 'Muestra'
# most_common_muestra_row = df[df['Muestra'] == most_common_muestra]

# print(most_common_muestra_row)

























# import altair as alt

# fPath = 'ttc-bus-delay-data-2022.csv'
# df_ttc = load_file(fPath)
# analyze_dataframe(df_ttc)

# # A bus is considered delayed if 'Min Delay' is greater than 0
# buses_delayed = df_ttc[df_ttc['Min Delay'] > 0].shape[0]
# buses_not_delayed = df_ttc[df_ttc['Min Delay'] == 0].shape[0]

# print('Number of buses delayed:', buses_delayed)
# print('Number of buses not delayed:', buses_not_delayed)

# # Create the 'delay' column
# df_ttc['delay'] = df_ttc['Min Delay'] + df_ttc['Min Gap']

# # Count the number of entries where 'delay' is in the range [0, 200] inclusive
# delay_count = df_ttc[(df_ttc['delay'] >= 0) & (df_ttc['delay'] <= 200)].shape[0]

# print('Number of entries where delay is in the range [0, 200]:', delay_count)

# # Calculate the total number of buses for each day of the week
# total_buses_per_day = df_ttc['Day'].value_counts()

# # Calculate the number of delayed buses for each day of the week
# delayed_buses_per_day = df_ttc[df_ttc['Min Delay'] > 0]['Day'].value_counts()

# # Calculate the proportion of delayed buses for each day of the week
# proportion_delayed_per_day = (delayed_buses_per_day / total_buses_per_day) * 100

# # Identify the day with the highest and lowest proportion of delayed buses
# highest_proportion_day = proportion_delayed_per_day.idxmax()
# lowest_proportion_day = proportion_delayed_per_day.idxmin()

# print('Day with the highest proportion of delayed buses:', highest_proportion_day)
# print('Day with the lowest proportion of delayed buses:', lowest_proportion_day)







# # Filter the data on `Mechanical` incidents
# df_filtered = df[df['Incident'] == 'Mechanical'].copy()

# # Group by 'Date' and count the number of incidents
# df_grouped = df_filtered.groupby('Date').size().reset_index(name='Incident Count')

# # Sort grouped df
# df_grouped = df_grouped.sort_values(by='Date')

# # Display the first 5 rows
# print(df_grouped.head())

# # Create the line plot
# chart = alt.Chart(df_grouped).mark_line().encode(
#     x=alt.X('Date:T', axis=alt.Axis(title='Date', format='%Y-%m-%d')),
#     y=alt.Y('Incident Count:Q', axis=alt.Axis(title='Count')),
#     tooltip=[alt.Tooltip('Date:T', format='%Y-%m-%d'), 'Incident Count:Q']
# ).properties(
#     title='Mechanical Incidents Over Time'
# ).interactive()

# # Save the chart
# chart.save('./OutPlots/mechanical_incidents_over_time_line_chart2.html')




# # Create a new column `Delay` by adding `Min Delay` and `Min Gap`
# df['Delay'] = df['Min Delay'] + df['Min Gap']

# # Create a new column `Delayed` which is 'True' if `Delay` is greater than 0, otherwise 'False'
# df['Delayed'] = df['Delay'] > 0

# # Print the number of delayed and not delayed buses
# print(f"Number of delayed buses: {df['Delayed'].sum()}")
# print(f"Number of not delayed buses: {len(df) - df['Delayed'].sum()}")

# # Convert the `Time` column to datetime
# df['Time'] = pd.to_datetime(df['Time'], format='%H:%M')

# # Extract the hour from the `Time` column and store it in a new column `Hour`
# df['Hour'] = df['Time'].dt.hour
# print("AFTTER")
# analyze_dataframe(df)

# # Group the data by `Hour` and calculate the mean of `Delayed`
# delays_by_hour = df.groupby('Hour')['Delayed'].mean()

# # Group the data by `Day` and calculate the mean of `Delayed`
# delays_by_day = df.groupby('Day')['Delayed'].mean()

# # Group the data by `Incident` and calculate the mean of `Delayed`
# delays_by_incident = df.groupby('Incident')['Delayed'].mean()

# # Print the results
# print("Delays by hour:")
# print(delays_by_hour.to_markdown(numalign="left", stralign="left"))

# print("\nDelays by day:")
# print(delays_by_day.to_markdown(numalign="left", stralign="left"))

# print("\nDelays by incident:")
# print(delays_by_incident.to_markdown(numalign="left", stralign="left"))

# import altair as alt

# # Plot a histogram of the `Delay` column
# chart = alt.Chart(df).mark_bar().encode(
#     x=alt.X('Delay:Q', bin=True, title='Delay (minutes)'),
#     y=alt.Y('count()', title='Number of Buses'),
#     tooltip=[alt.Tooltip('Delay:Q', bin=True, title='Delay (minutes)'), 'count()']
# ).properties(
#     title='Distribution of Delays'
# ).interactive()

# chart.save('./OutPlots/delay_distribution_histogram.html')


# # Plot a histogram of the `Delay` column
# chart = alt.Chart(df).mark_bar().encode(
#     x=alt.X('Delayed:Q', bin=True, title='Delayed (minutes)'),
#     y=alt.Y('count()', title='Number of Buses'),
#     tooltip=[alt.Tooltip('Delayed:Q', bin=True, title='Delayed (minutes)'), 'count()']
# ).properties(
#     title='Distribution of Delays'
# ).interactive()

# chart.save('./OutPlots/delay_distribution_histogram2.html')











# # Create a line plot of `delays_by_hour` with `Hour` on the x-axis and `Delayed` on the y-axis
# chart1 = alt.Chart(delays_by_hour.reset_index()).mark_line(point=True).encode(
#     x=alt.X('Hour:O', title='Hour'),
#     y=alt.Y('Delayed:Q', title='Proportion of Delayed Buses'),
#     tooltip=['Hour', 'Delayed']
# ).properties(
#     title='Proportion of Delayed Buses by Hour'
# ).interactive()

# # Create a bar plot of `delays_by_day` with `Day` on the x-axis and `Delayed` on the y-axis
# chart2 = alt.Chart(delays_by_day.reset_index()).mark_bar().encode(
#     x=alt.X('Day:O', title='Day'),
#     y=alt.Y('Delayed:Q', title='Proportion of Delayed Buses'),
#     tooltip=['Day', 'Delayed']
# ).properties(
#     title='Proportion of Delayed Buses by Day'
# ).interactive()

# # Create a bar plot of `delays_by_incident` with `Incident` on the x-axis and `Delayed` on the y-axis
# chart3 = alt.Chart(delays_by_incident.reset_index()).mark_bar().encode(
#     x=alt.X('Incident:N', title='Incident'),
#     y=alt.Y('Delayed:Q', title='Proportion of Delayed Buses'),
#     tooltip=['Incident', 'Delayed']
# ).properties(
#     title='Proportion of Delayed Buses by Incident'
# ).interactive()

# # Save the charts
# chart1.save('./OutPlots/delays_by_hour_line_chart.html')
# chart2.save('./OutPlots/delays_by_day_bar_chart.html')
# chart3.save('./OutPlots/delays_by_incident_bar_chart.html')








































# # Load the dataset
# filePath = 'finding_donors_for_charity.csv'
# df_donors = load_file(filePath)
# analyze_dataframe(df_donors)

# print(df_donors['income'].unique())

# # There is only two types of income: <=50k and >50k. I need the latter 

# # Count the number of peeps with an income exceeding 50k
# income_exceeding_50k_count = df_donors[df_donors['income'] == '>50K'].shape[0]
# print('Number of peeps with an income exceeding 50k:', income_exceeding_50k_count)

































# filePath = 'Real Estate Mumbai Database - Rgdcvvvh.csv'
# df_real_estate = load_file(filePath)
# analyze_dataframe(df_real_estate)

# # Convert 'PROPERTY STREET' and 'PROPERTY ADDRESS' to string type
# # df_real_estate['PROPERTY STREET'] = df_real_estate['PROPERTY STREET'].astype(str)
# df_real_estate['PROPERTY ADDRESS'] = df_real_estate['PROPERTY ADDRESS'].astype(str)

# # Collapse the 'PROPERTY STREET' and 'PROPERTY ADDRESS' columns into one column
# # Create a new column 'PROPERTY FULL ADDRESS' by concatenating 'PROPERTY STREET' and 'PROPERTY ADDRESS'
# df_real_estate['PROPERTY FULL ADDRESS'] = df_real_estate['PROPERTY STREET'] + ', ' + df_real_estate['PROPERTY ADDRESS']


# '''APPROACH 2'''

# df_real_estate['PROPERTY FULL ADDRESS'] = df_real_estate['PROPERTY STREET'] + ', ' + df_real_estate['PROPERTY ADDRESS'].astype(str)

# # Drop the original 'PROPERTY STREET' and 'PROPERTY ADDRESS' columns
# df_real_estate = df_real_estate.drop(columns=['PROPERTY STREET', 'PROPERTY ADDRESS'])

# print("AFTER")
# analyze_dataframe(df_real_estate)

# # Display the first few rows of the updated dataframe
# print(df_real_estate.head())
































# # Load the dataset
# df_book_sales = load_file('Book_Sales.csv')
# analyze_dataframe(df_book_sales)

# print(df_book_sales['Zone'].unique())

# # Clean up the ' Total Sales' column
# # NOTE: this also takes care of the quotes implicitly!
# df_book_sales[' Total Sales'] = df_book_sales[' Total Sales'].replace({',': '', '€': ''}, regex=True).astype(float)
# df_book_sales[' Unit Price'] = df_book_sales[' Unit Price'].replace({',': '', '€': ''}, regex=True).astype(float)

# # Create a new column 'Region' to categorize orders as 'Europe' or 'Rest of the World' based on the 'Zone' column
# df_book_sales['Region'] = df_book_sales['Zone'].apply(lambda x: 'Europe' if 'Europe' in x else 'Rest of the World')

# # Group by 'Region' and calculate the total number of orders, total sales, and average unit price
# region_summary = df_book_sales.groupby('Region').agg({'Order ID': 'count', 
#                                                       ' Total Sales': 'sum', 
#                                                       ' Unit Price': 'mean',
#                                                       'units': 'sum'}).reset_index()

# print(region_summary)

# # Verify the changes
# print("AFTER")
# print(df_book_sales[[' Total Sales']].head())
# analyze_dataframe(df_book_sales)

# # Group by 'Region' and calculate the total number of orders and total sales
# region_summary = df_book_sales.groupby('Zone').agg({' Total Sales': 'sum'}).reset_index()

# print(region_summary)








# # Load the dataset
# df = load_file('Book_Sales.csv')
# analyze_dataframe(df)

# df[' Total Sales'] = df[' Total Sales'].astype(str).str.replace('€', '', regex=False).str.replace(',', '', regex=False)

# # Convert ` Total Sales` column to numeric
# df[' Total Sales'] = pd.to_numeric(df[' Total Sales'])

# # Create a `Non-Europe` zone by filtering all rows where `Zone` column is not equal to 'Europe'
# df['Zone'] = df['Zone'].apply(lambda x: 'Non-Europe' if x != 'Europe' else x)
# print("AFTER")
# analyze_dataframe(df)

# # Group by `Zone` column and calculate the sum of ` Total Sales` and `units` columns, and count of distinct values of `Order ID` column
# df_agg = df.groupby('Zone').agg(
#     Total_Sales_Amount=(' Total Sales', 'sum'),
#     Total_Units_Sold=('units', 'sum'),
#     Total_Number_of_Orders=('Order ID', 'nunique')
# ).reset_index()

# print("\nSALES AND UNITS SOLD:\n")
# print(df_agg['Total_Sales_Amount'])
# print(df_agg['Total_Units_Sold'])

# # Divide the sum of ` Total Sales` by sum of `units` to get the average unit price
# df_agg['Average Unit Price'] = df_agg['Total_Sales_Amount'] / df_agg['Total_Units_Sold']

# # Print the above calculated metrics for `Europe` and `Non-Europe` zones
# print(df_agg)




























# # Load the dataset
# import pandas as pd

# filePath = 'last60.csv'
# df_last60 = load_file(filePath)
# analyze_dataframe(df_last60)

# # Gather all records that have missing values in the 'MapPrice' column
# missing_mapprice_records = df_last60[df_last60['MapPrice'].isna()]

# print(missing_mapprice_records.head())
# print('Total records with missing MapPrice:', missing_mapprice_records.shape[0])


'''2ND APPROACH'''
# # Filter the DataFrame to only include rows where `MapPrice` has missing values
# filtered_df = df_last60[df_last60['MapPrice'].isnull()]
# print('Total records with missing MapPrice:', filtered_df.shape[0])

























# import pandas as pd
# import seaborn as sns
# import matplotlib.pyplot as plt

# fPath = 'business_unit_system_cash_flow.csv'
# df_cash_flow = load_file(fPath)
# analyze_dataframe(df_cash_flow)
# print(df_cash_flow['Unidad de Negocio'].unique())

# # Plot the relationship
# plt.figure(figsize=(12, 6))
# sns.scatterplot(data=df_cash_flow, x='Efectivo', y='Banco', hue='Unidad de Negocio', s=100)
# plt.title('Relationship between Cash Balances and Bank Balances for Each Business Unit')
# plt.xlabel('Cash Balance (Efectivo)')
# plt.ylabel('Bank Balance (Banco)')
# plt.legend(title='Business Unit', bbox_to_anchor=(1.05, 1), loc='upper left')
# plt.grid(True)
# plt.show()


# # Group by 'Unidad de Negocio' and calculate the mean of 'Efectivo' and 'Banco'
# business_unit_summary = df_cash_flow.groupby('Unidad de Negocio').agg({'Efectivo': 'mean', 'Banco': 'mean'}).reset_index()

# # Plot the relationship
# plt.figure(figsize=(12, 6))
# sns.scatterplot(data=business_unit_summary, x='Efectivo', y='Banco', hue='Unidad de Negocio', s=100)
# plt.title('Relationship between Cash Balances and Bank Balances for Each Business Unit')
# plt.xlabel('Average Cash Balance (Efectivo)')
# plt.ylabel('Average Bank Balance (Banco)')
# plt.legend(title='Business Unit', bbox_to_anchor=(1.05, 1), loc='upper left')
# plt.grid(True)
# plt.show()

# print(business_unit_summary.head())





# import altair as alt

# # Group the data by `Unidad de Negocio` and sum the `Efectivo` and `Banco` columns
# grouped_df = df_cash_flow.groupby('Unidad de Negocio')[['Efectivo', 'Banco']].sum().reset_index()

# # Create a scatter plot with `Efectivo` on the x-axis and `Banco` on the y-axis.
# # Color the points by `Unidad de Negocio`.
# chart = alt.Chart(grouped_df).mark_circle().encode(
#     x='Efectivo',
#     y='Banco',
#     color='Unidad de Negocio',
#     tooltip=['Efectivo', 'Banco', 'Unidad de Negocio']
# ).properties(
#     title='Relationship between Cash Balances and Bank Balances by Business Unit'
# ).interactive()

# # Add a line of best fit to the plot.
# chart += chart.transform_regression('Efectivo', 'Banco').mark_line()

# # Save the chart
# chart.save('./OutPlots/cash_balances_vs_bank_balances_scatter_plot4.html')

# # Group by `Unidad de Negocio` and calculate the correlation between `Efectivo` and `Banco`.
# correlations = df_cash_flow.groupby('Unidad de Negocio')[['Efectivo', 'Banco']].corr().unstack().iloc[:, 1]

# # Sort the results in descending order by correlation.
# correlations = correlations.sort_values(ascending=False)

# # Display the first 10 rows
# print(correlations.head(10))































# # Load the dataset
# import pandas as pd
# import seaborn as sns
# import matplotlib.pyplot as plt

# fPath = 'current_accounts.csv'
# df_current_accounts = load_file(fPath)

# analyze_dataframe(df_current_accounts)
# print(df_current_accounts['Voucher Type'].unique())

# # Convert the `Credit` column to numeric, using errors='coerce' to convert non-numeric values to NaN
# df_current_accounts['Credit'] = pd.to_numeric(df_current_accounts['Credit'], errors='coerce')
# df_current_accounts['Debit'] = pd.to_numeric(df_current_accounts['Debit'], errors='coerce')

# # Replace missing values in `Credit` with 0
# df_current_accounts['Credit'] = df_current_accounts['Credit'].fillna(0)
# df_current_accounts['Debit'] = df_current_accounts['Debit'].fillna(0)

# filtered_df = df_current_accounts[(df_current_accounts['Debit'] > 0) & (df_current_accounts['Credit'] > 0)]

# print("AFTER")
# analyze_dataframe(df_current_accounts)
# analyze_dataframe(filtered_df)


# # Drop rows with missing values in 'Debit' or 'Credit' columns
# df_current_accounts_clean = df_current_accounts.dropna(subset=['Debit', 'Credit'])

# # Calculate the correlation
# correlation = df_current_accounts_clean[['Debit', 'Credit']].corr()

# # Plot the relationship
# plt.figure(figsize=(10, 6))
# sns.scatterplot(data=df_current_accounts_clean, x='Debit', y='Credit', hue='Voucher Type')
# plt.title('Relationship between Debit and Credit Amounts for Different Vouchers')
# plt.xlabel('Debit Amount')
# plt.ylabel('Credit Amount')
# plt.legend(title='Voucher Type', bbox_to_anchor=(1.05, 1), loc='upper left')
# plt.grid(True)
# plt.show()

# print('Correlation between Debit and Credit amounts:')
# print(correlation)





























'''
    HERE I COMPARE THREE VARIABLES AND ONE IS SMOKING STATUS, WHICH HAS FOUR UNIQUES, SO I MAP THE STATUSES TO SPECIFIC COLORS BC NONSMOKERS SHOULD BE DISTINCT AND NOT A HUE. THE OTHER THRE CAN BE HUES SINCE THEY ARE SUBSETS OF SMOKERS.
'''

'''WITHOUT DISTINCT NONSMOKERS'''
# import pandas as pd
# import seaborn as sns
# import matplotlib.pyplot as plt

# # Load the full_data.csv file
# fPath = 'full_data.csv'
# df = load_file(fPath)

# # Display the head of the dataframe to understand its structure
# analyze_dataframe(df)
# print(df['gender'].unique())
# print(df['smoking_status'].unique())
# print(df['stroke'].unique())

# # Create visualizations to compare BMI and smoking status effects on 
# # health (stroke incidence) in males vs females
# plt.figure(figsize=(14, 6))

# # Plot 1: BMI vs Stroke by Gender
# plt.subplot(1, 2, 1)
# sns.boxplot(x='stroke', 
#             y='bmi', 
#             hue='gender', 
#             data=df)
# plt.title('BMI vs Stroke by Gender')

# # Plot 2: Smoking Status vs Stroke by Gender
# plt.subplot(1, 2, 2)
# sns.countplot(x='stroke', 
#               hue='smoking_status', 
#               data=df[df['gender'] == 'Male'], 
#               palette='Blues', 
#               alpha=0.7)
# sns.countplot(x='stroke', 
#               hue='smoking_status', 
#               data=df[df['gender'] == 'Female'], 
#               palette='Reds', 
#               alpha=0.5)
# plt.title('Smoking Status vs Stroke by Gender')

# Create a scatter plot to illustrate the relationship between smoking status, gender, and BMI

# plt.figure(figsize=(10, 6))

# # Scatter plot: BMI vs Smoking Status by Gender
# sns.scatterplot(x='bmi', y='smoking_status', hue='gender', style='stroke', data=df, alpha=0.7)
# plt.title('BMI vs Smoking Status by Gender and Stroke Incidence')
# plt.xlabel('BMI')
# plt.ylabel('Smoking Status')
# plt.legend(title='Gender')

# # Show the plot
# plt.tight_layout()
# plt.show()

'''APPLY THIS TO GET DISTINCT NONSMOKERS'''

# # Define custom color palette
# # NOTE: these were default gpt, change them to make sense!
# custom_palette = {
#     'formerly smoked': '#1f77b4',  # Blue
#     'never smoked': '#ff7f0e',     # Orange
#     'smokes': '#2ca02c',           # Green
#     'Unknown': '#d62728'           # Red
# }

# # Plot 2: Smoking Status vs Stroke by Gender
# plt.figure(figsize=(14, 6))

# plt.subplot(1, 2, 1)
# sns.countplot(x='stroke', 
#               hue='smoking_status', 
#               data=df[df['gender'] == 'Male'], 
#               palette=custom_palette, 
#               alpha=0.7)
# plt.title('Smoking Status vs Stroke by Gender (Male)')

# plt.subplot(1, 2, 2)
# sns.countplot(x='stroke', 
#               hue='smoking_status', 
#               data=df[df['gender'] == 'Female'], 
#               palette=custom_palette, 
#               alpha=0.5)
# plt.title('Smoking Status vs Stroke by Gender (Female)')

# plt.tight_layout()
# plt.show()











'''MB'''

# import altair as alt

# # Group the data by `gender` and `smoking_status` and calculate the mean `bmi` for each group
# mean_bmi = df.groupby(['gender', 'smoking_status'])['bmi'].mean().reset_index()

# # Create two subplots: one for males and one for females
# male_chart = alt.Chart(df[df['gender'] == 'Male'], title='Male').mark_bar().encode(
#     x=alt.X('bmi:Q', bin=True, title='BMI'),
#     y=alt.Y('count()', title='Count'),
#     color=alt.Color('stroke:N', scale={'range': ['blue', 'red']}, legend=alt.Legend(title='Stroke')),
#     tooltip=[alt.Tooltip('bmi:Q', bin=True, title='BMI'), 'count()', 'stroke']
# ).properties(
#     width=300,
#     height=200
# )

# female_chart = alt.Chart(df[df['gender'] == 'Female'], title='Female').mark_bar().encode(
#     x=alt.X('bmi:Q', bin=True, title='BMI'),
#     y=alt.Y('count()', title='Count'),
#     color=alt.Color('stroke:N', scale={'range': ['blue', 'red']}, legend=alt.Legend(title='Stroke')),
#     tooltip=[alt.Tooltip('bmi:Q', bin=True, title='BMI'), 'count()', 'stroke']
# ).properties(
#     width=300,
#     height=200
# )

'''
    THIS IS AN INTERESTING PLOT BC IVE NEVER SEEN THE BLACK VERTICAL LINES BEFORE. 
'''

# # Add a vertical line to each subplot indicating the mean `bmi` for each group
# for gender in ['Male', 'Female']:
#     for smoking_status in df['smoking_status'].unique():
#         mean_bmi_value = mean_bmi[(mean_bmi['gender'] == gender) & (mean_bmi['smoking_status'] == smoking_status)]['bmi'].values[0]
#         rule = alt.Chart(pd.DataFrame({'x': [mean_bmi_value]})).mark_rule(color='black').encode(
#             x='x:Q'
#         )
#         if gender == 'Male':
#             male_chart = male_chart + rule
#         else:
#             female_chart = female_chart + rule

# # Combine the subplots
# fig = alt.hconcat(male_chart, female_chart)

# # Display the plots
# fig.save('./OutPlots/bmi_smoking_stroke_histograms.html')





















'''
    SIMULATE CHANGES IN OUTCOME
'''

# import pandas as pd

# fPath = 'business_unit_system_cash_flow.csv'
# df_cash_flow = load_file(fPath)
# analyze_dataframe(df_cash_flow)

# print(df_cash_flow['Unidad de Negocio'].unique())
# print(df_cash_flow['Período'].unique())
# impact_df = df_cash_flow.copy()

# # Apply a 10% increase in Ingresos
# impact_df['Ingresos_10%_increase'] = impact_df['Ingresos'] * 1.10
# impact_df['Total_10%_increase'] = impact_df['Ingresos_10%_increase'] - impact_df['Egresos']

# # Apply a 5% decrease in Egresos
# impact_df['Egresos_5%_decrease'] = impact_df['Egresos'] * 0.95
# impact_df['Total_5%_decrease'] = impact_df['Ingresos'] - impact_df['Egresos_5%_decrease']

# # Apply both changes simultaneously
# impact_df['Ingresos_10%_increase_Egresos_5%_decrease'] = impact_df['Ingresos'] * 1.10
# impact_df['Egresos_5%_decrease_simultaneous'] = impact_df['Egresos'] * 0.95
# impact_df['Total_simultaneous'] = impact_df['Ingresos_10%_increase_Egresos_5%_decrease'] - impact_df['Egresos_5%_decrease_simultaneous']

# # Create the final table
# final_table = pd.DataFrame({
#     'change': ['10% increase in Ingresos', '5% decrease in Egresos', 'Both changes'],
#     'ingresos': [impact_df['Ingresos_10%_increase'].mean(), impact_df['Ingresos'].mean(), impact_df['Ingresos_10%_increase_Egresos_5%_decrease'].mean()],
#     'egresos': [impact_df['Egresos'].mean(), impact_df['Egresos_5%_decrease'].mean(), impact_df['Egresos_5%_decrease_simultaneous'].mean()],
#     'total': [impact_df['Total_10%_increase'].mean(), impact_df['Total_5%_decrease'].mean(), impact_df['Total_simultaneous'].mean()]
# })

# print(final_table)











'''MB: TRANSLATE SPANISH MONTH NAMES'''
# import altair as alt


# # Create a dictionary to map the Spanish month names to English month names
# month_map = {
#     'Enero': 'January',
#     'Febrero': 'February',
#     'Marzo': 'March',
#     'Abril': 'April',
#     'Mayo': 'May',
#     'Junio': 'June',
#     'Julio': 'July',
#     'Agosto': 'August',
#     'Septiembre': 'September',
#     'Octubre': 'October',
#     'Noviembre': 'November',
#     'Diciembre': 'December'
# }

# # Replace the Spanish month names in the `Período` column with the corresponding English month names
# df_cash_flow['Período'] = df_cash_flow['Período'].astype(str).replace(month_map, regex=True)

# # Convert the `Período` column to datetime
# df_cash_flow['Período'] = pd.to_datetime(df_cash_flow['Período'], format='%B %Y')

# # Calculate the net cash flow
# df_cash_flow['Net Cash Flow'] = df_cash_flow['Ingresos'] - df_cash_flow['Egresos']

# # Group the data by `Unidad de Negocio` and `Período` and calculate the sum of `Net Cash Flow` for each group
# df_grouped = df_cash_flow.groupby(['Unidad de Negocio', 'Período'])['Net Cash Flow'].sum().reset_index()

# # Sort the data by `Unidad de Negocio` and `Período` in ascending order
# df_grouped = df_grouped.sort_values(['Unidad de Negocio', 'Período'])

# # Calculate the percentage change in `Net Cash Flow` for each `Unidad de Negocio` over time
# df_grouped['% Change in Net Cash Flow'] = df_grouped.groupby('Unidad de Negocio')['Net Cash Flow'].pct_change() * 100

# # Display the first 5 rows of the final dataframe
# print(df_grouped.head())

# # Create a line plot with `Período` on the x-axis and `% Change in Net Cash Flow` on the y-axis
# chart = alt.Chart(df_grouped).mark_line().encode(
#     x=alt.X('Período:T', axis=alt.Axis(title='Period')),
#     y=alt.Y('% Change in Net Cash Flow:Q', axis=alt.Axis(title='% Change in Net Cash Flow')),
#     color=alt.Color('Unidad de Negocio:N', legend=alt.Legend(title='Business Unit')),
#     tooltip=['Unidad de Negocio', 'Período', '% Change in Net Cash Flow']
# ).properties(
#     title='% Change in Net Cash Flow Over Time by Business Unit'
# ).interactive()

# # Save the chart in a json file
# chart.save('./OutPlots/percent_change_in_net_cash_flow_over_time_by_business_unit.html')










'''MA: A BETTER WAY TO TRANSLATE THE SPANISH MONTH NAMES'''
# import altair as alt
# import locale

# # Set the locale to Spanish
# locale.setlocale(locale.LC_TIME, 'es_ES.UTF-8')

# df_cash_flow['Período'] = pd.to_datetime(df_cash_flow['Período'], format='%B %Y')
# print("\nAFTER")
# analyze_dataframe(df_cash_flow)

# # Create the multiple line series plot over time.
# chart = (
#    alt.Chart(
#        df_cash_flow
#    ).mark_line()
#    .encode(
#        # Set `Período` on the x-axis. Sort it
#        x=alt.X('Período', sort='x', axis=alt.Axis(labelAngle=-45)),
#        # Set `Total` on the y-axis and add appropriate title to the axis
#        y=alt.Y('Total'),
#        # Have a different color for each `Unidad de Negocio`
#        color='Unidad de Negocio',
#        # Add tooltips for the relevant features to show details on hover
#        tooltip=[alt.Tooltip('Período', title='Period'), 'Total', 'Unidad de Negocio'],
#    )
#    .properties(title='Total Cash Flow Over Time by Business Unit')
#    .interactive() # Add interactive features for zoom and pan
# )
# Save the chart in a JSON file
# chart.save('./OutPlots/total_cash_flow_over_time_by_business_unit_line_chart2.html')



# # Calculate `Total` as the difference between `Ingresos` and `Egresos`
# df_cash_flow['Total'] = df_cash_flow['Ingresos'] - df_cash_flow['Egresos']

# # Group by `Unidad de Negocio` and calculate the mean of `Ingresos`, `Egresos`, and `Total`
# grouped_df = df_cash_flow.groupby('Unidad de Negocio')[['Ingresos', 'Egresos', 'Total']].mean().reset_index()

# # Sort the results in descending order of mean `Total`
# grouped_df = grouped_df.sort_values(by='Total', ascending=False)

# # Display the first 3 rows
# print(grouped_df.head(3).to_markdown(index=False, numalign="left", stralign="left"))


# # Create a new dataframe `changes_df` with columns `Change`, `Ingresos`, `Egresos`, and `Total`
# changes_df = pd.DataFrame(columns=['Change', 'Ingresos', 'Egresos', 'Total'])

# # Add rows to `changes_df` to represent different scenarios of changes in `Ingresos` and `Egresos`
# changes_df = pd.concat([changes_df, pd.DataFrame({'Change': '10% increase in Ingresos', 'Ingresos': 0.1, 'Egresos': 0, 'Total': 0}, index=[0])])
# changes_df = pd.concat([changes_df, pd.DataFrame({'Change': '5% decrease in Egresos', 'Ingresos': 0, 'Egresos': -0.05, 'Total': 0}, index=[0])])
# changes_df = pd.concat([changes_df, pd.DataFrame({'Change': '10% increase in Ingresos and 5% decrease in Egresos', 'Ingresos': 0.1, 'Egresos': -0.05, 'Total': 0}, index=[0])])

# # For each scenario, calculate the new `Total` based on the changes in `Ingresos` and `Egresos`

# for index, row in changes_df.iterrows():
#     changes_df.at[index, 'Total'] = (1 + row['Ingresos']) * grouped_df['Ingresos'].mean() - (1 + row['Egresos']) * grouped_df['Egresos'].mean()

# # Display the `changes_df` DataFrame
# print(changes_df)
































# import pandas as pd

# # Load the dataset
# etsy_statement_path = 'Avidproducts financials - Big Dave.xlsx - etsy_statement_2024_1.tsv'
# df_etsy = load_file(etsy_statement_path, 'tsv')
# analyze_dataframe(df_etsy)
# print(df_etsy['Type'].unique())

# # Extract refund-related data
# refund_data = df_etsy[df_etsy['Type'].str.contains('refund', case=False)]

# print("There are", refund_data.shape[0], "refunds.")
# # Save the refund data to a new TSV file
# refund_data.to_csv('./OutCSVs/refund_data.tsv', sep='\t', index=False)

# print('Refund data extracted and saved to refund_data.tsv')



# Filter all rows where the `Type` column contains 'refund'
# refund_df = df[df['Type'].str.contains('refund', case=False)]


































import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# # Load the dataset
# fPath = 'Top_1000_Bollywood_Movies.csv'
# bollywood_df = load_file(fPath)
# analyze_dataframe(bollywood_df)
# print(bollywood_df['Verdict'].unique())

# # Create the scatterplot
# plt.figure(figsize=(14, 14))

# # Use seaborn for better aesthetics
# scatter_plot = sns.scatterplot(data=bollywood_df, 
#                                x='India Net', 
#                                y='Overseas', 
#                                hue='Verdict', 
#                                size=bollywood_df['Budget'] / 1000000, 
#                                sizes=(100, 500), 
#                                alpha=0.7, 
#                                palette='viridis')

# # Set the title and labels
# scatter_plot.set_title('Scatterplot of Bollywood Movies: India Net vs Overseas', fontsize=20)
# scatter_plot.set_xlabel('India Net', fontsize=15)
# scatter_plot.set_ylabel('Overseas', fontsize=15)

# # Adjust the legend
# plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)

# # Show the plot
# plt.show()



























# import pandas as pd

# # Load the dataset
# fPath = 'FAL Projects NY - office NY - FAL Proyectos.xlsx'
# fal_projects_df = load_file(fPath, 'excel', 9) # NOTE: fix the load method such that it deduces the file type to omit one arg.
# analyze_dataframe(fal_projects_df)

# # Summarize the data characteristics
# num_transactions = fal_projects_df.shape[0]
# transaction_date_range = (fal_projects_df['Create Date:'].min(), fal_projects_df['Create Date:'].max())
# unique_vendors = fal_projects_df['Vendor Name:'].nunique()

# # Print the summary
# print('Number of transactions:', num_transactions)
# print('Range of transaction dates:', transaction_date_range)
# print('Variety of vendors:', unique_vendors)







































# import pandas as pd
# import seaborn as sns
# import matplotlib.pyplot as plt

# # Load the dataset
# fPath = 'LIFE INS ISSUE AGE AUDIT.xlsx'
# life_ins_df = load_file(fPath, 'excel')
# analyze_dataframe(life_ins_df)

# # Create age groups based on decades
# life_ins_df['Age Group'] = (life_ins_df['Issue Age'] // 10) * 10

# # Create a pivot table for the heatmap
# pivot_table = life_ins_df.pivot_table(index='Age Group', 
#                                       columns='Tobacco Use?', 
#                                       values='Mode Premium', 
#                                       aggfunc='mean')

# # Create the heatmap
# plt.figure(figsize=(12, 8))
# heatmap = sns.heatmap(pivot_table, 
#                       annot=True, 
#                       fmt='.2f', 
#                       cmap='coolwarm')

# # Set the title and labels
# heatmap.set_title('Heatmap of Premiums by Age Group and Tobacco Use', fontsize=20)
# heatmap.set_xlabel('Tobacco Use', fontsize=15)
# heatmap.set_ylabel('Age Group', fontsize=15)

# # Show the plot
# plt.show()





# import altair as alt

# fPath = 'LIFE INS ISSUE AGE AUDIT.xlsx'
# df_life_ins = load_file(fPath, 'excel')
# analyze_dataframe(df_life_ins)

# # Drop rows with missing values in the `Issue Age` column of the `df_life_ins` dataframe.
# # df_life_ins.dropna(subset=['Issue Age'], inplace=True)

# # Create a new column `Age Group` in the `df_life_ins` dataframe by dividing the `Issue Age` by 10, converting it to an integer, and then multiplying by 10 to get the decade.
# df_life_ins['Age Group'] = (df_life_ins['Issue Age'] // 10 * 10).astype(int)

# # Group the `df_life_ins` dataframe by `Age Group` and `Tobacco Use?`, and calculate the mean of `Mode Premium`. Store the result in `grouped_data`.
# grouped_data = df_life_ins.groupby(['Age Group', 'Tobacco Use?'])['Mode Premium'].mean().reset_index()

# # Create a pivot table from `grouped_data` with `Age Group` as the index, `Tobacco Use?` as the columns, and `Mode Premium` as the values. Store the result in `pivot_table`.
# pivot_table = grouped_data.pivot(index='Age Group', columns='Tobacco Use?', values='Mode Premium')

# # Create a heatmap using Seaborn with `Age Group` on the x-axis, `Tobacco Use?` on the y-axis, and the mean `Mode Premium` as the color intensity.
# chart = alt.Chart(grouped_data).mark_rect().encode(
#     x='Age Group:O',
#     y='Tobacco Use?:O',
#     color='mean(Mode Premium):Q',
#     tooltip=['Age Group', 'Tobacco Use?', 'mean(Mode Premium)']
# ).properties(
#     title='Mean Mode Premium by Age Group and Tobacco Use'
# )

# chart.save('./OutPlots/mean_mode_premium_by_age_group_and_tobacco_use_heatmap2.html')





































































































































































































































