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
    HERE, WE USE THE CONCAT METHOD TO COMBINE TWO SHEETS OF A MULTI-SHEET EXCELL FILE.
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



import pandas as pd

# Read the CSV file into a DataFrame
df = pd.read_excel('./CSVs/Data_Set_TransactionReport_All_01Nov2023_30Nov2023_20231214165632.xlsx')
print(df.head())
print(df.info())

# Correct the column name for 'Status' (PaymentStatus)
columns_of_interest = [
    'Transaction Date', 'RecordLocator', 'PaymentStatus', 'PaymentAmount'
]
df_filtered = df[columns_of_interest]

# Redefine the date range
start_date = pd.Timestamp('2023-09-01')
end_date = pd.Timestamp('2023-12-31')

# Extract the payment date and time only
payment_dates = df_filtered[
    (df_filtered['Transaction Date'] >= start_date) &
    (df_filtered['Transaction Date'] <= end_date) &
    (df_filtered['PaymentStatus'].str.lower() == 'approved') &
    (df_filtered['RecordLocator'].str.contains(r'\d'))
]['Transaction Date'].dt.strftime('%y-%m-%d %H:%M')

payment_dates_list = payment_dates.tolist()

# Display the results
print(f'\nPayment dates:\n{payment_dates_list}')













































































































































