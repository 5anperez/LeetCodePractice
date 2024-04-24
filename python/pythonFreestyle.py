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




















# import pandas as pd
# import matplotlib.pyplot as plt

# Read the CSV file into a DataFrame
# df = pd.read_csv('./CSVs/Top_1000_Bollywood_Movies.csv')

# # Filter data by verdict
# df_filtered = df[df['Verdict'].isin(['All Time Blockbuster', 'Blockbuster'])]

# # Select top 10 movies by `India Net`
# df_top_10 = df_filtered.nlargest(10, 'India Net')

# # Create figure and axes objects
# fig, ax1 = plt.subplots(figsize=(12, 8))
# ax2 = ax1.twinx()

# # Create scatter plot for WorldWide Gross
# scatter = ax1.scatter(df_top_10['India Net'], df_top_10['Worldwide'], color='blue', label='Worldwide')

# # Add labels to scatter plot points
# for i, txt in enumerate(df_top_10['Movie']):
#     ax1.annotate(txt, (df_top_10['India Net'][i], df_top_10['Worldwide'][i]), xytext=(5, 5), textcoords='offset points')

# # Create line plot for Budget
# line, = ax2.plot(df_top_10['India Net'], df_top_10['Budget'], color='red', label='Budget')

# # Set axis labels and title
# ax1.set_xlabel('India Net', fontsize=12)
# ax1.set_ylabel('WorldWide', fontsize=12)
# ax2.set_ylabel('Budget', fontsize=12)
# plt.title('India Net vs WorldWide and Budget for Top 10 Grossing Movies (Verdict: ATB or Blockbuster)', fontsize=14)

# # Add a legend
# lines = [scatter, line]
# labels = [l.get_label() for l in lines]
# ax1.legend(lines, labels, loc='upper left')

# # Show plot
# plt.tight_layout()
# plt.show()








import pandas as pd
import matplotlib.pyplot as plt

# Load the dataset
file_path = './CSVs/Top_1000_Bollywood_Movies.csv'
bollywood_data = pd.read_csv(file_path)
print(bollywood_data.head())
print(bollywood_data.info())

# Filter data for 'All Time Blockbuster' or 'Blockbuster' verdicts
filtered_data = bollywood_data[bollywood_data['Verdict'].isin(['All Time Blockbuster', 'Blockbuster'])]

# Sort the filtered data by 'India Net' earnings in descending order
sorted_filtered_data = filtered_data.sort_values(by='India Net', ascending=False)

# Select the top 10 highest-grossing movies
top_10_movies = sorted_filtered_data.head(10)

top_10_movies[['Movie', 'Worldwide', 'India Net', 'Budget', 'Verdict']]



# Recreate the plot with corrected legend handling and x-tick labels
fig, ax1 = plt.subplots(figsize=(12, 8))

# Names of the movies
movies = top_10_movies['Movie']

# First axis for 'Worldwide' earnings
color = 'tab:red'
ax1.set_xlabel('Movie')
ax1.set_ylabel('Worldwide Earnings (Billion INR)', color=color)
lns1 = ax1.bar(movies, top_10_movies['Worldwide']/1e9, color=color, label='Worldwide Earnings', alpha=0.6)
ax1.tick_params(axis='y', labelcolor=color)
ax1.set_xticks(movies.index)
ax1.set_xticklabels(movies, rotation=45, ha='right')

# Second axis for 'India Net' earnings
ax2 = ax1.twinx()
color = 'tab:blue'
ax2.set_ylabel('India Net Earnings (Billion INR)', color=color)
lns2 = ax2.plot(movies, top_10_movies['India Net']/1e9, color=color, label='India Net Earnings', marker='x')
ax2.tick_params(axis='y', labelcolor=color)

# Third axis for 'Budget'
ax3 = ax1.twinx()
color = 'tab:green'
ax3.set_ylabel('Budget (Billion INR)', color=color)
lns3 = ax3.plot(movies, top_10_movies['Budget']/1e9, color=color, label='Budget', marker='o')
ax3.tick_params(axis='y', labelcolor=color)
ax3.spines['right'].set_position(('outward', 60))

# Convert plot handles to a list for the legend
lns = list(lns1) + list(lns2) + list(lns3)
labs = [l.get_label() for l in lns]
ax1.legend(lns, labs, loc=0)

plt.title('Top 10 Highest-Grossing Bollywood Movies (Sorted by India Net Earnings)')
plt.show()
























                  




























































