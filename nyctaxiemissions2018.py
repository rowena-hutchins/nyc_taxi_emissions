# Databricks notebook source
#Import libraries
import pandas as pd
import matplotlib.pyplot as plt
from pyspark.sql.functions import col, avg, sum, count, min, max, when, to_date, year, month, date_format, date_trunc
from pyspark.sql import SparkSession

# COMMAND ----------

# Display sample NYC taxi trip data available in Databricks
display(dbutils.fs.ls('/databricks-datasets/nyctaxi/tripdata/yellow/'))

# COMMAND ----------

# Initialize Spark session
spark = SparkSession.builder.appName("NYC Taxi Emissions Analysis").getOrCreate()

# Import zone name data
zonename_df = pd.read_csv("https://d37ci6vzurychx.cloudfront.net/misc/taxi_zone_lookup.csv")

# Import NYC taxi trip data for yellow taxis in 2018
file_path = "dbfs:/databricks-datasets/nyctaxi/tripdata/yellow/yellow_tripdata_2018*.csv.gz"

# Define reduced data set for faster testing
# file_path = "dbfs:/databricks-datasets/nyctaxi/tripdata/yellow/yellow_tripdata_2018-01.csv.gz"

# Read all files matching the pattern
df_2018_raw = spark.read.csv(file_path, header=True, inferSchema=True)

# Show the first few rows
df_2018_raw.show()

# COMMAND ----------

# Cleanse data set to exclude non-2018 data
column_name = 'tpep_pickup_datetime' 
min_pickup = df_2018_raw.agg(min(column_name).alias("min_pickup")).collect()[0]["min_pickup"]
print(min_pickup)
max_pickup = df_2018_raw.agg(max(column_name).alias("max_pickup")).collect()[0]["max_pickup"]
print(max_pickup)
raw_row_count = df_2018_raw.count()

df_2018 = df_2018_raw.filter(year(col("tpep_pickup_datetime")) == 2018)
row_count_2018 = df_2018.count()

print(f'The df_2018 data frame has {row_count_2018} rows. {raw_row_count - row_count_2018} non-2018 rows were removed.')

# COMMAND ----------

# View data schema
df_2018.printSchema()

# COMMAND ----------

# Calculate fuel consumed (gallons) as Distance travelled (miles) / 22
df_2018 = df_2018.withColumn('fuel_consumed_gallons', col('trip_distance') / 22)

# Calculate CO2 emissions (kg) as Fuel consumed (gallons) x 8.89 kg per gallon
df_2018 = df_2018.withColumn('co2_emissions_kg', col('fuel_consumed_gallons') * 8.89)


# COMMAND ----------

# SUMMARISE EMISSIONS BY CALENDAR DATE AND GROUP SIZE

# Extract the calendar date from the pickup timestamp (to daily level)
df_2018 = df_2018.withColumn("pickup_date", date_format(col("tpep_pickup_datetime"), "yyyy-MM-dd"))

# Add 'group_size' column based on the number of passengers
df_2018 = df_2018.withColumn(
    'group_size', 
    when(col('passenger_count') <= 2, 'small_grp')
    .otherwise('large_grp')
)

# Group by date and group_size, and calculate the total CO2 emissions and total trip count
daily_emissions = df_2018.groupBy('pickup_date', 'group_size').agg(
    sum('co2_emissions_kg').alias('total_co2_emissions_kg'),
    count("tpep_pickup_datetime").alias("trip_count")
    )

# Calculate the ratio of emissions to trip count
daily_emissions = daily_emissions.withColumn("emissions_per_trip", col("total_co2_emissions_kg") / col("trip_count"))

# Convert to Pandas DataFrame for plotting
daily_emissions_pd = daily_emissions.toPandas()

# Ensure the 'date' column is in datetime format for proper sorting
daily_emissions_pd['pickup_date'] = pd.to_datetime(daily_emissions_pd['pickup_date'])

# Sort by date in ascending order
daily_emissions_pd = daily_emissions_pd.sort_values('pickup_date')

# Show the result
print(daily_emissions_pd.head(10))


# COMMAND ----------

# PLOT TOTAL EMISSIONS AND AVERAGE EMISSIONS BY GROUP SIZE PER DAY

# Plot - Total emissions per day by group size for 2018
plt.figure(figsize=(12, 6))
for size in daily_emissions_pd['group_size'].unique():
    subset = daily_emissions_pd[daily_emissions_pd['group_size'] == size]
    plt.plot(subset['pickup_date'], subset['total_co2_emissions_kg'], label=size)

plt.xlabel('Date')
plt.ylabel('Total Emissions (kg CO2)')
plt.title('Total Emissions per Day by Group Size for 2018')
plt.xticks(rotation=45)
plt.legend()
plt.tight_layout()
plt.show()

# Plot - Ratio of emissions/trip per day by group size for 2018
plt.figure(figsize=(12, 6))
for size in daily_emissions_pd['group_size'].unique():
    subset = daily_emissions_pd[daily_emissions_pd['group_size'] == size]
    plt.plot(subset['pickup_date'], subset['emissions_per_trip'], marker='o', label=size)

plt.xlabel('Date')
plt.ylabel('Average Emissions per Trip (kg CO2)')
plt.title('Average Emissions per Trip by Group Size for 2018')
plt.xticks(rotation=45)
plt.legend()
plt.tight_layout()
plt.show()


# COMMAND ----------

# PLOT TRIP COUNT vs PASSENGER COUNT

# Group by passenger count to calculate the trip count
trip_count_by_passenger = df_2018.groupBy("passenger_count").agg(count("tpep_pickup_datetime").alias("trip_count"))

# Convert to Pandas DataFrame for plotting
trip_count_pd = trip_count_by_passenger.toPandas()

# Plotting
plt.figure(figsize=(6, 6))
plt.bar(trip_count_pd['passenger_count'], trip_count_pd['trip_count'], color='skyblue')
plt.xlabel('Number of Passengers')
plt.xlim(0, 10)
plt.ylabel('Trip Count')
plt.title('Trip Count by Passenger Count')
plt.grid(axis='y')
plt.tight_layout()
plt.show()


# COMMAND ----------

# PLOT AVERAGE TRIP DISTANCE vs PASSENGER COUNT

# Group by passenger count to calculate the average trip distance
avg_trip_distance_by_passenger = df_2018.groupBy("passenger_count").agg(avg("trip_distance").alias("avg_trip_distance"))

# Convert to Pandas DataFrame for plotting
avg_trip_distance_pd = avg_trip_distance_by_passenger.toPandas()

# Plotting
plt.figure(figsize=(6, 6))
plt.bar(avg_trip_distance_pd['passenger_count'], avg_trip_distance_pd['avg_trip_distance'], color='lightgreen')
plt.xlabel('Number of Passengers')
plt.xlim(0, 10)
plt.ylabel('Average Trip Distance (miles)')
plt.title('Average Trip Distance by Passenger Count')
plt.grid(axis='y')
plt.tight_layout()
plt.show()
