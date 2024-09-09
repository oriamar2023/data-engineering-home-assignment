from pyspark.sql import SparkSession
from pyspark.sql.functions import *
from pyspark.sql.window import Window
from pyspark.sql.types import StructType, StructField, StringType, DoubleType, IntegerType

import os
from dotenv import load_dotenv

load_dotenv()

aws_access_key_id = os.getenv('AWS_ACCESS_KEY_ID')
aws_secret_access_key = os.getenv('AWS_SECRET_ACCESS_KEY')
stack_name = os.getenv('STACK_NAME')

def upload_df_to_s3_as_parquet(df, bucket_name, s3_path, partition_col=None):
    if partition_col:
        # Write partitioned data
        df.write \
            .mode('overwrite') \
            .partitionBy(partition_col) \
            .parquet(f"s3://{bucket_name}/{s3_path}")
    else:
        # Write non-partitioned data
        df.write \
            .mode('overwrite') \
            .parquet(f"s3://{bucket_name}/{s3_path}")


spark = SparkSession.builder \
    .appName("StockAnalysis") \
    .config("spark.hadoop.fs.file.impl.disable.cache", "true") \
    .getOrCreate()

# Read the data
file_path = '/Users/yuval/Documents/vi_de_task/stocks_data.csv'
df = spark.read.option("header", True).csv(file_path, inferSchema=True)

# Preprocess the data
df = df.withColumn("Date", to_date(col("Date"), "yyyy-MM-dd")) \
    .withColumn("open", col("open").cast("double")) \
    .withColumn("high", col("high").cast("double")) \
    .withColumn("low", col("low").cast("double")) \
    .withColumn("close", col("close").cast("double")) \
    .withColumn("volume", col("volume").cast("long"))

# Sort the data by date and ticker
df = df.orderBy("Date", "ticker")

# Create window specifications
window_spec_desc = Window.partitionBy("ticker").orderBy(col("Date").desc())

# Handle null values in 'open' , 'close' and volume columns
df = df.withColumn("prev_open", last("close", ignorenulls=True).over(window_spec_desc))
df = df.withColumn("prev_close", last("open").over(window_spec_desc))
df = df.withColumn("last_volume", last("volume", ignorenulls=True).over(window_spec_desc))
df = df.withColumn("open", when(col("open").isNull(), col("prev_open")).otherwise(col("open")))
df = df.withColumn("close", when(col("close").isNull(), col("prev_close")).otherwise(col("close")))
df = df.withColumn("volume", when(col("volume").isNull(), col("last_volume")).otherwise(col("volume")))


# Calculate daily returns
df = df.withColumn("return", (col("close") - col("open")) / col("open"))


# 1.Compute the average daily return of all stocks for every date
average_daily_return = df.groupBy("Date") \
    .agg(mean("return").alias("average_return")) \
    .orderBy("Date")

#average_daily_return.coalesce(1).write.csv("/Users/yuval/Documents/vi_de_task/output/average_daily_return.csv", header=True,mode='overwrite')
#average_daily_return.coalesce(1).write.parquet("/Users/yuval/Documents/vi_de_task/output/average_daily_return")
upload_df_to_s3_as_parquet(average_daily_return, 'yuval-amar-data-engineering-assignment', 'highest_worth_stock/highest_worth_stock.parquet',partition_col='Date')

# 2.Which stock was traded with the highest worth - as measured by closing price * volume - on average?
df = df.withColumn('daily_ticker_worth', col('close')*col('volume'))

highest_worth_stock = df.groupBy("ticker") \
    .agg(mean("daily_ticker_worth").alias("daily_ticker_worth_avg")) \
    .orderBy(col("daily_ticker_worth_avg").desc()) \
    .limit(1) \
    .withColumn("value", format_number("daily_ticker_worth_avg",0)) \
    .select("ticker", "value")

highest_worth_stock.coalesce(1).write.csv("/Users/yuval/Documents/vi_de_task/output/highest_worth_stock.csv", header=True,mode='overwrite')
# 3.Which stock was the most volatile as measured by the annualized standard deviation of daily returns?
trading_days = 252
volatilities = df.groupBy("ticker").agg((sqrt(lit(trading_days)) * stddev("return")).alias("standard_deviation"))
df_volatilities_result = volatilities.select(
    col("ticker"),
    format_number(col("standard_deviation"), 6).alias("standard_deviation")
)
df_top_volatilities_result = df_volatilities_result.orderBy(col("standard_deviation").desc()).limit(1)
df_top_volatilities_result.coalesce(1).write.csv("/Users/yuval/Documents/vi_de_task/output/top_volatility_stock.csv", header=True,mode='overwrite')

# 4.What were the top three 30-day return dates as measured by % increase in closing price compared to the closing price 30 days prior? present the top three ticker and date combinations.
df = df.withColumn("30_days_back_close", lead("Close", 30).over(window_spec_desc))
df = df.withColumn("30_days_back_close_date", lead("Date", 30).over(window_spec_desc))
df = df.withColumn("30_days_back_return",
                   when(col("30_days_back_close").isNotNull(),
                        (col("Close") - col("30_days_back_close")) / col("30_days_back_close"))
                   .otherwise(None))

df_returns_from_30_days = df.withColumn("date_edit",
                                           concat(col("Date").cast("string"),
                                                  lit(" - "),
                                                  col("30_days_back_close_date").cast("string")))
df_top_returns_final = df_returns_from_30_days.orderBy(col("30_days_back_return").desc()).limit(3).select(col("Ticker").alias("ticker"), col("date_edit").alias("date"))
df_top_returns_final.coalesce(1).write.csv("/Users/yuval/Documents/vi_de_task/output/df_top_returns.csv", header=True,mode='overwrite')


spark.stop()