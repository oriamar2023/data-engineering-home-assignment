from awsglue.transforms import *
from pyspark.context import SparkContext
from awsglue.context import GlueContext
from awsglue.job import Job
from awsglue.dynamicframe import DynamicFrame
from pyspark.sql.functions import *
from pyspark.sql.window import Window
from pyspark.sql.types import StructType, StructField, StringType, DoubleType, IntegerType


sc = SparkContext.getOrCreate()
glueContext = GlueContext(sc)
spark = glueContext.spark_session
job = Job(glueContext)
job.init('stock_etl_analysis')

def upload_df_to_s3_as_parquet(df, bucket_name, prefix_path):
    s3_path = f"s3://{bucket_name}/{prefix_path}"
    df.write \
            .mode('overwrite') \
            .parquet(s3_path)

input_path = "s3://yuval-amar-data-engineering-assignment/raw_data/stocks_data.csv"
df = spark.read.option("header", True).csv(input_path, inferSchema=True)

df = df.withColumn("Date", to_date(col("Date"), "yyyy-MM-dd")) \
    .withColumn("open", col("open").cast("double")) \
    .withColumn("high", col("high").cast("double")) \
    .withColumn("low", col("low").cast("double")) \
    .withColumn("close", col("close").cast("double")) \
    .withColumn("volume", col("volume").cast("long"))

df = df.orderBy("ticker","Date")

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

# 1. Compute the average daily return of all stocks for every date
average_daily_return = df.groupBy(col("Date").alias("date")) \
    .agg(mean("return").alias("average_return")) \
    .orderBy("date")

upload_df_to_s3_as_parquet(average_daily_return, 'yuval-amar-data-engineering-assignment', 'daily_average_return')

# 2. Which stock was traded with the highest worth
df = df.withColumn('daily_ticker_worth', col('close')*col('volume'))

highest_worth_stock = df.groupBy("ticker") \
    .agg(mean("daily_ticker_worth").alias("daily_ticker_worth_avg")) \
    .orderBy(col("daily_ticker_worth_avg").desc()) \
    .limit(1) \
    .withColumn("value", format_number("daily_ticker_worth_avg", 0)) \
    .select("ticker", "value")

upload_df_to_s3_as_parquet(highest_worth_stock, 'yuval-amar-data-engineering-assignment', 'highest_worth_ticker')

# 3. Which stock was the most volatile
trading_days = 252
volatilities = df.groupBy("ticker").agg((sqrt(lit(trading_days)) * stddev("return")).alias("standard_deviation"))
df_volatilities_result = volatilities.select(
    col("ticker"),
    format_number(col("standard_deviation"), 6).alias("standard_deviation")
)

upload_df_to_s3_as_parquet(df_volatilities_result, 'yuval-amar-data-engineering-assignment', 'annualized_std_daily_returns')

# 4. What were the top three 30-day return dates
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

upload_df_to_s3_as_parquet(df_top_returns_final, 'yuval-amar-data-engineering-assignment', 'top_three_30_day_return')

job.commit()