from pyspark.sql import SparkSession
from pyspark.sql.functions import *
from pyspark.sql.types import *
import pyspark.pandas as ps
import matplotlib.pyplot as plt
from datetime import datetime

# Spark Oturumunu Başlatma
spark = SparkSession.builder \
    .appName("E-Ticaret Data Analiz") \
    .config("spark.driver.memory", "8g") \
    .getOrCreate()

# CSV oku
df = spark.read.csv('C:/Users/Casper/Desktop/CaseBigdata/dataset/sales_data.csv', header=True, inferSchema=True)

# Created/Modified files :
print("Files created/modified:")
print("1. sales_analysis_report.csv")
print("2. category_sales_trends.png")
print("3. user_segmentation.csv")

# 1. Veri İşleme ve Optimizasyon
def process_data(df):
    # timestamp to datetime
    df = df.withColumn("timestamp", to_timestamp(col("timestamp")))

    # eksik değerler
    df = df.na.drop()

    # optimize data types
    df = df.select(
        col("order_id").cast("integer"),
        col("user_id").cast("integer"),
        col("product_id").cast("integer"),
        col("quantity").cast("integer"),
        col("price").cast("double"),
        col("timestamp"),
        col("category")
    )

    return df

processed_df = process_data(df)

#  analiz
def perform_analysis(df):
    # kategorilere göre satış trendleri (aylık)
    monthly_category_sales = df.groupBy(
        year(col("timestamp")).alias("year"),
        month(col("timestamp")).alias("month"),
        col("category")
    ).agg(
        sum(col("price") * col("quantity")).alias("total_sales")
    ).orderBy("year", "month", "total_sales", ascending=[True, True, False])

    # satışlara göre en iyi kategori
    top_categories = df.groupBy("category") \
        .agg(
            sum(col("price") * col("quantity")).alias("total_sales"),
            count("*").alias("order_count")
        ).orderBy("total_sales", ascending=False)

    # Kullanıcı satın alma davranışı
    user_purchase_summary = df.groupBy("user_id") \
        .agg(
            count("*").alias("total_orders"),
            sum(col("price") * col("quantity")).alias("total_spent"),
            avg(col("price")).alias("avg_order_value")
        )

    # analiz sonuç
    monthly_category_sales.write.csv("monthly_category_sales.csv", mode="overwrite")
    top_categories.write.csv("top_categories.csv", mode="overwrite")
    user_purchase_summary.write.csv("user_purchase_summary.csv", mode="overwrite")

    return monthly_category_sales, top_categories, user_purchase_summary

monthly_sales, top_categories, user_summary = perform_analysis(processed_df)

# Gelişmiş analiz - User Segmentation
def user_segmentation(user_summary):
    # Pandas dönüştürme
    user_segments = user_summary.toPandas()

    # Kullanıcılara göre Toplam harcama ve sipariş sıklığı
    def categorize_user(row):
        if row['total_spent'] > 5000 and row['total_orders'] > 10:
            return 'High-Value'
        elif row['total_spent'] > 1000 and row['total_orders'] > 5:
            return 'Medium-Value'
        else:
            return 'Low-Value'

    user_segments['user_segment'] = user_segments.apply(categorize_user, axis=1)

    # save user segments
    user_segments.to_csv('user_segmentation.csv', index=False)

    return user_segments

segmented_users = user_segmentation(user_summary)

# ürün önerme
def product_recommendation(df):
    # sık satın alınanlar
    product_pairs = df.groupBy("user_id") \
        .agg(collect_list("product_id").alias("products"))

    # eşleşme matrisi
    def find_recommendations(products):
        recommendations = {}
        for i in range(len(products)):
            for j in range(i+1, len(products)):
                pair = tuple(sorted([products[i], products[j]]))
                recommendations[pair] = recommendations.get(pair, 0) + 1
        return recommendations

    # mantıksal tavsiye
    recommendations_df = product_pairs.rdd \
        .map(lambda x: find_recommendations(x.products)) \
        .collect()

    return recommendations_df

recommendations = product_recommendation(processed_df)

# Satış trendleri kategorik görselleştirme
def visualize_category_sales(monthly_sales):
    # pandas a çevirme
    sales_trend = monthly_sales.toPandas()

    plt.figure(figsize=(12,6))
    for category in sales_trend['category'].unique():
        category_data = sales_trend[sales_trend['category'] == category]
        plt.plot(
            category_data['month'],
            category_data['total_sales'],
            label=category
        )

    plt.title('Monthly Sales Trends by Category')
    plt.xlabel('Month')
    plt.ylabel('Total Sales')
    plt.legend()
    plt.savefig('category_sales_trends.png')
    plt.close()

visualize_category_sales(monthly_category_sales)

# Stop Spark Session
spark.stop()
