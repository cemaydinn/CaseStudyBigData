import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from unicodedata import category

# Yüklenen CSV dosyalarını okunması
user_interactions = pd.read_csv('C:/Users/Casper/Desktop/CaseBigdata/dataset/user_interactions.csv')
orders = pd.read_csv('C:/Users/Casper/Desktop/CaseBigdata/dataset/sales_data.csv')

# Veri kümeleri hakkında ilk bilgileri yazdırma
print("User Interactions Dataset:")
print(user_interactions.info())
print("\nOrders Dataset:")
print(orders.info())

# Veri türü optimizasyonu
user_interactions['timestamp'] = pd.to_datetime(user_interactions['timestamp'])
orders['timestamp'] = pd.to_datetime(orders['timestamp'])

# device_type'ı kategorik dönüştürülmesi
user_interactions['device_type'] = user_interactions['device_type'].astype('category')
orders['category'] = orders['category'].astype('category')

# Eksik değerlerin kontrolü
print("\nMissing Values:")
print(user_interactions.isnull().sum())
print(orders.isnull().sum())

# Kullanıcı başına ortalama oturum süresi
session_duration_stats = user_interactions.groupby('user_id')['session_duration'].agg(['mean', 'std'])
print("\nSession Duration Statistics:")
print(session_duration_stats)

# En çok etkileşim alan ilk 10 sayfa
page_engagement = user_interactions['page_id'].value_counts().head(10)
print("\nTop 10 Most Engaged Pages:")
print(page_engagement)

# Kategori trendlerine göre satışlar
monthly_sales = orders.groupby([pd.Grouper(key='timestamp', freq='M'), 'category'])['price'].sum().unstack()
print("\nMonthly Sales by Category:")
print(monthly_sales)

# Cihaz tipine göre dönüşüm oranları
def calculate_conversion_rate(df_interactions, df_orders):
    # Cihaz türüne göre etkileşimlerdeki ve siparişlerdeki benzersiz kullanıcıları saydırma
    interactions_by_device = df_interactions.groupby('device_type')['user_id'].nunique()
    orders_by_device = df_orders.groupby(pd.Grouper(key='timestamp', freq='D'))['user_id'].nunique()

    conversion_rates = (orders_by_device / interactions_by_device) * 100
    return conversion_rates

conversion_rates = calculate_conversion_rate(user_interactions, orders)
print("\nConversion Rates by Device Type:")
print(conversion_rates)

# Kullanıcıların satın almadan önce ortalama sayfa görüntüleme sayısı
def avg_page_views_before_purchase(df_interactions, df_orders):
    # Etkileşimleri ve siparişleri birleştirme
    merged_data = pd.merge(df_interactions, df_orders, on='user_id')
    page_views = merged_data.groupby('user_id')['page_id'].count()
    return page_views.mean()

avg_page_views = avg_page_views_before_purchase(user_interactions, orders)
print("\nAverage Page Views Before Purchase:", avg_page_views)

#İleri Seviye Analiz

# Kullanıcı Segmentation using K-means
def segment_users(df_interactions, df_orders):
    # Kümeleme için özellikleri hazırlama
    user_features = df_interactions.groupby('user_id').agg({
        'session_duration': 'mean',
        'page_id': 'count'
    }).reset_index()

    # Siparişle ilgili özellikler ekleyin
    order_features = df_orders.groupby('user_id').agg({
        'price': ['mean', 'sum'],
        'order_id': 'count'
    }).reset_index()
    order_features.columns = ['user_id', 'avg_order_value', 'total_spend', 'order_count']

    # Merge özellik
    user_segments = pd.merge(user_features, order_features, on='user_id')

    # Scale
    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(user_segments.drop('user_id', axis=1))

    #  K-means clustering
    kmeans = KMeans(n_clusters=4, random_state=42)
    user_segments['segment'] = kmeans.fit_predict(scaled_features)

    # Visualization of User Segments
    plt.figure(figsize=(15, 10))

    # Segment Distribution
    plt.subplot(2, 2, 1)
    segment_counts = user_segments['segment'].value_counts()
    segment_counts.plot(kind='pie', autopct='%1.1f%%',
                        labels=[f'Segment {i}' for i in segment_counts.index])
    plt.title('User Segment Distribution')

    # Scatter plot of Total Spend vs Average Order Value
    plt.subplot(2, 2, 2)
    scatter = plt.scatter(user_segments['avg_order_value'],
                          user_segments['total_spend'],
                          c=user_segments['segment'],
                          cmap='viridis')
    plt.title('User Segments: Spend Analysis')
    plt.xlabel('Average Order Value')
    plt.ylabel('Total Spend')
    plt.colorbar(scatter, label='Segment')

    # Box plot of Session Duration by Segment
    plt.subplot(2, 2, 3)
    sns.boxplot(x='segment', y='session_duration', data=user_segments)
    plt.title('Session Duration by Segment')
    plt.xlabel('Segment')
    plt.ylabel('Average Session Duration')

    # Bar plot of Order Count by Segment
    plt.subplot(2, 2, 4)
    segment_order_counts = user_segments.groupby('segment')['order_count'].mean()
    segment_order_counts.plot(kind='bar')
    plt.title('Average Order Count by Segment')
    plt.xlabel('Segment')
    plt.ylabel('Average Order Count')

    plt.tight_layout()
    plt.show()

    return user_segments

user_segments = segment_users(user_interactions, orders)
print("\nUser Segments:")
print(user_segments['segment'].value_counts())

# Basit Ürün Önerme
def simple_product_recommendation(df_orders, top_n=5):
    # Her kategoride en sık satın alınan ürünlere göre tavsiye
    category_top_products = df_orders.groupby('category')['product_id'].agg(lambda x: x.value_counts().index[0])

    # Ürün Önerilerinin Görselleştirilmesi
    plt.figure(figsize=(12, 6))

    # Kategoriye göre ürün sayıları grafiği
    category_product_counts = df_orders.groupby('category')['product_id'].nunique()
    category_product_counts.plot(kind='bar')
    plt.title('Kategoriye Göre Benzersiz Ürün Sayısı')
    plt.xlabel('Category')
    plt.ylabel('Benzersiz Ürün Sayısı')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

    return category_top_products

recommendations = simple_product_recommendation(orders)
print("\nProduct Recommendations by Category:")
print(recommendations)

# Satış Tahminleri (Aylık Ortalama)
def sales_forecast(df_orders, category, periods=30):
    monthly_sales = df_orders[df_orders['category'] == category].groupby(pd.Grouper(key='timestamp', freq='M'))['price'].sum()
    forecast = monthly_sales.rolling(window=3).mean().tail(periods)

    # Visualization of Sales Forecast
    plt.figure(figsize=(15, 10))

    # Subplot 1: Historical Monthly Sales
    plt.subplot(2, 1, 1)
    monthly_sales.plot(kind='line', marker='o')
    plt.title(f'Monthly Sales for {category} Category')
    plt.xlabel('Month')
    plt.ylabel('Total Sales')

    # Subplot 2: Sales Forecast
    plt.subplot(2, 1, 2)
    forecast.plot(kind='line', marker='o', color='red')
    plt.title(f'Sales Forecast for {category} Category')
    plt.xlabel('Month')
    plt.ylabel('Forecasted Sales')

    plt.tight_layout()
    plt.show()

    return forecast

Clothing = sales_forecast(orders, 'Clothing')
print("\nClothing Sales Forecast:")
print(Clothing)

