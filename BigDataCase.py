import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor

# 1. Veri İşleme ve Optimizasyon
def load_and_preprocess_data(file_path):
    #  bellek optimize ile CSV okuma
    df = pd.read_csv(file_path,
                     dtype={
                         'order_id': 'int32',
                         'user_id': 'int32',
                         'product_id': 'int32',
                         'quantity': 'int8',
                         'price': 'float32'
                     },
                     parse_dates=['timestamp'])

    # Eksik değerler
    print("Missing Values:\n", df.isnull().sum())

    # Kritik verileri eksik olan tüm satırları kaldırın
    df = df.dropna(subset=['order_id', 'user_id', 'product_id'])

    return df

# Veriseti oku
sales_df = load_and_preprocess_data('C:/Users/Casper/Desktop/CaseBigdata/dataset/sales_data.csv')

# 2. Analiz ve İçgörüler
def perform_analysis(df):
    # Kategorilere Göre Satış Trendleri
    category_sales = df.groupby([pd.Grouper(key='timestamp', freq='M'), 'category'])['price'].sum().unstack()

    # Kategori Satış Trendlerinin Görselleştirilmesi
    plt.figure(figsize=(12, 6))
    category_sales.plot(kind='bar', stacked=True)
    plt.title('Monthly Sales by Category')
    plt.xlabel('Month')
    plt.ylabel('Total Sales')
    plt.tight_layout()
    plt.savefig('category_sales_trends.png')
    plt.close()

    # Kullanıcı Satın Alma Davranışı Analizi
    user_metrics = df.groupby('user_id').agg({
        'order_id': 'count',  # Sipariş sayısı
        'price': ['mean', 'sum'],  # Ortalama ve toplam harcama
        'category': lambda x: x.nunique()  # Kategori çeşitliliği
    })
    user_metrics.columns = ['order_count', 'avg_order_value', 'total_spend', 'category_diversity']

    return user_metrics

# Analiz
user_metrics = perform_analysis(sales_df)

# 3. Gelişmiş Analiz - Kullanıcı Segmentasyonu
def segment_users(user_metrics):
    # Kümeleme için veri hazırlama
    X = StandardScaler().fit_transform(user_metrics)

    # K-means clustering
    kmeans = KMeans(n_clusters=4, random_state=42)
    user_metrics['cluster'] = kmeans.fit_predict(X)

    # Küme görsel
    plt.figure(figsize=(10, 6))
    scatter = plt.scatter(user_metrics['total_spend'],
                          user_metrics['order_count'],
                          c=user_metrics['cluster'],
                          cmap='viridis')
    plt.title('User Segments')
    plt.xlabel('Total Spend')
    plt.ylabel('Number of Orders')
    plt.colorbar(scatter)
    plt.savefig('user_segments.png')
    plt.close()

    return user_metrics

# kullanıcılar
segmented_users = segment_users(user_metrics)

# Satış Tahminleme
def forecast_sales(df):
    # veri hazırlama
    monthly_sales = df.groupby([pd.Grouper(key='timestamp', freq='M'), 'category'])['price'].sum().reset_index()

    # Tahmin için verileri pivotlama
    sales_pivot = monthly_sales.pivot(index='timestamp', columns='category', values='price').fillna(0)

    # Özellikleri ve hedefi hazırlayın
    X = np.arange(len(sales_pivot)).reshape(-1, 1)
    forecasts = {}

    plt.figure(figsize=(15, 8))
    for category in sales_pivot.columns:
        # Random Forest Regressor tahminleme
        model = RandomForestRegressor(n_estimators=100, random_state=42)
        model.fit(X, sales_pivot[category])

        # gelecek aylar tahmin
        future_X = np.append(X, len(X)).reshape(-1, 1)
        forecast = model.predict(future_X)

        forecasts[category] = forecast[-1]

        # Plot ile geçmiş ve tahmini veriler
        plt.plot(future_X, forecast, label=f'{category} Forecast')

    plt.title('Sales Forecast by Category')
    plt.xlabel('Months')
    plt.ylabel('Sales')
    plt.legend()
    plt.savefig('sales_forecast.png')
    plt.close()

    return forecasts

# satış tahminleme
sales_forecast = forecast_sales(sales_df)

# Sonuçlar
print("\nUser Segments Summary:")
print(segmented_users.groupby('cluster').mean())

print("\nSales Forecast for Next Month:")
for category, forecast in sales_forecast.items():
    print(f"{category}: ${forecast:,.2f}")

# Created/Modified files
print("\nGenerated Files:")
generated_files = [
    'category_sales_trends.png',
    'user_segments.png',
    'sales_forecast.png'
]
print(generated_files)