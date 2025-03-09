import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
import folium
import streamlit_folium as st_folium
import numpy as np
from folium.plugins import HeatMap
from deep_translator import GoogleTranslator
from sklearn.feature_extraction.text import CountVectorizer

# Load data
@st.cache_data
def load_data():
    df = pd.read_csv("dashboard/main_data.csv")
    return df

order = load_data()

st.title("🛒 Brazilian E-Commerce | Dashboard")
st.sidebar.header("🧭 Navigasi")
option = st.sidebar.selectbox("Pilih Aspek Analisis", 
                             ["🚚 Pengiriman", "👤 Pelanggan", "🛍️ Produk dan Penjualan", "💳 Pembayaran", "📈 Analisis RFM"])

# CSS untuk menambahkan garis vertikal antar kolom
st.markdown("""
    <style>
        .column-divider {
            height: 100%;
            border-left: 5px solid #ddd;
            margin: 0 10px;
        }
    </style>
""", unsafe_allow_html=True)

# Pastikan semua kolom waktu dalam format datetime
order["order_purchase_timestamp"] = pd.to_datetime(order["order_purchase_timestamp"])
order["order_delivered_customer_date"] = pd.to_datetime(order["order_delivered_customer_date"])

# FILTER
## === Filter Rentang Tanggal ===
min_date = order["order_purchase_timestamp"].min().date()
max_date = order["order_purchase_timestamp"].max().date()
selected_date = st.sidebar.slider("Pilih Rentang Tanggal", min_value=min_date, max_value=max_date, 
                                value=(min_date, max_date))
order = order[(order["order_purchase_timestamp"].dt.date >= selected_date[0]) & 
            (order["order_purchase_timestamp"].dt.date <= selected_date[1])]

# === Filter Kategori Produk ===
product_category = st.sidebar.multiselect("Pilih Kategori Produk", order["product_category_name"].unique())
if product_category:
    order = order[order["product_category_name"].isin(product_category)]
else:
    order = order.copy()

# === Filter Metode Pembayaran ===
payment_type = st.sidebar.multiselect("Pilih Metode Pembayaran", order["payment_type"].unique())
if payment_type:
    order = order[order["payment_type"].isin(payment_type)]
else:
    order = order.copy()

# === Filter Status Pesanan ===
order_status = st.sidebar.multiselect("Pilih Status Pesanan", order["order_status"].unique())
if order_status:
    order = order[order["order_status"].isin(order_status)]
else:
    order = order.copy()

if option == "🚚 Pengiriman":    
    st.header("Analisis Pengiriman")
    st.write("Menampilkan analisis keterlambatan pengiriman, distribusi wilayah, dan hubungan dengan rating pelanggan.")
    
    tab1, tab2 = st.tabs(['Waktu Pengiriman', 'Keterlambatan Pengiriman'])
    with tab1:
        col1, col2, col3 = st.columns(3)
        st.subheader("Persebaran Waktu Pengiriman")
        
        with col1:
            with st.container():
                # Hitung rata-rata waktu pengiriman
                average_time = (order["order_delivered_customer_date"] - order["order_purchase_timestamp"]).dt.days.mean()
                st.metric(label="Rata-rata Waktu Pengiriman", value=f"{average_time:.2f} hari")
        
        with col2:
            with st.container():
                # Hitung jumlah total pesanan
                total_orders = order["order_id"].nunique()
                st.metric(label="Total Pesanan", value=f"{total_orders:,}")
        
        with col3:
            with st.container():
                # Hitung rata-rata nilai pesanan
                average_order_value = order["payment_value"].mean()
                st.metric(label="Rata-rata Nilai Pesanan", value=f"{average_order_value:,.2f} RBL")

        with st.container():
                
            ## Visualisasi waktu pengiriman
            # Buat kolom durasi pengiriman dalam hari
            order["delivery_duration"] = (order["order_delivered_customer_date"] - order["order_purchase_timestamp"]).dt.days

            # Kelompokkan data berdasarkan latitude & longitude, lalu hitung rata-rata durasi pengiriman
            customer_geo_delivery = order.groupby(["customer_lat", "customer_lng"])["delivery_duration"].mean().reset_index()

            # Buat peta dengan pusat di Brasil
            m = folium.Map(location=[-14.2350, -51.9253], zoom_start=5, control_scale=True)  # Brasil sebagai pusat peta

            # Tambahkan heatmap berdasarkan rata-rata waktu pengiriman di setiap lokasi
            HeatMap(
                data=customer_geo_delivery[['customer_lat', 'customer_lng', 'delivery_duration']].values, 
                radius=5, blur=12, max_zoom=5,
                gradient={0.2: "blue", 0.4: "green", 0.6: "yellow", 0.8: "orange", 1: "red"}  # Custom warna
            ).add_to(m)

            # Tampilkan peta
            st_folium.folium_static(m)

            with st.expander("ℹ️ Keterangan Peta"):
                st.write("Peta ini menunjukkan rata-rata waktu pengiriman berdasarkan lokasi pelanggan. Area dengan warna lebih hijau menunjukkan waktu pengiriman yang lebih cepat, sedangkan area dengan warna lebih biru atau putih menunjukkan waktu pengiriman yang lebih lama. Pola ini dapat membantu memahami efektivitas logistik di berbagai wilayah.")

    with tab2:
        col1, col2, col3 = st.columns([1, 0.05, 1])
        with col1:
            # Gunakan st.container() dengan shadow melalui CSS
            with st.container():
                # Hitung persentase keterlambatan
                late_orders = order[order["order_delivered_customer_date"] > order["order_estimated_delivery_date"]].shape[0]
                total_orders = order.shape[0]
                late_percentage = (late_orders / total_orders) * 100

                st.metric(label="Persentase Keterlambatan", value=f"{late_percentage:.2f}%")


            with st.container():
                # Konversi kolom tanggal ke datetime terlebih dahulu
                order["order_delivered_customer_date"] = pd.to_datetime(order["order_delivered_customer_date"], errors='coerce')
                order["order_estimated_delivery_date"] = pd.to_datetime(order["order_estimated_delivery_date"], errors='coerce')

                # Buat kolom keterlambatan
                order["late_delivery"] = order["order_delivered_customer_date"] > order["order_estimated_delivery_date"]

                # Hitung total pesanan dan pesanan terlambat per seller
                seller_late = order.groupby("seller_id").agg(
                    total_orders=("order_id", "count"),
                    late_orders=("late_delivery", "sum")
                ).reset_index()

                # Hitung persentase keterlambatan
                seller_late["late_percentage"] = (seller_late["late_orders"] / seller_late["total_orders"]) * 100

                # Filter hanya seller dengan jumlah order tinggi (misalnya lebih dari 100 order)
                top_sellers = seller_late[seller_late["total_orders"] > 100].sort_values("late_percentage", ascending=False).head(5)

                # Tampilkan di Streamlit
                st.subheader("Top 5 Seller dengan Keterlambatan Pengiriman Tertinggi")

                # Plot Bar Chart di Streamlit
                fig, ax = plt.subplots(figsize=(6, 9))
                sns.barplot(x="late_percentage", y="seller_id", data=top_sellers, palette="Reds_r", ax=ax)

                # Tambahkan label
                ax.set_xlabel("Persentase Keterlambatan (%)")
                ax.set_ylabel("Seller ID")
                ax.set_title("Top 5 Seller dengan Keterlambatan Pengiriman Tertinggi")
                ax.grid(axis="x", linestyle="--", alpha=0.7)

                # Tampilkan plot di Streamlit
                st.pyplot(fig)

            with st.container():
                order['delay_days'] = (order['order_delivered_customer_date'] - order['order_estimated_delivery_date']).dt.days
                fig, ax = plt.subplots()
                st.subheader("Distribusi Keterlambatan")
                st.set_option('deprecation.showPyplotGlobalUse', False)
                sns.histplot(order["delay_days"], bins=30, kde=True)
                st.pyplot()
        
        with col2:
            st.markdown('<div class="column-divider"></div>', unsafe_allow_html=True)

        with col3:
            with st.container():
                # Konversi kolom tanggal ke datetime terlebih dahulu
                order["order_delivered_customer_date"] = pd.to_datetime(order["order_delivered_customer_date"], errors='coerce')
                order["order_purchase_timestamp"] = pd.to_datetime(order["order_purchase_timestamp"], errors='coerce')

                # Hitung durasi pengiriman
                order["delivery_duration"] = (order["order_delivered_customer_date"] - order["order_purchase_timestamp"]).dt.days

                # Hitung rata-rata durasi pengiriman berdasarkan review score
                review_delivery_avg = order.groupby("review_score")["delivery_duration"].mean()

                # Tampilkan di Streamlit
                st.subheader("Rata-rata Durasi Pengiriman untuk Setiap Review Score")

                # Plot Bar Chart di Streamlit
                fig, ax = plt.subplots(figsize=(10, 5))
                sns.barplot(x=review_delivery_avg.index, y=review_delivery_avg.values, palette="coolwarm", ax=ax)

                # Tambahkan label
                ax.set_xlabel("Review Score")
                ax.set_ylabel("Rata-rata Durasi Pengiriman (Hari)")
                ax.set_title("Rata-rata Durasi Pengiriman untuk Setiap Review Score")
                ax.grid(axis='y', linestyle='--', alpha=0.7)

                # Tampilkan plot di Streamlit
                st.pyplot(fig)
            
            with st.container():
                # Konversi kolom tanggal ke datetime terlebih dahulu
                order["order_delivered_customer_date"] = pd.to_datetime(order["order_delivered_customer_date"], errors='coerce')

                # Ambil hari dari tanggal pengiriman
                order["delivery_day"] = order["order_delivered_customer_date"].dt.day_name()

                # Hitung jumlah keterlambatan per hari
                late_by_day = order.groupby("delivery_day").agg(
                    total_orders=("order_id", "count"),
                    late_orders=("late_delivery", "sum")
                ).reset_index()

                # Hitung persentase keterlambatan
                late_by_day["late_percentage"] = (late_by_day["late_orders"] / late_by_day["total_orders"]) * 100

                # Urutkan berdasarkan urutan hari (bukan alfabet)
                order_of_days = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
                late_by_day["delivery_day"] = pd.Categorical(late_by_day["delivery_day"], categories=order_of_days, ordered=True)
                late_by_day = late_by_day.sort_values("delivery_day")

                # Tampilkan di Streamlit
                st.subheader("Persentase Keterlambatan Pengiriman per Hari")

                # Plot Bar Chart di Streamlit
                fig, ax = plt.subplots(figsize=(10, 5))
                sns.barplot(x="delivery_day", y="late_percentage", data=late_by_day, palette="Blues_r", ax=ax)

                # Tambahkan label
                ax.set_xlabel("Hari dalam Seminggu")
                ax.set_ylabel("Persentase Keterlambatan (%)")
                ax.set_title("Persentase Keterlambatan Pengiriman per Hari")
                ax.grid(axis="y", linestyle="--", alpha=0.7)

                # Tampilkan plot di Streamlit
                st.pyplot(fig)

            with st.container():
                # Hitung keterlambatan
                order["late_delivery"] = order["order_delivered_customer_date"] > order["order_estimated_delivery_date"]

                # Buat dataframe agregasi rata-rata rating berdasarkan keterlambatan
                late_review_avg = order.groupby("late_delivery")["review_score"].mean().reset_index()

                # Tampilkan di Streamlit
                st.subheader("Pengaruh Keterlambatan terhadap Rating Ulasan")

                # Plot Bar Chart di Streamlit
                fig, ax = plt.subplots(figsize=(8, 5))
                sns.barplot(x=late_review_avg["late_delivery"], y=late_review_avg["review_score"], palette="coolwarm", ax=ax)

                # Tambahkan label
                ax.set_xticks([0, 1])
                ax.set_xticklabels(["Tepat Waktu", "Terlambat"])
                ax.set_xlabel("Status Pengiriman")
                ax.set_ylabel("Rata-rata Review Score")
                ax.set_title("Pengaruh Keterlambatan terhadap Rating Ulasan")
                ax.set_ylim(0, 5)  # Batas rating 1-5
                ax.grid(axis="y", linestyle="--", alpha=0.7)

                # Tampilkan plot di Streamlit
                st.pyplot(fig)

        with st.container():
            # Buat kategori ongkos kirim
            order["freight_category"] = pd.cut(order["freight_value"], bins=[0, 10, 20, 50, 100, 500], labels=["0-10", "10-20", "20-50", "50-100", "100+"])

            # Hitung keterlambatan berdasarkan kategori ongkos kirim
            freight_late = order.groupby("freight_category").agg(
                total_orders=("order_id", "count"),
                late_orders=("late_delivery", "sum")
            ).reset_index()

            # Hitung persentase keterlambatan
            freight_late["late_percentage"] = (freight_late["late_orders"] / freight_late["total_orders"]) * 100

            # Tampilkan di Streamlit
            st.subheader("Hubungan antara Ongkos Kirim dan Tingkat Keterlambatan")

            # Plot Line Chart di Streamlit
            fig, ax = plt.subplots(figsize=(10, 5))
            sns.lineplot(x=freight_late["freight_category"], y=freight_late["late_percentage"], marker="o", linestyle="-", color="red", ax=ax)

            # Tambahkan label
            ax.set_xlabel("Kategori Ongkos Kirim (BRL)")
            ax.set_ylabel("Persentase Keterlambatan (%)")
            ax.set_title("Hubungan antara Ongkos Kirim dan Tingkat Keterlambatan")
            ax.grid(axis="y", linestyle="--", alpha=0.7)

            # Tampilkan plot di Streamlit
            st.pyplot(fig)



elif option == "👤 Pelanggan":
    st.header("Analisis Pelanggan")
    st.write("Menampilkan kepuasan pelanggan berdasarkan ulasan dan tren transaksi.")
    tab1, tab2 = st.tabs(['Transaksi', 'Kepuasan Pelanggan'])
    with tab1:
        col1, col2, col3 = st.columns(3)
        with col1:
            with st.container():
                # Hitung jumlah transaksi per pelanggan
                customer_orders = order.groupby("customer_unique_id")["order_id"].count().reset_index()
                customer_orders.columns = ["customer_unique_id", "order_count"]

                # Hitung median jumlah order per pelanggan
                median_orders = customer_orders["order_count"].median()
                st.metric(label="Median Order per Pelanggan", value=f"{median_orders:.2f}")
        
        with col2:
            with st.container():
                # Urutkan data berdasarkan waktu pembelian
                order = order.sort_values("order_purchase_timestamp")

                # Identifikasi pelanggan pertama kali melakukan pembelian
                order["is_new_customer"] = ~order.duplicated(subset=["customer_unique_id"], keep="first")

                # Kelompokkan data berdasarkan bulan
                order["purchase_month"] = order["order_purchase_timestamp"].dt.to_period("M")

                # Hitung jumlah pelanggan baru dan lama setiap bulan
                customer_trend = order.groupby(["purchase_month", "is_new_customer"])["customer_unique_id"].nunique().unstack().reset_index()
                customer_trend.columns = ["purchase_month", "existing_customers", "new_customers"]

                # Hitung total pelanggan baru dan lama
                total_new_customers = customer_trend["new_customers"].sum()
                total_existing_customers = customer_trend["existing_customers"].sum()

                st.metric(label="Jumlah Pelanggan Baru", value=f"{total_new_customers:,}")
        
        with col3:
            with st.container():
                st.metric(label="Jumlah Pelanggan Lama", value=f"{total_existing_customers:,}")

        with st.container():
            # Kelompokkan data berdasarkan latitude & longitude, lalu hitung jumlah transaksi di setiap lokasi
            customer_geo = order.groupby(["customer_lat", "customer_lng"]).size().reset_index(name="transaction_count")

            # Buat peta dengan pusat di Brasil
            m = folium.Map(location=[-14.2350, -51.9253], zoom_start=5, control_scale=True)

            # Tambahkan heatmap berdasarkan jumlah transaksi di setiap lokasi
            HeatMap(data=customer_geo[['customer_lat', 'customer_lng', 'transaction_count']].values, 
                    radius=10, blur=15, max_zoom=1).add_to(m)

            # Tambahkan Judul ke Peta
            title_html = '''
                <h3 align="center" style="font-size:16px"><b>Peta Sebaran Pelanggan Berdasarkan Jumlah Transaksi</b></h3>
            '''
            m.get_root().html.add_child(folium.Element(title_html))

            # Tambahkan Legend (Keterangan)
            legend_html = '''
            <div style="
                position: fixed;
                bottom: 20px; left: 20px; width: 300px; height: 100px; 
                background-color: white; z-index:9999; font-size:14px;
                border-radius: 5px; padding: 10px; box-shadow: 2px 2px 5px rgba(0,0,0,0.3);">
                <b>Legenda</b><br>
                🔴 = Kepadatan transaksi tinggi<br>
                🟡 = Kepadatan transaksi sedang<br>
                🔵 = Kepadatan transaksi rendah
            </div>
            '''
            m.get_root().html.add_child(folium.Element(legend_html))

            # Tampilkan di Streamlit
            st.subheader("Peta Sebaran Pelanggan Berdasarkan Jumlah Transaksi")
            st_folium.folium_static(m)

    with tab2:
        col1, col2 = st.columns(2)
        with col1:
            with st.container():
                # Konversi kolom tanggal ke datetime terlebih dahulu
                order["review_creation_date"] = pd.to_datetime(order["review_creation_date"], errors='coerce')

                # Buat kolom hari dari tanggal review
                order['review_creation_day'] = order['review_creation_date'].dt.date

                # Hitung rata-rata review score per hari
                review_trend_daily = order.groupby('review_creation_day')['review_score'].mean()

                # Hitung rata-rata keseluruhan
                overall_mean = np.mean(order['review_score'])

                # Tampilkan di Streamlit
                st.subheader("Tren Kepuasan Pelanggan Berdasarkan Ulasan (Per Hari)")

                # Plot Line Chart di Streamlit
                fig, ax = plt.subplots(figsize=(15, 6))
                sns.lineplot(x=review_trend_daily.index, y=review_trend_daily.values, marker='o', color='blue', label='Mean Review Score per Day', ax=ax)

                # Tambahkan garis rata-rata keseluruhan
                ax.axhline(y=overall_mean, color='red', linestyle='dashed', label='Overall Mean Review Score')

                # Tambahkan label
                ax.set_xlabel("Tanggal")
                ax.set_ylabel("Review Score")
                ax.set_title("Tren Kepuasan Pelanggan Berdasarkan Ulasan (Per Hari)")
                ax.set_xticklabels(review_trend_daily.index, rotation=45)
                ax.legend()
                ax.grid(axis='y', linestyle='--', alpha=0.7)

                # Tampilkan plot di Streamlit
                st.pyplot(fig)

        with col2:
            with st.container():
                # Hitung rata-rata rating pelanggan
                average_rating = order["review_score"].mean()

                # Tampilkan metrik di Streamlit
                st.metric(label="Rata-rata Rating Pelanggan", value=f"{average_rating:.2f}")

        with st.container():
            # Pastikan ada ulasan yang tidak kosong
            if "review_comment_message" in order.columns and order["review_comment_message"].dropna().shape[0] > 0:
                # Vectorizer untuk mengambil kata-kata paling sering muncul
                vectorizer = CountVectorizer(max_features=100)
                X = vectorizer.fit_transform(order["review_comment_message"].dropna())

                # Gabungkan kata-kata paling sering muncul
                text_reviews = " ".join(vectorizer.get_feature_names_out())

                # Terjemahkan teks ke dalam bahasa Inggris
                translator = GoogleTranslator(source="auto", target="en")
                translated_text = translator.translate(text_reviews)

                # Buat WordCloud
                wordcloud = WordCloud(width=800, height=400, background_color="white", colormap="viridis", max_words=200).generate(translated_text)

                # Tampilkan di Streamlit
                st.subheader("WordCloud dari Ulasan Pelanggan (Diterjemahkan ke Bahasa Inggris)")

                fig, ax = plt.subplots(figsize=(10, 5))
                ax.imshow(wordcloud, interpolation="bilinear")
                ax.axis("off")
                ax.set_title("WordCloud of Customer Reviews (Translated to English)")

                st.pyplot(fig)
            else:
                st.warning("Tidak ada data review yang tersedia untuk WordCloud.")


elif option == "🛍️ Produk dan Penjualan":
    st.header("Analisis Produk dan Penjualan")
    st.write("Menampilkan produk terlaris dan pola pembelian pelanggan.")
    
    # Hitung jumlah total produk yang terjual berdasarkan kategori
    product_sales = order["product_category_name"].value_counts()

    # Ambil 10 kategori dengan penjualan tertinggi
    top_product_sales = product_sales.nlargest(10)

    # Ambil 10 kategori dengan penjualan terendah
    bottom_product_sales = product_sales.nsmallest(10)

    col1, col2 = st.columns(2)

    # Visualisasi: Kategori dengan Penjualan Tertinggi
    with col1:
        st.subheader("Penjualan Tertinggi")
        fig, ax = plt.subplots(figsize=(6, 6))
        sns.barplot(x=top_product_sales.values, y=top_product_sales.index, palette="crest", ax=ax)
        ax.set_xlabel("Jumlah Produk Terjual")
        ax.set_ylabel("Kategori Produk")
        ax.set_title("10 Kategori Produk dengan Penjualan Tertinggi")
        ax.grid(axis='x', linestyle='--', alpha=0.7)
        st.pyplot(fig)

    # Visualisasi: Kategori dengan Penjualan Terendah
    with col2:
        st.subheader("Penjualan Terendah")
        fig, ax = plt.subplots(figsize=(6, 6))
        sns.barplot(x=bottom_product_sales.values, y=bottom_product_sales.index, palette="flare", ax=ax)
        ax.set_xlabel("Jumlah Produk Terjual")
        ax.set_ylabel("Kategori Produk")
        ax.set_title("10 Kategori Produk dengan Penjualan Terendah")
        ax.grid(axis='x', linestyle='--', alpha=0.7)
        st.pyplot(fig)

    # Hitung rata-rata harga produk
    average_price = order["price"].mean()

    # Buat dua kolom di Streamlit
    st.subheader("Distribusi Harga Produk & Rata-rata Harga Produk")
    col1, col2 = st.columns([2, 1])  # Kolom kiri lebih besar untuk plot

    # Visualisasi: Distribusi Harga Produk
    with col1:
        st.subheader("Distribusi Harga Produk")
        fig, ax = plt.subplots(figsize=(8, 6))
        sns.histplot(order["price"], bins=100, kde=True, color="blue", ax=ax)
        ax.set_xlabel("Harga Produk (BRL)")
        ax.set_ylabel("Jumlah Produk Terjual")
        ax.set_title("Distribusi Harga Produk yang Paling Sering Dibeli")
        ax.grid(axis='y', linestyle='--', alpha=0.7)
        st.pyplot(fig)

    # Metrik Rata-rata Harga Produk
    with col2:
        st.subheader("Rata-rata Harga Produk")
        st.metric(label="Rata-rata Harga Produk", value=f"{average_price:.2f} RBL")

        with st.container():
            # Hitung jumlah produk terjual per hari
            daily_product_sales = order.groupby(order["order_purchase_timestamp"].dt.date)["product_id"].count()

            # Hitung rata-rata produk terjual per hari
            average_daily_sales = daily_product_sales.mean()

            # Tampilkan metrik di Streamlit
            st.metric(label="Rata-rata Produk Terjual per Hari", value=f"{average_daily_sales:.2f}")


    # Ekstrak jam dan hari dari waktu pembelian
    order["purchase_hour"] = order["order_purchase_timestamp"].dt.hour
    order["purchase_day"] = order["order_purchase_timestamp"].dt.day_name()

    # Urutkan hari dalam seminggu
    order_of_days = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
    order["purchase_day"] = pd.Categorical(order["purchase_day"], categories=order_of_days, ordered=True)

    # Hitung jumlah pesanan per jam
    hourly_orders = order.groupby("purchase_hour")["order_id"].count().reset_index()

    # Hitung jumlah pesanan per hari
    daily_orders = order.groupby("purchase_day")["order_id"].count().reset_index()

    # Buat dua kolom di Streamlit
    st.subheader("Pola Pembelian Pelanggan Berdasarkan Waktu")
    col1, col2 = st.columns(2)

    # Plot Line Chart (Pesanan per Jam)
    with col1:
        st.subheader("Pesanan per Jam")
        fig, ax = plt.subplots(figsize=(6, 5))
        sns.lineplot(x="purchase_hour", y="order_id", data=hourly_orders, marker="o", linestyle="-", color="blue", ax=ax)
        ax.set_xlabel("Jam dalam Sehari")
        ax.set_ylabel("Jumlah Pesanan")
        ax.set_title("Pola Pembelian Pelanggan Berdasarkan Jam dalam Sehari")
        ax.grid(axis="y", linestyle="--", alpha=0.7)
        ax.set_xticks(range(0, 24))  # Pastikan semua jam tampil
        st.pyplot(fig)

    # Plot Bar Chart (Pesanan per Hari)
    with col2:
        st.subheader("Pesanan per Hari")
        fig, ax = plt.subplots(figsize=(6, 5))
        sns.barplot(x="purchase_day", y="order_id", data=daily_orders, palette="Blues_r", ax=ax)
        ax.set_xlabel("Hari dalam Seminggu")
        ax.set_ylabel("Jumlah Pesanan")
        ax.set_title("Pola Pembelian Pelanggan Berdasarkan Hari dalam Seminggu")
        ax.grid(axis="y", linestyle="--", alpha=0.7)
        st.pyplot(fig)



elif option == "💳 Pembayaran":
    st.header("Analisis Pembayaran")
    st.write("Menampilkan metode pembayaran paling sering digunakan.")

    order["order_approved_at"] = pd.to_datetime(order["order_approved_at"], errors='coerce')
    # Hitung durasi proses pembayaran dalam menit
    order["payment_processing_time"] = (order["order_approved_at"] - order["order_purchase_timestamp"]).dt.total_seconds() / 60

    # Hitung rata-rata durasi pembayaran per metode
    payment_time_avg = order.groupby("payment_type")["payment_processing_time"].mean().reset_index()

    # Buat dictionary untuk memastikan metrik tetap ada meskipun ada metode pembayaran yang hilang
    payment_methods = ["credit_card", "boleto", "voucher", "debit_card"]
    avg_times = {method: payment_time_avg[payment_time_avg["payment_type"] == method]["payment_processing_time"].values[0] 
                if method in payment_time_avg["payment_type"].values else 0 
                for method in payment_methods}

    # Buat 4 kolom untuk menampilkan metrik secara bersebelahan
    st.subheader("Rata-rata Waktu Proses Pembayaran Berdasarkan Metode Pembayaran")

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric(label="Credit Card", value=f"{avg_times['credit_card']:.2f} menit")

    with col2:
        st.metric(label="Boleto", value=f"{avg_times['boleto']:.2f} menit")

    with col3:
        st.metric(label="Voucher", value=f"{avg_times['voucher']:.2f} menit")

    with col4:
        st.metric(label="Debit Card", value=f"{avg_times['debit_card']:.2f} menit")


    # Hitung jumlah penggunaan setiap metode pembayaran
    payment_methods = order["payment_type"].value_counts()

    # Buat kolom biner untuk menandai pembatalan pesanan
    order["is_canceled"] = (order["order_status"] == "canceled").astype(int)

    # Hitung jumlah pesanan dan jumlah pembatalan per metode pembayaran
    cancellation_rate = order.groupby("payment_type").agg(
        total_orders=("order_id", "count"),
        canceled_orders=("is_canceled", "sum")
    ).reset_index()

    # Hitung persentase pembatalan
    cancellation_rate["cancellation_rate"] = (cancellation_rate["canceled_orders"] / cancellation_rate["total_orders"]) * 100

    # Buat dua kolom di Streamlit
    st.subheader("Distribusi Metode Pembayaran & Tingkat Pembatalan")

    col1, col2 = st.columns(2)

    # Visualisasi: Bar Chart Metode Pembayaran
    with col1:
        st.subheader("Distribusi Metode Pembayaran")
        fig, ax = plt.subplots(figsize=(6, 5))
        sns.barplot(x=payment_methods.index, y=payment_methods.values, palette="pastel", ax=ax)
        ax.set_xlabel("Metode Pembayaran")
        ax.set_ylabel("Jumlah Penggunaan")
        ax.set_title("Metode Pembayaran yang Paling Sering Digunakan")
        ax.grid(axis='y', linestyle='--', alpha=0.7)
        st.pyplot(fig)

    # Visualisasi: Tingkat Pembatalan per Metode Pembayaran
    with col2:
        st.subheader("Tingkat Pembatalan Pesanan")
        fig, ax = plt.subplots(figsize=(6, 5))
        sns.barplot(x="payment_type", y="cancellation_rate", data=cancellation_rate, palette="coolwarm", ax=ax)
        ax.set_xlabel("Metode Pembayaran")
        ax.set_ylabel("Persentase Pembatalan (%)")
        ax.set_title("Tingkat Pembatalan Berdasarkan Metode Pembayaran")
        ax.grid(axis="y", linestyle="--", alpha=0.7)
        st.pyplot(fig)



elif option == "📈 Analisis RFM":
    st.header("Analisis RFM")
    st.write("Menampilkan segmentasi pelanggan berdasarkan analisis RFM.")

    segment_mapping = {
        "Champions": ["555", "554", "544", "545", "454", "455", "445"],
        "Loyal": ["543", "444", "435", "355", "354", "345", "344", "335"],
        "Potential Loyalist": ["553", "551", "552", "541", "542", "533", "532", "531", "452", "451",
                            "442", "441", "431", "453", "433", "432", "423", "353", "352", "351",
                            "342", "341", "333", "323"],
        "New Customers": ["512", "511", "422", "421", "412", "411", "311"],
        "Promising": ["525", "524", "523", "522", "521", "515", "514", "513", "425", "424", "413",
                    "414", "415", "315", "314", "313"],
        "Need Attention": ["535", "534", "443", "434", "343", "334", "325", "324"],
        "About to Sleep": ["331", "321", "312", "221", "213", "231", "241", "251"],
        "Cannot Lose Them but Losing": ["155", "154", "144", "214", "215", "115", "114", "113"],
        "At Risk": ["255", "254", "245", "244", "253", "252", "243", "242", "235", "234", "225",
                    "224", "153", "152", "145", "143", "142", "135", "134", "133", "125", "124"],
        "Hibernating Customers": ["332", "322", "233", "232", "223", "222", "132", "123", "122",
                                "212", "211"],
        "Losing but Engaged": ["111", "112", "121", "131", "141", "151"],
        "Lost Customers": ["111", "112", "121", "131", "141", "151"]
    }

    # Mapping RFM Score ke Segment
    def map_rfm_segment(rfm_score):
        for segment, scores in segment_mapping.items():
            if rfm_score in scores:
                return segment
        return "Other"

    # Gunakan data order
    rfm_data = order.copy()

    # Konversi order_purchase_timestamp ke datetime
    rfm_data["order_purchase_timestamp"] = pd.to_datetime(rfm_data["order_purchase_timestamp"])

    # Tentukan tanggal referensi (hari terakhir transaksi dalam dataset)
    latest_date = rfm_data["order_purchase_timestamp"].max()

    # Hitung RFM Metrics
    rfm = rfm_data.groupby("customer_unique_id").agg({
        "order_purchase_timestamp": lambda x: (latest_date - x.max()).days,  # Recency: Hari sejak transaksi terakhir
        "order_id": "count",  # Frequency: Jumlah transaksi
        "payment_value": "sum"  # Monetary: Total belanja
    })

    # Ubah nama kolom
    rfm.columns = ["Recency", "Frequency", "Monetary"]

    # Perbaiki binning untuk Frequency dan Monetary
    rfm["R_Score"] = pd.qcut(rfm["Recency"], 5, labels=[5,4,3,2,1])
    rfm["F_Score"] = pd.cut(rfm["Frequency"], bins=[0,1,2,5,10,rfm["Frequency"].max()], labels=[1,2,3,4,5], include_lowest=True)
    rfm["M_Score"] = pd.cut(rfm["Monetary"], bins=[0,50,100,200,500,rfm["Monetary"].max()], labels=[1,2,3,4,5], include_lowest=True)

    # Gabungkan skor RFM untuk segmentasi
    rfm["RFM_Score"] = rfm["R_Score"].astype(str) + rfm["F_Score"].astype(str) + rfm["M_Score"].astype(str)

    # Terapkan segmentasi
    rfm["Segment"] = rfm["RFM_Score"].apply(map_rfm_segment)

    # Hitung jumlah pelanggan per segmen
    segment_counts = rfm["Segment"].value_counts()
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric(label="Total Champions", value=f"{segment_counts.get('Champions', 0):,}")
    with col2:
        st.metric(label="Total Loyal Customers", value=f"{segment_counts.get('Loyal', 0):,}")
    with col3:
        st.metric(label="Total At Risk", value=f"{segment_counts.get('At Risk', 0):,}")
    with col4:
        st.metric(label="Total Lost Customers", value=f"{segment_counts.get('Lost Customers', 0):,}")

    # Buat dua kolom untuk visualisasi
    st.subheader("Distribusi Segmen Pelanggan Berdasarkan RFM")

    col1, col2 = st.columns(2)

    # Visualisasi Bar Chart
    with col1:
        fig, ax = plt.subplots(figsize=(8, 5))
        sns.barplot(x=segment_counts.index, y=segment_counts.values, palette="coolwarm", ax=ax)
        ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right")
        ax.set_xlabel("Customer Segment")
        ax.set_ylabel("Number of Customers")
        ax.set_title("Customer Segmentation based on RFM Analysis")
        ax.grid(axis="y", linestyle="--", alpha=0.7)
        st.pyplot(fig)

    # Visualisasi Pie Chart
    with col2:
        fig, ax = plt.subplots(figsize=(5, 5))
        ax.pie(segment_counts, labels=segment_counts.index, autopct='%1.1f%%', colors=sns.color_palette("coolwarm", len(segment_counts)))
        ax.set_title("Customer Segmentation Distribution")
        st.pyplot(fig)
