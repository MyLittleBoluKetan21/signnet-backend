import pandas as pd
import mysql.connector
import json

# Konfigurasi Database (Sama dengan .env Laravel Anda)
db_config = {
    'host': '127.0.0.1',
    'user': 'root',
    'password': '',
    'database': 'sign_language_db'
}

try:
    print("Mencoba mengambil data dari MySQL...")
    # Tarik data dari MySQL ke DataFrame Pandas
    conn = mysql.connector.connect(**db_config)
    
    # 'datasets' adalah nama tabel default, sesuaikan jika berbeda nanti
    query = "SELECT features, label FROM datasets"
    df_db = pd.read_sql(query, conn)
    conn.close()

    if df_db.empty:
        print("\n[INFO] Koneksi sukses, tetapi tabel 'datasets' di database masih kosong.")
        exit()

    # Parse JSON string kolom 'features' menjadi list kolom f0-f125
    features_list = [json.loads(x) for x in df_db['features']]
    df_features = pd.DataFrame(features_list, columns=[f'f{i}' for i in range(126)])
    df_features['label'] = df_db['label']

    print("\n=== HASIL CEK DATASET ===")
    
    # 1. Cek data 2 tangan (f63 tidak boleh 0 semua)
    dua_tangan = df_features[df_features['f63'] != 0]
    print(f"Total data 2 tangan: {len(dua_tangan)}")

    # 2. Cek sebaran data per label
    print("\nJumlah data per label:")
    print(df_features['label'].value_counts())

    # 3. Cek apakah ada nilai kosong/error
    print("\nCek nilai kosong:")
    print(df_features.isnull().sum().sum())

except Exception as e:
    print(f"\n❌ Gagal mengecek data: {e}")
    print("Pastikan database MySQL (XAMPP) menyala dan tabel 'datasets' sudah di-migrate di Laravel.")