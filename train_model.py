import json
import pandas as pd
import numpy as np
import os
import joblib
import mysql.connector
import requests
import gzip

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder

# TAMBAHAN LIBRARY UNTUK KONVERSI ONNX
from skl2onnx import convert_sklearn
from skl2onnx.common.data_types import FloatTensorType

APP_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(APP_DIR, 'model')
MODEL_PATH = os.path.join(MODEL_DIR, 'rf_model.joblib')
ENCODER_PATH = os.path.join(MODEL_DIR, 'label_encoder.joblib')
META_PATH = os.path.join(MODEL_DIR, 'meta_model.json')
LABELS_PATH = os.path.join(MODEL_DIR, 'labels.json') # <-- TAMBAHAN PATH LABELS
ONNX_PATH = os.path.join(MODEL_DIR, 'rf_model.onnx')

LARAVEL_RECEIVE_URL = os.getenv('APP_URL', 'https://signnet-web-production.up.railway.app') + '/api/sync-model'

db_config = {
    'host': os.getenv('DB_HOST', '127.0.0.1'),
    'user': os.getenv('DB_USERNAME', 'root'),
    'password': os.getenv('DB_PASSWORD', ''),
    'database': os.getenv('DB_DATABASE', 'sign_language_db'),
    'port': int(os.getenv('DB_PORT', 3306))
}

def convert_to_onnx(scikit_model, target_path):
    try:
        initial_type = [('float_input', FloatTensorType([None, 126]))]
        onnx_model = convert_sklearn(scikit_model, initial_types=initial_type, options={'zipmap': False})
        
        # 1. Simpan file asli .onnx
        with open(target_path, "wb") as f:
            f.write(onnx_model.SerializeToString())
            
        # 2. BUAT VERSI KOMPRESI (.onnx.gz) -> Ini yang akan di-download browser
        gz_path = target_path + ".gz"
        with open(target_path, "rb") as f_in:
            with gzip.open(gz_path, "wb") as f_out:
                f_out.writelines(f_in)
                
        print(f"✅ Berhasil mengonversi ke ONNX dan membuat kompresi GZ: {gz_path}")
    except Exception as e:
        print(f"⚠️ Gagal melakukan konversi ONNX: {str(e)}")

def send_assets_to_laravel():
    print("🔄 Mengirim file model (Compressed) dan metadata terbaru ke Laravel frontend...")
    try:
        # Gunakan path versi GZ untuk pengiriman
        ONNX_GZ_PATH = ONNX_PATH + ".gz"
        
        if not os.path.exists(ONNX_GZ_PATH) or not os.path.exists(META_PATH) or not os.path.exists(LABELS_PATH):
            print("⚠️ File model compressed (.gz), metadata, atau labels tidak ditemukan untuk dikirim.")
            return

        print(f"🔗 Menghubungi URL: {LARAVEL_RECEIVE_URL} ...")
        
        # PERUBAHAN: Buka file rf_model.onnx.gz (bukan .onnx mentah)
        with open(ONNX_GZ_PATH, 'rb') as onnx_file, \
             open(META_PATH, 'rb') as json_file, \
             open(LABELS_PATH, 'rb') as labels_file:
            
            files = {
                # Kirim file .gz namun tetap beri nama berekstensi .onnx.gz agar Laravel tahu
                'onnx_model': ('rf_model.onnx.gz', onnx_file, 'application/gzip'),
                'meta_model': ('meta_model.json', json_file, 'application/json'),
                'labels': ('labels.json', labels_file, 'application/json')
            }
            headers = {'Accept': 'application/json'}
            response = requests.post(LARAVEL_RECEIVE_URL, files=files, headers=headers, timeout=25) # Timeout dinaikkan ke 25s untuk data besar
            
            print(f"📥 Respon diterima dengan Status Code: {response.status_code}")
            
            if response.status_code == 200:
                print("🚀 [AUTO-SYNC] Berhasil menyalin model terkompresi, metadata, dan labels ke Laravel!")
            else:
                print(f"⚠️ [AUTO-SYNC] Laravel menolak file. Respon Raw: {response.text}")
                
    except requests.exceptions.Timeout:
        print("❌ [AUTO-SYNC] Timeout! Upload model membutuhkan waktu lebih lama. Cek koneksi Railway.")
    except Exception as e:
        print(f"⚠️ [AUTO-SYNC] Gagal terhubung ke Laravel untuk sinkronisasi file: {str(e)}")

def train():
    files_to_clean = [MODEL_PATH, ENCODER_PATH, META_PATH, LABELS_PATH, ONNX_PATH] # <-- PERUBAHAN: Ikut bersihkan LABELS_PATH
    for file_path in files_to_clean:
        if os.path.exists(file_path):
            try:
                os.remove(file_path)
                print(file_path)
            except Exception as e:
                print(f"⚠️ Gagal menghapus file lama {file_path}: {e}")
    try:
        conn = mysql.connector.connect(**db_config)
        cursor = conn.cursor(dictionary=True)
        cursor.execute("SELECT features, label FROM datasets") 
        rows = cursor.fetchall()
        cursor.close()
        conn.close()
    except Exception as e:
        return {"status": "error", "message": f"Gagal terkoneksi ke database: {str(e)}"}

    if not rows:
        return {"status": "error", "message": "Dataset di database masih kosong"}

    X_list = []
    y_list = []

    for row in rows:
        try:
            raw_features = row['features']
            if isinstance(raw_features, str):
                fitur_array = json.loads(raw_features)
            else:
                fitur_array = raw_features
            
            if isinstance(fitur_array, str):
                fitur_array = json.loads(fitur_array)
            
            if isinstance(fitur_array, list) and len(fitur_array) == 126:
                X_list.append(fitur_array)
                y_list.append(row['label'])
        except Exception as e:
            print(f"⚠️ Melewati baris data yang bermasalah: {e}")
            continue 

    X_raw = np.array(X_list, dtype=np.float32)
    y_raw = np.array(y_list)

    if len(X_raw) < 5: 
        return {
            "status": "error", 
            "message": f"Jumlah data valid terlalu sedikit untuk training. Ditemukan {len(X_raw)} data valid."
        }

    # 2. PRE-PROCESSING
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y_raw)

    # =========================================================================
    # PERBAIKAN UTAMA: CHRONOLOGICAL/BLOCK SPLIT (Mencegah Kebocoran Data)
    # =========================================================================
    # Jangan gunakan train_test_split acak. Kita ambil 20% data terakhir secara berurutan
    # dari masing-masing label sebagai data uji agar tidak tercampur data tiruan.
    X_train_list, X_test_list = [], []
    y_train_list, y_test_list = [], []
    
    for label_idx in np.unique(y_encoded):
        indices = np.where(y_encoded == label_idx)[0]
        
        # Tentukan batas 80% data awal
        split_boundary = int(len(indices) * 0.8)
        
        train_idx = indices[:split_boundary]
        test_idx = indices[split_boundary:]
        
        X_train_list.append(X_raw[train_idx])
        X_test_list.append(X_raw[test_idx])
        y_train_list.append(y_encoded[train_idx])
        y_test_list.append(y_encoded[test_idx])
        
    X_train_orig = np.vstack(X_train_list)
    X_test = np.vstack(X_test_list)
    y_train_orig = np.concatenate(y_train_list)
    y_test = np.concatenate(y_test_list)

    print("\n========= INVESTIGASI DATA KEMBAR =========")
    print(f"Bentuk X_train: {X_train_orig.shape}, Bentuk X_test: {X_test.shape}")
    
    # Ambil 3 baris pertama dari data latih kelas pertama
    print("\nSample 3 baris pertama X_train (5 fitur pertama saja):")
    print(X_train_orig[:3, :5])
    
    # Ambil 3 baris pertama dari data uji kelas pertama
    print("\nSample 3 baris pertama X_test (5 fitur pertama saja):")
    print(X_test[:3, :5])
    print("===========================================\n")
    # =========================================================================

    # 4. AUGMENTASI (Hanya diterapkan pada Data Latih)
    X_train_aug = []
    y_train_aug = []
    for i in range(len(X_train_orig)):
        row = X_train_orig[i]
        label = y_train_orig[i]
        X_train_aug.append(row)
        y_train_aug.append(label)

        # Skala noise 0.02 - 0.03 sudah cukup memberikan variasi posisi koordinat landmark
        noise = np.random.normal(0, 0.02, row.shape)
        X_train_aug.append(row + noise)
        y_train_aug.append(label)

    X_train = np.array(X_train_aug)
    y_train = np.array(y_train_aug)

    # 5. TRAINING MODEL
    model = RandomForestClassifier(
        n_estimators=100,
        max_depth=10,
        min_samples_leaf=4,
        max_features="sqrt",
        bootstrap=True,
        n_jobs=1,
        random_state=42
    )
    model.fit(X_train, y_train)

    # 6. EVALUASI JURUSAN BARU
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(
        y_test, y_pred,
        target_names=[str(c) for c in label_encoder.classes_],
        output_dict=True
    )

    cm = confusion_matrix(y_test, y_pred)
    classes = label_encoder.classes_

    total_tp = total_tn = total_fp = total_fn = 0
    detail_evaluasi = {}
    for i, label_name in enumerate(classes):
        tp = int(cm[i, i])
        fn = int(np.sum(cm[i, :]) - tp)
        fp = int(np.sum(cm[:, i]) - tp)
        tn = int(len(y_test) - (tp + fp + fn))
        
        total_tp += tp
        total_tn += tn
        total_fp += fp
        total_fn += fn
        
        detail_evaluasi[str(label_name)] = {
            "TP": tp, "TN": tn, "FP": fp, "FN": fn,
            "Precision": round(float(report[str(label_name)]['precision']), 2),
            "Recall": round(float(report[str(label_name)]['recall']), 2),
            "F1-Score": round(float(report[str(label_name)]['f1-score']), 2)
        }

    confusion_matrix_formatted = {}
    for i, row_label in enumerate(classes):
        confusion_matrix_formatted[str(row_label)] = {}
        for j, col_label in enumerate(classes):
            confusion_matrix_formatted[str(row_label)][str(col_label)] = int(cm[i][j])

    # 7. SIMPAN ASSETS (.joblib & .json)
    os.makedirs(MODEL_DIR, exist_ok=True)
    joblib.dump(model, MODEL_PATH)
    joblib.dump(label_encoder, ENCODER_PATH)

    convert_to_onnx(model, ONNX_PATH)
    label_mapping = {str(i): str(label) for i, label in enumerate(label_encoder.classes_)}

    # PERBUATAN BARU: Ekstrak pure array list label murni untuk kebutuhan tracking cepat di frontend
    pure_labels = [str(label) for label in label_encoder.classes_]

    result = {
        "status": "success",
        "accuracy": float(accuracy),
        "total_data": len(rows),
        "total_uji": int(len(y_test)),
        "total_labels": int(len(classes)),
        "label_mapping": label_mapping,
        "total_statistik": {
            "TOTAL_TP": total_tp, "TOTAL_TN": total_tn, "TOTAL_FP": total_fp, "TOTAL_FN": total_fn
        },
        "detail_evaluasi": detail_evaluasi, 
        "confusion_matrix": confusion_matrix_formatted,
        "classification_report": report
    }

    # Simpan file meta lengkap (Tetap utuh seperti bawaan Anda)
    with open(META_PATH, "w") as f:
        json.dump(result, f, indent=4)

    # PERBUATAN BARU: Simpan file murni labels.json super kecil tanpa spasi/indentasi
    with open(LABELS_PATH, "w") as f:
        json.dump(pure_labels, f)

    send_assets_to_laravel()
    return result

if __name__ == "__main__":
    print("🔄 Menjalankan fungsi training secara mandiri via CLI...")
    hasil = train()
    print(json.dumps(hasil, indent=4))