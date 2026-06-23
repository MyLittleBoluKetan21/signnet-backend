from flask import Flask, request, jsonify
from train_model import train as start_training
from flask_cors import CORS
import json
import joblib
import os
import numpy as np
import mysql.connector
import warnings

warnings.filterwarnings("ignore")

rf_model = None
label_encoder = None

APP_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(APP_DIR, 'model', 'rf_model.joblib')
ENCODER_PATH = os.path.join(APP_DIR, 'model', 'label_encoder.joblib')

app = Flask(__name__)
CORS(app)

# db_config = {
#     'host': '127.0.0.1',
#     'user': 'root',
#     'password': '',
#     'database': 'sign_language_db'
# }

db_config = {
    'host': os.getenv('DB_HOST', '127.0.0.1'),
    'user': os.getenv('DB_USERNAME', 'root'),
    'password': os.getenv('DB_PASSWORD', ''),
    'database': os.getenv('DB_DATABASE', 'sign_language_db'),
    'port': int(os.getenv('DB_PORT', 3306))
}

def load_all_assets():
    global rf_model, label_encoder
    try:
        if os.path.exists(MODEL_PATH) and os.path.exists(ENCODER_PATH):
            rf_model = joblib.load(MODEL_PATH)
            label_encoder = joblib.load(ENCODER_PATH)
            print("✅ Model & Encoder ML Berhasil Dimuat.")
        else:
            print("❌ Assets ML belum lengkap (.joblib belum ada). Silakan training dulu nanti.")
    except Exception as e:
        print(f"⚠️ Gagal memuat assets: {e}")

load_all_assets()

@app.route('/api/predict', methods=['POST'])
def predict():

    global rf_model, label_encoder

    try:

        # =========================
        # CHECK MODEL
        # =========================
        if rf_model is None or label_encoder is None:

            return jsonify({
                "label": "-",
                "probability": 0,
                "error": "Model belum dilatih"
            }), 500

        # =========================
        # GET JSON
        # =========================
        data = request.get_json()

        if not data:

            return jsonify({
                "label": "-",
                "probability": 0,
                "error": "Request kosong"
            }), 400

        features = data.get('features')

        # =========================
        # VALIDASI FEATURES
        # =========================
        if features is None:

            return jsonify({
                "label": "-",
                "probability": 0,
                "error": "Features tidak ditemukan"
            }), 400

        if not isinstance(features, list):

            return jsonify({
                "label": "-",
                "probability": 0,
                "error": "Features harus array/list"
            }), 400

        if len(features) != 126:

            return jsonify({
                "label": "-",
                "probability": 0,
                "error": f"Jumlah fitur harus 126, dapat {len(features)}"
            }), 400

        # =========================
        # CONVERT NUMPY
        # =========================
        try:

            features_array = np.array(
                features,
                dtype=np.float32
            ).reshape(1, -1)

        except Exception as e:

            return jsonify({
                "label": "-",
                "probability": 0,
                "error": f"Gagal convert features: {str(e)}"
            }), 400

        # =========================
        # PREDICT
        # =========================
        prediction = rf_model.predict(features_array)

        probabilities = rf_model.predict_proba(features_array)

        max_prob = float(np.max(probabilities))

        label = label_encoder.inverse_transform(
            prediction
        )[0]

        # =========================
        # CONFIDENCE FILTER
        # =========================
        CONFIDENCE_THRESHOLD = 0.70

        if max_prob < CONFIDENCE_THRESHOLD:

            return jsonify({
                "label": "-",
                "probability": round(max_prob, 4)
            })

        # =========================
        # SUCCESS
        # =========================
        return jsonify({
            "label": str(label),
            "probability": round(max_prob, 4)
        })

    except Exception as e:

        print("PREDICT ERROR:", str(e))

        return jsonify({
            "label": "-",
            "probability": 0,
            "error": str(e)
        }), 500

@app.route('/api/train', methods=['POST'])
def train_model_route():
    try:
        print("Menerima request training...")
        result = start_training()
        if result and result.get('status') == 'success':
            load_all_assets()
            return jsonify(result)
        else:
            print("Gagal:", result)
            return jsonify(result), 400
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({'status': 'error', 'message': str(e)}), 500

@app.route('/api/get_total_samples', methods=['GET'])
def get_total_samples():
    try:
        conn = mysql.connector.connect(**db_config)
        cursor = conn.cursor(dictionary=True)
        cursor.execute("SELECT label, COUNT(*) as jumlah FROM datasets GROUP BY label")
        rows = cursor.fetchall()
        
        stats = {}
        total = 0
        for row in rows:
            stats[row['label']] = row['jumlah']
            total += row['jumlah']
            
        cursor.close()
        conn.close()
        return jsonify({'total': total, 'stats': stats})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/get_model_stats', methods=['GET'])
def get_model_stats():
    meta_path = os.path.join(APP_DIR, 'model', 'meta_model.json')
    if os.path.exists(meta_path):
        with open(meta_path, 'r') as f:
            return jsonify(json.load(f))
    return jsonify({"status": "error", "message": "Belum ada data training"}), 404

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)