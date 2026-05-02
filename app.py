from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
import pandas as pd
import numpy as np
import os

app = Flask(__name__)
CORS(app)

# ── Load all saved model files ──────────────────────────────────────
BASE = os.path.join(os.path.dirname(__file__), 'model_files')

model           = joblib.load(os.path.join(BASE, 'model.pkl'))
mlb_hard        = joblib.load(os.path.join(BASE, 'mlb_hard.pkl'))
mlb_soft        = joblib.load(os.path.join(BASE, 'mlb_soft.pkl'))
mlb_int         = joblib.load(os.path.join(BASE, 'mlb_int.pkl'))
encoder         = joblib.load(os.path.join(BASE, 'encoder.pkl'))
feature_columns = joblib.load(os.path.join(BASE, 'feature_columns.pkl'))

# ── Helper: encode input and predict ────────────────────────────────
def predict_top3(user_input):
    # Step 1: Encode multi-label fields
    hard_encoded = pd.DataFrame(
        mlb_hard.transform([user_input['hard_skills']]),
        columns=["hard_" + c for c in mlb_hard.classes_]
    )
    soft_encoded = pd.DataFrame(
        mlb_soft.transform([user_input['soft_skills']]),
        columns=["soft_" + c for c in mlb_soft.classes_]
    )
    int_encoded = pd.DataFrame(
        mlb_int.transform([user_input['interests']]),
        columns=["int_" + c for c in mlb_int.classes_]
    )

    # Step 2: Encode categorical fields
    cat_df = pd.DataFrame([{
        'education_level': user_input['education_level'],
        'degree_field':    user_input['degree_field'],
        'profile_type':    user_input['profile_type']
    }])
    cat_encoded = pd.DataFrame(
        encoder.transform(cat_df),
        columns=encoder.get_feature_names_out(['education_level', 'degree_field', 'profile_type'])
    )

    # Step 3: Combine all
    final_input = pd.concat([
        hard_encoded,
        soft_encoded,
        int_encoded,
        cat_encoded,
        pd.DataFrame([{'experience_level': user_input['experience_level']}])
    ], axis=1)

    # Step 4: Align columns to exactly match training order
    final_input = final_input.reindex(columns=feature_columns, fill_value=0)

    # Step 5: Predict top 3
    probs = model.predict_proba(final_input)
    top3_idx = np.argsort(probs, axis=1)[0][-3:][::-1]

    results = []
    for idx in top3_idx:
        results.append({
            "career": model.classes_[idx],
            "confidence": round(float(probs[0][idx]) * 100, 2)
        })

    return results


# ── Routes ───────────────────────────────────────────────────────────
from flask import render_template

@app.route('/')
def home():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()

        # Validate required fields
        required = ['hard_skills', 'soft_skills', 'interests',
                    'education_level', 'degree_field', 'profile_type', 'experience_level']
        
        for field in required:
            if field not in data:
                return jsonify({"error": f"Missing field: {field}"}), 400

        # Validate types
        if not isinstance(data['hard_skills'], list):
            return jsonify({"error": "hard_skills must be a list"}), 400
        if not isinstance(data['soft_skills'], list):
            return jsonify({"error": "soft_skills must be a list"}), 400
        if not isinstance(data['interests'], list):
            return jsonify({"error": "interests must be a list"}), 400

        top3 = predict_top3(data)

        return jsonify({
            "success": True,
            "recommendations": top3
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ── Run ──────────────────────────────────────────────────────────────
if __name__ == '__main__':
    app.run(debug=True, port=5000)