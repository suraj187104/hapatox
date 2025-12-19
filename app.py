"""
Flask Application - Hepatotoxicity Prediction Dashboard
"""
from flask import Flask, render_template, request, jsonify, send_file
from pathlib import Path
import json
import pandas as pd

from config import *
from models import ModelLoader, ToxicityPredictor
from utils import validate_smiles

# Initialize Flask app
app = Flask(__name__)
app.config['SECRET_KEY'] = SECRET_KEY
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size

# Load models once on startup
print(">> Loading models...")
model_loader = ModelLoader()
predictor = ToxicityPredictor(model_loader)
print(">> Models loaded successfully!")

# Load dataset for explorer
print(">> Loading dataset...")
try:
    dataset = pd.read_csv(DATASET_PATH)
    print(f">> Loaded {len(dataset):,} molecules")
except Exception as e:
    print(f">> Warning: Could not load dataset: {e}")
    dataset = None

# Load metrics
try:
    with open(METRICS_PATH, 'r') as f:
        metrics = json.load(f)
except:
    metrics = None


@app.route('/')
def home():
    """Home page"""
    stats = {
        'dataset_size': len(dataset) if dataset is not None else 0,
        'models_loaded': 3,
        'gaht_auc': metrics.get('gaht_auroc', 0.55) if metrics else 0.55,
        'rf_auc': metrics.get('rf_auroc', 0.70) if metrics else 0.70,
        'mlp_auc': metrics.get('mlp_auroc', 0.68) if metrics else 0.68,
    }
    return render_template('index.html', stats=stats)


@app.route('/predict')
def predict_page():
    """Prediction interface"""
    return render_template('predict.html')


@app.route('/api/predict', methods=['POST'])
def api_predict():
    """API endpoint for single molecule prediction"""
    try:
        data = request.get_json()
        smiles = data.get('smiles', '').strip()
        
        # Validate SMILES
        is_valid, error = validate_smiles(smiles)
        if not is_valid:
            return jsonify({'error': error}), 400
        
        # Get predictions
        results = predictor.predict(smiles)
        
        return jsonify(results)
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/model_status')
def api_model_status():
    """Check which models are available"""
    return jsonify({
        'gaht_available': model_loader.get_gaht() is not None,
        'rf_available': model_loader.get_rf() is not None,
        'mlp_available': model_loader.get_mlp() is not None
    })


@app.route('/explainability')
def explainability_page():
    """XAI visualization page"""
    return render_template('explainability.html')


@app.route('/api/explain', methods=['POST'])
def api_explain():
    """API endpoint for explainability analysis"""
    try:
        data = request.get_json()
        smiles = data.get('smiles', '').strip()
        
        # Validate SMILES
        is_valid, error = validate_smiles(smiles)
        if not is_valid:
            return jsonify({'error': error}), 400
        
        # Check if GAHT is available
        if model_loader.get_gaht() is None:
            return jsonify({
                'error': 'Explainability feature requires GAHT model',
                'message': 'This feature is not available in demo mode. GAHT model requires PyTorch and is too large for free tier hosting.'
            }), 503
        
        # Get explanation
        explanation = predictor.explain(smiles)
        
        return jsonify(explanation)
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/dataset')
def dataset_page():
    """Dataset explorer page"""
    return render_template('dataset.html')


@app.route('/api/dataset')
def api_dataset():
    """API endpoint for dataset exploration"""
    try:
        if dataset is None:
            return jsonify({'error': 'Dataset not loaded'}), 500
        
        # Pagination
        page = int(request.args.get('page', 1))
        per_page = int(request.args.get('per_page', MOLECULES_PER_PAGE))
        
        # Filtering
        toxicity_filter = request.args.get('toxicity', None)
        
        # Apply filters
        filtered_df = dataset.copy()
        if toxicity_filter is not None:
            filtered_df = filtered_df[filtered_df['label'] == int(toxicity_filter)]
        
        # Pagination
        total = len(filtered_df)
        start = (page - 1) * per_page
        end = start + per_page
        page_data = filtered_df.iloc[start:end]
        
        # Convert to JSON
        molecules = []
        for _, row in page_data.iterrows():
            molecules.append({
                'smiles': row['SMILES'],
                'label': 'Toxic' if row['label'] == 1 else 'Non-toxic',
                'source': row.get('source', 'Unknown')
            })
        
        return jsonify({
            'molecules': molecules,
            'total': total,
            'page': page,
            'per_page': per_page,
            'pages': (total + per_page - 1) // per_page
        })
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/performance')
def performance_page():
    """Model performance dashboard"""
    return render_template('performance.html', metrics=metrics)


@app.route('/research')
def research_page():
    """Research documentation and figures"""
    return render_template('research.html')


@app.route('/batch')
def batch_page():
    """Batch prediction page"""
    return render_template('batch.html')


@app.route('/api/batch', methods=['POST'])
def api_batch():
    """API endpoint for batch predictions"""
    try:
        if 'file' not in request.files:
            return jsonify({'error': 'No file provided'}), 400
        
        file = request.files['file']
        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400
        
        # Read CSV
        df = pd.read_csv(file)
        if 'SMILES' not in df.columns:
            return jsonify({'error': 'CSV must contain a SMILES column'}), 400
        
        # Limit batch size
        if len(df) > MAX_BATCH_SIZE:
            return jsonify({'error': f'Batch size limited to {MAX_BATCH_SIZE} molecules'}), 400
        
        # Predict for each molecule
        results = []
        for smiles in df['SMILES']:
            try:
                pred = predictor.predict(smiles)
                results.append(pred)
            except:
                results.append({'error': 'Prediction failed'})
        
        # Create results DataFrame
        results_df = pd.DataFrame(results)
        results_df.insert(0, 'SMILES', df['SMILES'])
        
        # Save to temporary file
        output_path = Path(app.root_path) / 'data' / 'temp_results.csv'
        results_df.to_csv(output_path, index=False)
        
        return send_file(output_path, as_attachment=True, download_name='predictions.csv')
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500


if __name__ == '__main__':
    print(f"\n{'='*60}")
    print(f">> {APP_NAME} v{APP_VERSION}")
    print(f"{'='*60}")
    print(f">> Starting server at http://{HOST}:{PORT}")
    print(f"{'='*60}\n")
    
    app.run(host=HOST, port=PORT, debug=DEBUG)
