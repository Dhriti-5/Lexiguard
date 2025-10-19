from flask import Flask, request, jsonify
from werkzeug.utils import secure_filename
import os

app = Flask(__name__)

UPLOAD_FOLDER = './uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

@app.route('/analyze-pdf', methods=['POST'])
def analyze_pdf():
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400
    file = request.files['file']
    filename = secure_filename(file.filename)
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(filepath)
    
    # Dummy analysis - replace with actual AI processing
    analysis_result = {
        'summary': f'Simulated summary for {filename}',
        'risks': ['Clause A risk', 'Clause B risk']
    }
    
    return jsonify(analysis_result)

@app.route('/analyze', methods=['POST'])
def analyze_text():
    data = request.get_json()
    text = data.get('text', '')
    if not text:
        return jsonify({'error': 'No text provided'}), 400
    
    # Dummy analysis - replace with actual AI processing
    analysis_result = {
        'summary': f'Simulated summary for text input',
        'risks': ['Risk 1', 'Risk 2']
    }
    
    return jsonify(analysis_result)

if __name__ == "__main__":
    app.run(port=8000, debug=True)