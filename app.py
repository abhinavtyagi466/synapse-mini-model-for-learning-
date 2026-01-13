from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
from bpe_tokenizer import ProTokenizer
import os

app = Flask(__name__)
CORS(app)

tokenizer = ProTokenizer()

@app.route('/')
def index():
    return send_from_directory('.', 'index.html')

@app.route('/train', methods=['POST'])
def train():
    data = request.json
    text = data.get('text', '')
    target_size = data.get('vocab_size', 300)
    
    if not text:
        return jsonify({"error": "No text provided"}), 400
    
    tokenizer.train(text, target_size)
    return jsonify({
        "message": "Training successful",
        "new_vocab_size": len(tokenizer.vocab)
    })

@app.route('/encode', methods=['POST'])
def encode():
    data = request.json
    text = data.get('text', '')
    
    if not text:
        return jsonify({"tokens": [], "pieces": []})
    
    token_ids = tokenizer.encode(text)
    
    # Get the actual text pieces for visualization
    pieces = []
    for tid in token_ids:
        raw_bytes = bytes(tokenizer.vocab.get(tid, []))
        pieces.append(raw_bytes.decode('utf-8', errors='replace'))
        
    return jsonify({
        "tokens": token_ids,
        "pieces": pieces
    })

@app.route('/status', methods=['GET'])
def status():
    return jsonify({
        "vocab_size": len(tokenizer.vocab),
        "merges_count": len(tokenizer.merges)
    })

if __name__ == '__main__':
    app.run(debug=True, port=5000)
