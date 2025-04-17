from flask import Flask, request, jsonify
from inference import run_inference, pdf_to_jpeg

app = Flask(__name__)

# Constants for static paths
CROPPED_FOLDER = "/home/dev/practice/Inference/PDFs/test_pdf"
JSON_PATH = "/home/dev/practice/Inference/PDFs/result_test/result.json"

@app.route('/process_pdf', methods=['POST'])
def process_pdf():
    # Get PDF path and model names from request
    pdf_path = request.json.get('pdf_path', None)
    model_names = request.json.get('model_names', None)

    if not pdf_path or not model_names:
        return jsonify({"error": "PDF path and model names are required"}), 400

    # Process PDF to JPEG
    pdf_to_jpeg(pdf_path, CROPPED_FOLDER)

    # Run inference for each model
    for model_name in model_names:
        run_inference(model_name, JSON_PATH, CROPPED_FOLDER)

    return jsonify({"message": "PDF processed and inference complete"}), 200

if __name__ == "__main__":
    app.run(debug=True, use_reloader=False,threaded=False)