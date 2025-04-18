from flask import Flask, request, jsonify
import json
from inference import run_inference, pdf_to_jpeg,clear_dir
from db_manager import init_db, insert_test_table_data
import os

app = Flask(__name__)

# Initialize database
init_db(app)

# Constants for static paths
CROPPED_FOLDER = "/home/dev/practice/Inference/PDFs/test_pdf"
JSON_PATH = "/home/dev/practice/Inference/PDFs/result.json"

@app.route('/process_pdf', methods=['POST'])
def process_pdf():
    # Get PDF path, model names, and pdf_id from request
    pdf_path = request.json.get('pdf_path', None)
    model_names = request.json.get('model_names', None)
    pdf_id = request.json.get('pdf_id', None)

    if not pdf_path or not model_names or not pdf_id:
        return jsonify({"error": "PDF path, model names, and pdf_id are required"}), 400
    
    initial_data = {
        "silt_fence": {},
        "rock_berm": {},
        "inlet_protection": {}
    }

    # Ensure the directory for JSON_PATH exists
    os.makedirs(os.path.dirname(JSON_PATH), exist_ok=True)

    # Write initial JSON data to file (overwrite if exists)
    with open(JSON_PATH, 'w') as f:
        json.dump(initial_data, f, indent=2)


    try:
        # Process PDF to JPEG
        pdf_to_jpeg(pdf_path, CROPPED_FOLDER)
        # Run inference for each model
        for model_name in model_names:
            obj_count =0
            polygon_count = run_inference(model_name, JSON_PATH, CROPPED_FOLDER, obj_count)
            with open(JSON_PATH, 'r') as file:
                json_data = json.load(file)
            print("polygon_count count in main is for ", model_name , " : ", polygon_count)
            json_data[model_name]["totalDetectedObjects"] += polygon_count

                # Write the updated JSON data back to the file
            with open(JSON_PATH, 'w') as file:
                json.dump(json_data, file, indent=2)



        # Insert data into test_table
        new_entry_id = insert_test_table_data(pdf_id, json_data, status="processed")
        clear_dir("/home/dev/practice/Inference/PDFs/result_test")
        clear_dir("/home/dev/practice/Inference/PDFs/test_pdf")

        return jsonify({
            "message": "PDF processed, inference complete, and data inserted",
            # "test_table_id": new_entry_id
        }), 200

    except Exception as e:
        clear_dir("/home/dev/practice/Inference/PDFs/result_test")
        clear_dir("/home/dev/practice/Inference/PDFs/test_pdf")
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(debug=True, use_reloader=False, threaded=False)