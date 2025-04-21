from flask import Flask, request, jsonify
import json
import os
from inference import run_inference, pdf_to_jpeg, clear_dir
from db_manager import init_db, insert_test_table_data

app = Flask(__name__)
init_db(app)

CROPPED_FOLDER = "/home/dev/practice/Inference/PDFs/test_pdf"
JSON_PATH = "/home/dev/practice/Inference/PDFs/result.json"

@app.route('/process_pdf', methods=['POST'])
def process_pdf():
    pdf_path = request.json.get('pdf_path')
    model_names = request.json.get('model_names')
    pdf_id = request.json.get('pdf_id')

    if not pdf_path or not model_names or not pdf_id:
        return jsonify({"error": "PDF path, model names, and pdf_id are required"}), 400

    initial_data = {
        "silt_fence": {},
        "rock_berm": {},
        "inlet_protection": {}
    }

    try:
        os.makedirs(os.path.dirname(JSON_PATH), exist_ok=True)
    except OSError as e:
        return jsonify({"error": f"Failed to create JSON directory: {str(e)}"}), 500

    try:
        with open(JSON_PATH, 'w') as f:
            json.dump(initial_data, f, indent=2)
    except (IOError, json.JSONDecodeError) as e:
        return jsonify({"error": f"Failed to write initial JSON: {str(e)}"}), 500

    try:
        pdf_to_jpeg(pdf_path, CROPPED_FOLDER)
    except Exception as e:
        return jsonify({"error": f"Failed during PDF to JPEG conversion: {str(e)}"}), 500

    try:
        for model_name in model_names:
            obj_count = 0
            try:
                polygon_count = run_inference(model_name, JSON_PATH, CROPPED_FOLDER, obj_count)
            except Exception as e:
                return jsonify({"error": f"Inference failed for model '{model_name}': {str(e)}"}), 500

            try:
                with open(JSON_PATH, 'r') as file:
                    json_data = json.load(file)
            except (IOError, json.JSONDecodeError) as e:
                return jsonify({"error": f"Failed to read JSON for model '{model_name}': {str(e)}"}), 500

            if model_name in json_data:
                json_data[model_name]["totalDetectedObjects"] = json_data[model_name].get("totalDetectedObjects", 0) + polygon_count

            try:
                with open(JSON_PATH, 'w') as file:
                    json.dump(json_data, file, indent=2)
            except (IOError, json.JSONDecodeError) as e:
                return jsonify({"error": f"Failed to update JSON for model '{model_name}': {str(e)}"}), 500

        try:
            with open(JSON_PATH, 'r') as file:
                json_data = json.load(file)
        except (IOError, json.JSONDecodeError) as e:
            return jsonify({"error": f"Failed to read final JSON: {str(e)}"}), 500

        try:
            insert_test_table_data(pdf_id, json_data, status="processed")
        except Exception as e:
            return jsonify({"error": f"Database insertion failed: {str(e)}"}), 500

        clear_dir("/home/dev/practice/Inference/PDFs/result_test")
        clear_dir("/home/dev/practice/Inference/PDFs/test_pdf")

        return jsonify({"message": "Coordinates have been successfully saved to the database with status 'processed'."}), 200


    except Exception as e:
        clear_dir("/home/dev/practice/Inference/PDFs/result_test")
        clear_dir("/home/dev/practice/Inference/PDFs/test_pdf")
        return jsonify({"error": f"Unexpected failure: {str(e)}"}), 500

if __name__ == "__main__":
    app.run(debug=True, use_reloader=False, threaded=False)





