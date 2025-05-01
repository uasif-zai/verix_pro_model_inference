from flask import Flask, request, jsonify
import json
import os
import requests
import tempfile
from inference import run_inference, pdf_to_jpeg, clear_dir, download_pdf_from_url
from concurrent.futures import ProcessPoolExecutor
from app import Inference

app = Flask(__name__)

infer = Inference(["silt_fence","rock_berm","inlet_protection","tree_protection","scp"])

# Initialize the process pool executor with a reasonable number of workers
executor = ProcessPoolExecutor(max_workers=4)  # Adjust based on your system's CPU cores

def process_pdf_async(pdf_path, model_names, pdf_id, project_id, callback_url):
    """Process the PDF in a separate process."""
    try:
        # Create a unique temporary directory for this task
        with tempfile.TemporaryDirectory() as temp_dir:
            cropped_folder = os.path.join(temp_dir, "cropped")
            output_folder = os.path.join(temp_dir, "output")
            json_path = os.path.join(temp_dir, "result.json")
            download_pdf_path = os.path.join(temp_dir, "inf_pdf.pdf")
            
            os.makedirs(cropped_folder, exist_ok=True)
            os.makedirs(output_folder, exist_ok=True)

            # Initialize empty JSON data
            initial_data = {}
            with open(json_path, 'w') as f:
                json.dump(initial_data, f, indent=2)

            # Download PDF
            print(f"Downloading PDFx to {download_pdf_path}")
            download_pdf_from_url(pdf_path, download_pdf_path)

            # Convert PDF to JPEG
            print(f"Converting PDF to JPEG in {cropped_folder}")
            pdf_to_jpeg(download_pdf_path, cropped_folder)

            # Run inference for each model
            for model_name in model_names:
                print(f"Running inference for model: {model_name}")
                obj_count = 0
                polygon_count = run_inference(infer, model_name, json_path, cropped_folder, output_folder, obj_count)
                
                with open(json_path, 'r') as file:
                    json_data = json.load(file)

                if model_name in json_data:
                    json_data[model_name]["totalDetectedObjects"] = json_data[model_name].get("totalDetectedObjects", 0) + polygon_count

                with open(json_path, 'w') as file:
                    json.dump(json_data, file, indent=2)

            with open(json_path, 'r') as file:
                json_data = json.load(file)

            # Send the callback with success status
            result = {
                "status": "success",
                "result_data": json_data,
                "project_id": project_id,
                "file_id": pdf_id
            }
            print("\n\n\n\n\n\n\n", json_data, "\n\n\n\n\n\n\n")
            requests.post(callback_url, json=result, headers={"Content-Type": "application/json"})
            
            print("[✓] Processing completed successfully")

    except Exception as e:
        # Send failure status back to callback URL
        error_result = {
            "status": "failure",
            "result_data": str(e),
            "project_id": project_id,
            "file_id": pdf_id
        }
        print(f"Error processing PDF: {str(e)}")
        requests.post(callback_url, json=error_result, headers={"Content-Type": "application/json"})

@app.route('/process_pdf', methods=['POST'])
def process_pdf():
    """Handle the POST request and delegate processing to a separate process."""
    pdf_path = request.json.get('pdf_path')
    model_names = request.json.get('model_names')
    pdf_id = request.json.get('pdf_id')
    callback_url = request.json.get('callback_url')
    project_id = request.json.get('project_id')

    print("PDF PATH: ", pdf_path)
    print("Model Names: ", model_names)

    if not pdf_path or not model_names or not pdf_id or not callback_url:
        return jsonify({"error": "PDF path, model names, pdf_id, and callback_url are required"}), 400

    # Submit the processing task to the process pool
    executor.submit(process_pdf_async, pdf_path, model_names, pdf_id, project_id, callback_url)

    # Respond immediately that processing has started
    response = {
        "message": "Processing started successfully",
        "status": "processing-started",
        "project_id": project_id,
        "file_id": pdf_id,
        "success": True,
    }
    return jsonify(response), 202  # Accepted status code

if __name__ == "__main__":
    app.run(debug=True, use_reloader=False, threaded=False)
