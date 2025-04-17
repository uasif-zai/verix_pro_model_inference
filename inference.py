import os
import json
import cv2
import numpy as np
from PIL import Image
from pdf2image import convert_from_path
from mrcnn import model as modellib
from mrcnn.config import Config
from mrcnn.visualize import apply_mask, random_colors

from flask import Flask, request, jsonify
import os
import json

# Constants
# PDF_PATH = "/home/dev/my_projects/MaskRCNN/verix-pro-v2/PDFs/test_pdf.pdf"
# CROPPED_FOLDER = "/home/dev/my_projects/MaskRCNN/verix-pro-v2/PDFs/test_pdf"
OUTPUT_FOLDER = "/home/dev/practice/Inference/PDFs/result_test"
WEIGHTS_DIR = "/home/dev/practice/Inference/data/weights"
METADATA_DIR = "/home/dev/practice/Inference/data/JSON_files"
# JSON_PATH = "/home/dev/my_projects/MaskRCNN/verix-pro-v2/PDFs/result_test/result.json"


def convert_coords_to_str(obj):
    if isinstance(obj, list):
        if len(obj) == 2 and all(isinstance(i, int) for i in obj):
            return f"({obj[0]}, {obj[1]})"
        else:
            return [convert_coords_to_str(i) for i in obj]
    elif isinstance(obj, dict):
        return {k: convert_coords_to_str(v) for k, v in obj.items()}
    else:
        return obj



def print_object_coordinates(image_name, masks, class_ids, class_names, model_name, JSON_data, coords_per_line=5):
    totalDetectedObjects = masks.shape[-1]
    print("$$$ totalDetectedObjects : ", totalDetectedObjects)
    
    # Initialize model_name key if missing or empty
    if model_name not in JSON_data or not JSON_data[model_name]:
        JSON_data[model_name] = {
            "totalDetectedObjects": 0,
            "detections": []
        }
    
    JSON_data[model_name]["totalDetectedObjects"] += totalDetectedObjects
    
    # Prepare detection entry for current page/image
    detection_entry = {
        "page": image_name,
        "objects": []
    }
    
    for i in range(totalDetectedObjects):
        mask = masks[:, :, i].astype(np.uint8)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        label = class_names[class_ids[i]]
        print(f"\nImage: {image_name}\nObject {i+1} ({label}):")
        for contour in contours:
            coords = contour.reshape(-1, 2).tolist()
            print("  Polygon coordinates:")
            for j in range(0, len(coords), coords_per_line):
                chunk = coords[j:j+coords_per_line]
                print(f"    {chunk}")
            detection_entry["objects"].append(coords)
    
    # Check if detection for this page already exists
    existing_detection = next((d for d in JSON_data[model_name]["detections"] if d["page"] == image_name), None)
    if existing_detection:
        existing_detection["objects"].extend(detection_entry["objects"])
    else:
        JSON_data[model_name]["detections"].append(detection_entry)
    
    return JSON_data




def load_class_names(model_name):
    json_path = f"/home/dev/practice/Inference/data/JSON_files/{model_name}.json"
    with open(json_path, "r") as f:
        data = json.load(f)
    return  data["types"]


# Find corresponding weight file
def find_weights_file(model_name):
    weight_path = os.path.join(WEIGHTS_DIR, f"{model_name}.h5")
    if not os.path.exists(weight_path):
        raise FileNotFoundError(f"Weight file not found: {weight_path}")
    return weight_path

# Dynamic config
class CustomConfig(Config):

    NAME = "object"
    IMAGES_PER_GPU = 1
    STEPS_PER_EPOCH = 2
    DETECTION_MIN_CONFIDENCE = 0.9

    def __init__(self, num_classes):
        self.NUM_CLASSES = 1 + num_classes  # Background + dynamic classes
        super().__init__()  # Call parent constructor





# Convert PDF to images
def pdf_to_jpeg(pdf_path, output_folder, max_dim=7200, dpi=72):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    images = convert_from_path(pdf_path, dpi=dpi)
    image_paths = []

    for i, image in enumerate(images):
        image = image.convert("RGB")
        np_image = np.array(image)
        resized_np_image = resize_image(np_image, max_dim=max_dim)

        image_pil = Image.fromarray(resized_np_image)
        image_path = os.path.join(output_folder, f"page_{i+1}.jpeg")
        image_pil.save(image_path, 'JPEG')
        print(f"[✓] Saved: {image_path}")
        image_paths.append(image_path)

    return image_paths


# Resize image
def resize_image(image, max_dim=7200):
    h, w = image.shape[:2]
    scale = min(max_dim / max(h, w), 1.0)
    return cv2.resize(image, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)

# Save masked result
def save_masked_image(image, boxes, masks, class_ids, class_names, scores, output_path):
    masked = image.copy()
    for i in range(boxes.shape[0]):
        color = random_colors(1)[0]
        masked = apply_mask(masked, masks[:, :, i], color)
        y1, x1, y2, x2 = boxes[i]
        cv2.rectangle(masked, (x1, y1), (x2, y2), color, 2)
        label = f"{class_names[class_ids[i]]}: {scores[i]:.2f}"
        cv2.putText(masked, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
    cv2.imwrite(output_path, cv2.cvtColor(masked, cv2.COLOR_RGB2BGR))

# Save results to a PDF
def create_pdf_from_images(image_paths, output_pdf_path):
    images = [Image.open(p) for p in image_paths if os.path.isfile(p)]
    if images:
        images[0].save(output_pdf_path, save_all=True, append_images=images[1:], resolution=100.0, quality=95)
        print(f"[✓] Result PDF saved at {output_pdf_path}")

# Inference pipeline
def run_inference(model_name,JSON_PATH,CROPPED_FOLDER):
    class_names = load_class_names(model_name)
    weights_path = find_weights_file(model_name)

    config = CustomConfig(num_classes=len(class_names))
    model = modellib.MaskRCNN(mode="inference", model_dir=WEIGHTS_DIR, config=config)
    model.load_weights(weights_path, by_name=True)

    image_paths = sorted([os.path.join(CROPPED_FOLDER, f) for f in os.listdir(CROPPED_FOLDER) if f.endswith('.jpeg')])
    os.makedirs(OUTPUT_FOLDER, exist_ok=True)
    output_paths = []

    for path in image_paths:
        image = cv2.imread(path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = resize_image(image)
        filename = os.path.basename(path)
        result = model.detect([image], verbose=0)[0]
        output_path = os.path.join(OUTPUT_FOLDER, os.path.basename(path))

        # Print coordinates of detected objects
        print("############################################")
        #print(filename, result['masks'], result['class_ids'], class_names)
        with open(JSON_PATH, 'r') as file:
            json_data = json.load(file)

        json_data = print_object_coordinates(filename, result['masks'], result['class_ids'], class_names,model_name,json_data)
        # After your JSON_data is ready:
        json_data_with_str_coords = convert_coords_to_str(json_data)

        with open(JSON_PATH, 'w') as f:
            json.dump(json_data_with_str_coords, f, indent=None, separators=(',', ':'))    
                  
        save_masked_image(image, result['rois'], result['masks'], result['class_ids'], ['BG'] + class_names, result['scores'], output_path)
        output_paths.append(output_path)

    create_pdf_from_images(output_paths, os.path.join(OUTPUT_FOLDER, f"{model_name}_results.pdf"))

# # # MAIN
# if __name__ == "__main__":
#     model_names = ["silt_fence", "rock_berm"]
#     pdf_to_jpeg(PDF_PATH, CROPPED_FOLDER)
#     for model_name in model_names:
#         run_inference(model_name,JSON_PATH,CROPPED_FOLDER)

