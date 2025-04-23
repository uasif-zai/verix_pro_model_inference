import os
import json
import cv2
import numpy as np
from PIL import Image
from pdf2image import convert_from_path
from mrcnn import model as modellib
from mrcnn.config import Config
from mrcnn.visualize import apply_mask, random_colors
import shutil
from pathlib import Path
from flask import Flask, request, jsonify
import requests


# Constants
OUTPUT_FOLDER = "/home/dev/practice/Inference/PDFs/result_test"
WEIGHTS_DIR = "/home/dev/practice/Inference/data/weights"
METADATA_DIR = "/home/dev/practice/Inference/data/JSON_files"

def clear_dir(path):
    """Clear all files and directories in the specified path."""
    try:
        [shutil.rmtree(p) if p.is_dir() else p.unlink() for p in Path(path).glob('*')]
    except Exception as e:
        print(f"[!] Failed to clear directory {path}: {e}")
        raise

def convert_coords_to_str(obj):
    """Convert coordinates in lists or dicts to string format."""
    if isinstance(obj, list):
        if len(obj) == 2 and all(isinstance(i, int) for i in obj):
            return f"({obj[0]}, {obj[1]})"
        else:
            return [convert_coords_to_str(i) for i in obj]
    elif isinstance(obj, dict):
        return {k: convert_coords_to_str(v) for k, v in obj.items()}
    else:
        return obj

def print_object_coordinates(image_name, masks, class_ids, class_names, model_name, JSON_data, obj_count, coords_per_line=5):
    """Print object polygon coordinates and update JSON data."""
    try:
        totalDetectedObjects = masks.shape[-1]
        if model_name not in JSON_data or not JSON_data[model_name]:
            JSON_data[model_name] = {"totalDetectedObjects": 0, "detections": []}
        detection_entry = {"page": image_name, "objects": []}
        for i in range(totalDetectedObjects):
            mask = masks[:, :, i].astype(np.uint8)
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            # label = class_names[class_ids[i]]
            # Adjust the class ID if it's out of bounds
            class_id = class_ids[i]
            if class_id >= len(class_names):
                print(f"Warning: Class ID {class_id} is out of bounds.  Using the last class name.")
                class_id = len(class_names) - 1  # Use the last class as a fallback]

            for contour in contours:
                coords = contour.reshape(-1, 2).tolist()
                obj_count += 1
                for j in range(0, len(coords), coords_per_line):
                    chunk = coords[j:j+coords_per_line]
                    print(f"    {chunk}")
                detection_entry["objects"].append(coords)
        existing_detection = next((d for d in JSON_data[model_name]["detections"] if d["page"] == image_name), None)
        if existing_detection:
            existing_detection["objects"].extend(detection_entry["objects"])
        else:
            JSON_data[model_name]["detections"].append(detection_entry)
        return JSON_data, obj_count
    except Exception as e:
        print(f"[!] Error in print_object_coordinates: {e}")
        return JSON_data, obj_count

# def load_class_names(model_name):
#     """Load class names from a JSON metadata file."""
#     try:
#         json_path = os.path.join(METADATA_DIR, f"{model_name}.json")
#         with open(json_path, "r") as f:
#             data = json.load(f)
#         return data["types"]
#     except Exception as e:
#         raise FileNotFoundError(f"[!] Failed to load class names for {model_name}: {e}")

# def find_weights_file(model_name):
#     """Find the weights file path for the model."""
#     weight_path = os.path.join(WEIGHTS_DIR, f"{model_name}.h5")
#     if not os.path.exists(weight_path):
#         raise FileNotFoundError(f"Weight file not found: {weight_path}")
#     return weight_path

# class CustomConfig(Config):
#     """Custom config for Mask R-CNN inference."""
#     NAME = "object"
#     IMAGES_PER_GPU = 1
#     STEPS_PER_EPOCH = 2
#     DETECTION_MIN_CONFIDENCE = 0.9

#     def __init__(self, num_classes):
#         self.NUM_CLASSES = 1 + num_classes
#         super().__init__()



def download_pdf_from_url(url, save_path):
    response = requests.get(url)
    response.raise_for_status()
    with open(save_path, 'wb') as f:
        f.write(response.content)
    return save_path


def pdf_to_jpeg(pdf_path, output_folder, max_dim=7200, dpi=72):
    """Convert a PDF into JPEG images."""
    try:
        os.makedirs(output_folder, exist_ok=True)
        images = convert_from_path(pdf_path, dpi=dpi)
        image_paths = []
        for i, image in enumerate(images):
            image = image.convert("RGB")
            np_image = np.array(image)
            resized_np_image = resize_image(np_image, max_dim=max_dim)
            image_pil = Image.fromarray(resized_np_image)
            image_path = os.path.join(output_folder, f"page_{i+1}.jpeg")
            image_pil.save(image_path, 'JPEG')
            image_paths.append(image_path)
        return image_paths
    except Exception as e:
        print(f"[!] Failed to convert PDF to JPEG: {e}")
        raise

def resize_image(image, max_dim=7200):
    """Resize image while maintaining aspect ratio."""
    try:
        h, w = image.shape[:2]
        scale = min(max_dim / max(h, w), 1.0)
        return cv2.resize(image, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
    except Exception as e:
        print(f"[!] Failed to resize image: {e}")
        raise

def save_masked_image(image, boxes, masks, class_ids, class_names, scores, output_path):
    """Save image with instance masks and class labels."""
    try:
        masked = image.copy()
        for i in range(boxes.shape[0]):
            color = random_colors(1)[0]
            masked = apply_mask(masked, masks[:, :, i], color)
            y1, x1, y2, x2 = boxes[i]
            cv2.rectangle(masked, (x1, y1), (x2, y2), color, 2)
            label = f"{class_names[class_ids[i]]}: {scores[i]:.2f}"
            cv2.putText(masked, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
        cv2.imwrite(output_path, cv2.cvtColor(masked, cv2.COLOR_RGB2BGR))
    except Exception as e:
        print(f"[!] Failed to save masked image: {e}")
        raise

def create_pdf_from_images(image_paths, output_pdf_path):
    """Create a PDF from a list of image paths."""
    try:
        images = [Image.open(p) for p in image_paths if os.path.isfile(p)]
        if images:
            images[0].save(output_pdf_path, save_all=True, append_images=images[1:], resolution=100.0, quality=95)
            print(f"[✓] Result PDF saved at {output_pdf_path}")
    except Exception as e:
        print(f"[!] Failed to create result PDF: {e}")
        raise

def run_inference(infer, model_name, JSON_PATH, CROPPED_FOLDER, obj_count):
    """Run inference using the given model and save results."""
    try:
        # class_names = load_class_names(model_name)
        # weights_path = find_weights_file(model_name)

        # config = CustomConfig(num_classes=len(class_names))
        # model = modellib.MaskRCNN(mode="inference", model_dir=WEIGHTS_DIR, config=config)


        # try:
        #     model.load_weights(weights_path, by_name=True)
        # except Exception as e:
        #     print(f"[!] Failed to load weights: {e}")
        #     return 0

        class_names = infer.models[model_name]["class_names"]
        model = infer.models[model_name]["model"]
        print("in inference  model loaded for : ", model_name)

        image_paths = sorted([
            os.path.join(CROPPED_FOLDER, f) 
            for f in os.listdir(CROPPED_FOLDER) if f.endswith('.jpeg')
        ])
        os.makedirs(OUTPUT_FOLDER, exist_ok=True)
        output_paths = []
        polygon_count = 0 

        for path in image_paths:
            try:
                image = cv2.imread(path)
                if image is None:
                    raise ValueError(f"Image at path {path} could not be read.")
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                image = resize_image(image)
                filename = os.path.basename(path)
                output_path = os.path.join(OUTPUT_FOLDER, filename)  # ← ADD THIS LINE
                try:
                    result = model.detect([image], verbose=0)[0]
                except Exception as e:
                    print(f"[!] Inference failed on image {filename}: {e}")
                    continue

                with open(JSON_PATH, 'r') as file:
                    json_data = json.load(file)

                json_data, temp = print_object_coordinates(filename, result['masks'], result['class_ids'], class_names, model_name, json_data, obj_count)
                polygon_count += temp

                json_data_with_str_coords = convert_coords_to_str(json_data)
                with open(JSON_PATH, 'w') as f:
                    json.dump(json_data_with_str_coords, f, indent=None, separators=(',', ':'))

                save_masked_image(image, result['rois'], result['masks'], result['class_ids'], ['BG'] + class_names, result['scores'], output_path)
                output_paths.append(output_path)

            except Exception as e:
                print(f"[!] Error processing image {path}: {e}")
                raise

        create_pdf_from_images(output_paths, os.path.join(OUTPUT_FOLDER, f"{model_name}_results.pdf"))
        return polygon_count

    except Exception as e:
        print(f"[!] Error during run_inference setup: {e}")
        raise
