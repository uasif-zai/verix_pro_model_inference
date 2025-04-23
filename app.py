from mrcnn import model as modellib
from mrcnn.config import Config

import os
import json


WEIGHTS_DIR = "/home/dev/practice/Inference/data/weights"
METADATA_DIR = "/home/dev/practice/Inference/data/JSON_files"


def load_class_names(model_name):
    """Load class names from a JSON metadata file."""
    try:
        json_path = os.path.join(METADATA_DIR, f"{model_name}.json")
        with open(json_path, "r") as f:
            data = json.load(f)
        return data["types"]
    except Exception as e:
        raise FileNotFoundError(f"[!] Failed to load class names for {model_name}: {e}")

def find_weights_file(model_name):
    """Find the weights file path for the model."""
    weight_path = os.path.join(WEIGHTS_DIR, f"{model_name}.h5")
    if not os.path.exists(weight_path):
        raise FileNotFoundError(f"Weight file not found: {weight_path}")
    return weight_path

class CustomConfig(Config):
    """Custom config for Mask R-CNN inference."""
    NAME = "object"
    IMAGES_PER_GPU = 1
    STEPS_PER_EPOCH = 2
    DETECTION_MIN_CONFIDENCE = 0.9

    def __init__(self, num_classes):
        self.NUM_CLASSES = 1 + num_classes
        super().__init__()




class Inference:
    #["silt_fence", "rock"]
    def __init__(self, model_names):
        self.models = {}
        for name in model_names:
            class_names = load_class_names(name)
            weights_path = find_weights_file(name)
            config = CustomConfig(num_classes=len(class_names))
            model = modellib.MaskRCNN(mode="inference", model_dir=WEIGHTS_DIR, config=config)
            try:
                model.load_weights(weights_path, by_name=True)
                self.models[name] = {
                    "model": model,
                    "class_names": class_names
                }
                print("model loaded for : ",name)
            except Exception as e:
                print(f"Error loading weights for {name}: {e}")
