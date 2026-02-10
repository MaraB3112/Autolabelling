import glob
import os
import random
import shutil
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont
import torch
from transformers import OwlViTProcessor, OwlViTForObjectDetection
import numpy as np
import cv2
from typing import Optional, Tuple, List, Dict
from ultralytics import YOLO  # ← ADD THIS LINE

INPUT_DIR = "image"
OUTPUT_DIR = "teddybear"
CLASS_NAME = "teddy_bear"
CLASS_ID = 0
TRAIN_SPLIT = 0.8

DETECTION_CONFIG = {
    'text_prompt': ' teddy bear',
    'confidence_threshold': 0.9, 
    'color_proximity_threshold': 100,  # Maximum distance in pixels between color regions
    'min_overlap_ratio': 0.1,  # Minimum overlap/proximity ratio required
    'iou_threshold': 0.5,       
    'color_fallback': ['brown'],
}


class ModularDetector:
    def __init__(self):
        try:
            self.model = YOLO('yolov8s-world.pt')
            print("YOLOWorld loaded successfully")
        except Exception as e:
            print(f"Failed to load YOLOWorld: {e}")
            print("Install with: pip install ultralytics")
            self.model = None
    
    def detect_with_yoloworld(
        self, 
        image: Image.Image, 
        text_prompt: str,
        confidence: float = 0.1,
        iou: float = 0.5
    ) -> List[Tuple[List[float], float]]:
        """
        Detect objects using YOLOWorld.
        Returns list of (box, score) tuples.
        """
        if self.model is None:
            return []
        
        self.model.set_classes([text_prompt])

        results = self.model.predict(
            image, 
            conf=confidence,
            iou=iou,
            verbose=False
        )
        
        detections = []
        
        if len(results) > 0 and results[0].boxes is not None:
            boxes = results[0].boxes.xyxy.cpu().numpy() 
            scores = results[0].boxes.conf.cpu().numpy()
            
            for box, score in zip(boxes, scores):
                detections.append((box.tolist(), float(score)))
        
        return detections

    def get_color_mask(self, image: Image.Image, color_name: str) -> Optional[np.ndarray]:
        """Get a binary mask for a specific color."""
        color_ranges = {
            "yellow": [
                ((20, 100, 100), (40, 255, 255)),  # Bright yellow
                ((20, 50, 50), (40, 255, 150)),     # Dark yellow
            ],
            "red": [
                ((0, 100, 100), (10, 255, 255)),    # Bright red lower 
                ((170, 100, 100), (180, 255, 255)), # Bright red upper
                ((0, 50, 50), (10, 255, 150)),      # Dark red lower
                ((170, 50, 50), (180, 255, 150)),   # Dark red upper
            ],
            "blue": [
                ((100, 100, 100), (130, 255, 255)), # Bright blue
                ((100, 50, 40), (130, 255, 150)),   # Dark blue
            ],
            "green": [
                ((40, 100, 100), (80, 255, 255)),   # Bright green
                ((40, 50, 40), (80, 255, 150)),     # Dark green
            ],
            "orange": [
                ((10, 100, 100), (25, 255, 255)),   # Bright orange
                ((10, 50, 50), (25, 255, 150)),     # Dark orange
            ],
            "purple": [
                ((130, 100, 100), (160, 255, 255)), # Bright purple
                ((130, 50, 40), (160, 255, 150)),   # Dark purple
            ],
            "grey": [
                ((0, 0, 40), (180, 50, 200)),       # Standard grey
                ((0, 0, 20), (180, 30, 100)),       # Dark grey
            ],
        }
        
        ranges_list = color_ranges.get(color_name.lower())
        if ranges_list is None:
            print(f"Warning: Color '{color_name}' not defined in color_ranges.")
            return None
        
        if isinstance(ranges_list, tuple) and len(ranges_list) == 2:
            ranges_list = [ranges_list]

        cv_image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        hsv = cv2.cvtColor(cv_image, cv2.COLOR_BGR2HSV)
        combined_mask = np.zeros(hsv.shape[:2], dtype=np.uint8)
        
        for lower_hsv, upper_hsv in ranges_list:
            mask = cv2.inRange(hsv, np.array(lower_hsv), np.array(upper_hsv))
            combined_mask = cv2.bitwise_or(combined_mask, mask)
        
        return combined_mask

    def detect_by_multi_color(
        self, 
        image: Image.Image, 
        color_names: List[str],
        proximity_threshold: float = 100,
        min_overlap_ratio: float = 0.1
    ) -> Optional[Tuple[List[float], float]]:
        """
        Detect objects that contain multiple colors in close proximity.
        
        Args:
            image: Input image
            color_names: List of color names to detect (e.g., ['red', 'green'])
            proximity_threshold: Maximum distance in pixels between color regions
            min_overlap_ratio: Minimum ratio of overlapping/nearby pixels required
        
        Returns:
            Bounding box and confidence score, or None if no valid detection
        """
        if not color_names:
            return None
        
        masks = {}
        for color in color_names:
            mask = self.get_color_mask(image, color)
            if mask is not None and np.any(mask):
                masks[color] = mask
        
        if len(masks) < len(color_names):
            print(f"  Not all colors found. Required: {color_names}, Found: {list(masks.keys())}")
            return None

        all_contours = {}
        for color, mask in masks.items():
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            valid_contours = [c for c in contours if cv2.contourArea(c) >= 100]
            if valid_contours:
                all_contours[color] = valid_contours
        
        if len(all_contours) < len(color_names):
            return None
        
        best_box = None
        best_score = 0
 
        first_color = color_names[0]
        for anchor_contour in all_contours[first_color]:
            anchor_box = cv2.boundingRect(anchor_contour)
            anchor_x, anchor_y, anchor_w, anchor_h = anchor_box
            anchor_center = (anchor_x + anchor_w // 2, anchor_y + anchor_h // 2)
            
            nearby_contours = {first_color: anchor_contour}
            distances = []
            
            for other_color in color_names[1:]:
                min_dist = float('inf')
                closest_contour = None
                
                for other_contour in all_contours[other_color]:
                    other_box = cv2.boundingRect(other_contour)
                    other_x, other_y, other_w, other_h = other_box
                    other_center = (other_x + other_w // 2, other_y + other_h // 2)
                    
                
                    dist = np.sqrt((anchor_center[0] - other_center[0])**2 + 
                                 (anchor_center[1] - other_center[1])**2)
        
                    x_overlap = max(0, min(anchor_x + anchor_w, other_x + other_w) - max(anchor_x, other_x))
                    y_overlap = max(0, min(anchor_y + anchor_h, other_y + other_h) - max(anchor_y, other_y))
                    overlap_area = x_overlap * y_overlap
                    
                    effective_dist = dist if overlap_area == 0 else dist * 0.5
                    
                    if effective_dist < min_dist:
                        min_dist = effective_dist
                        closest_contour = other_contour
                
                if min_dist <= proximity_threshold and closest_contour is not None:
                    nearby_contours[other_color] = closest_contour
                    distances.append(min_dist)
            
            # If we found all colors nearby, create a combined bounding box
            if len(nearby_contours) == len(color_names):
               
                all_points = []
                for contour in nearby_contours.values():
                    all_points.extend(contour.reshape(-1, 2))
                all_points = np.array(all_points)
              
                x, y, w, h = cv2.boundingRect(all_points)
             
                avg_distance = np.mean(distances) if distances else 0
                proximity_score = max(0, 1 - (avg_distance / proximity_threshold))
                area_score = min(1.0, (w * h) / 10000)
                combined_score = 0.7 * proximity_score + 0.3 * area_score
                
                if combined_score > best_score:
                    best_score = combined_score
                    best_box = [float(x), float(y), float(x + w), float(y + h)]
        
        if best_box is not None and best_score >= min_overlap_ratio:
            return best_box, min(0.95, best_score)
        
        return None

    def detect_by_color(self, image: Image.Image, color_input) -> Optional[Tuple[List[float], float]]:
        """
        Detect by color(s). Supports both single color (string) and multiple colors (list).
        
        Args:
            image: Input image
            color_input: Either a string (single color) or list of strings (multiple colors)
        
        Returns:
            Bounding box and confidence score, or None if no valid detection
        """
        if isinstance(color_input, str):
            # Single color =
            mask = self.get_color_mask(image, color_input)
            if mask is None or not np.any(mask):
                return None
            
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if len(contours) == 0:
                return None
            
            largest = max(contours, key=cv2.contourArea)
            if cv2.contourArea(largest) < 100:
                return None
            
            x, y, w, h = cv2.boundingRect(largest)
            return [float(x), float(y), float(x + w), float(y + h)], 0.95
        
        elif isinstance(color_input, list):
            # Multiple colors - proximity detection
            return self.detect_by_multi_color(image, color_input)
        
        else:
            print(f"Warning: Invalid color_input type: {type(color_input)}")
            return None

    def xyxy_to_xywh_norm(self, box: List[float], img_width: int, img_height: int) -> List[float]:
        x1, y1, x2, y2 = box
        x_center = ((x1 + x2) / 2) / img_width
        y_center = ((y1 + y2) / 2) / img_height
        width = (x2 - x1) / img_width
        height = (y2 - y1) / img_height
        return [x_center, y_center, width, height]

    def create_binary_mask(self, box: List[float], img_width: int, img_height: int) -> np.ndarray:
        mask = np.zeros((img_height, img_width), dtype=np.uint8)
        x1, y1, x2, y2 = map(int, box)
        mask[y1:y2, x1:x2] = 255
        return mask

    def detect(self, image: Image.Image, config: Dict) -> Optional[Dict]:
        box = None
        score = None
        method = None
        
    
        if prompt := config.get("text_prompt"):
            yoloworld_results = self.detect_with_yoloworld(
                image,
                prompt,
                confidence=config.get("confidence_threshold", 0.1),
                iou=config.get("iou_threshold", 0.5)
            )
            
            if yoloworld_results:
                box, score = yoloworld_results[0]
                method = "yoloworld"
        
        # Fallback to color detection if YOLOWorld didn't find anything
        if box is None and (color := config.get("color_fallback")):
            if result := self.detect_by_color(image, color):
                box, score = result
                method = f"color_{'_'.join(color) if isinstance(color, list) else color}"
        
        if box is None:
            return None
        
        img_width, img_height = image.size
        yolo_box = self.xyxy_to_xywh_norm(box, img_width, img_height)
        mask = self.create_binary_mask(box, img_width, img_height)
        
        return {
            "box_xyxy": box,
            "box_yolo": yolo_box,
            "confidence": score,
            "method": method,
            "mask": mask,
            "label": config.get("label", CLASS_NAME)
        }

def process_images(input_folder: str, output_folder: str, config: Dict):
    detector = ModularDetector()
    
    os.makedirs(output_folder, exist_ok=True)
    
    image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp', '*.tiff']
    image_paths = []
    for ext in image_extensions:
        image_paths.extend(glob.glob(os.path.join(input_folder, ext)))
        image_paths.extend(glob.glob(os.path.join(input_folder, ext.upper())))
    
    print(f"Found {len(image_paths)} images to process")
    
    for idx, image_path in enumerate(image_paths, 1):
        try:
            print(f"Processing {idx}/{len(image_paths)}: {os.path.basename(image_path)}")
            
            image = Image.open(image_path).convert("RGB")
            result = detector.detect(image, config)
            
            base_name = os.path.splitext(os.path.basename(image_path))[0]
            
            if result:
                print(f"  Detected using {result['method']}, confidence: {result['confidence']:.2f}")
                
                draw = ImageDraw.Draw(image)
                x1, y1, x2, y2 = result['box_xyxy']
                draw.rectangle([x1, y1, x2, y2], outline="red", width=3)
                
                label_text = result['label']
                
                try:
                    font = ImageFont.truetype("arial.ttf", 40)
                except:
                    font = ImageFont.load_default()
                
                text_bbox = draw.textbbox((0, 0), label_text, font=font)
                text_width = text_bbox[2] - text_bbox[0]
                text_height = text_bbox[3] - text_bbox[1]
                
                text_bg_x1 = x1
                text_bg_y1 = y1 - text_height - 10
                text_bg_x2 = x1 + text_width + 10
                text_bg_y2 = y1
                
                draw.rectangle([text_bg_x1, text_bg_y1, text_bg_x2, text_bg_y2], fill="red")
                draw.text((x1 + 5, y1 - text_height - 5), label_text, fill="white", font=font)
                
                output_path = os.path.join(output_folder, f"{base_name}_labeled.jpg")
                image.save(output_path)
                
                mask_output_path = os.path.join(output_folder, f"{base_name}_labeled_mask.jpg")
                cv2.imwrite(mask_output_path, result['mask'])
            else:
                print(f"  No detection found")
                
        except Exception as e:
            print(f"  Error processing {image_path}: {str(e)}")
    
    print(f"\nProcessing complete. Results saved to {output_folder}")

class YOLOLabeler(ModularDetector):
    def __init__(self):
        super().__init__()
        self.setup_folders()

    def setup_folders(self):
        """Creates the directory structure required by Ultralytics YOLOv8."""
        for split in ['train', 'val']:
            os.makedirs(f"{OUTPUT_DIR}/{split}/images", exist_ok=True)
            os.makedirs(f"{OUTPUT_DIR}/{split}/labels", exist_ok=True)

    def normalize_to_yolo(self, box, img_w, img_h):
        """
        Converts [x1, y1, x2, y2] to YOLO format.
        Coordinates are: $x_{center}, y_{center}, width, height$ (all normalized 0-1).
        """
        x1, y1, x2, y2 = box
        bw = (x2 - x1)
        bh = (y2 - y1)
        return (
            (x1 + (bw / 2)) / img_w,
            (y1 + (bh / 2)) / img_h,
            bw / img_w,
            bh / img_h
        )

    def create_yaml(self):
        """Generates the data.yaml file."""
        yaml_content = f"""
path: {os.path.abspath(OUTPUT_DIR)}
train: train/images
val: val/images

names:
  {CLASS_ID}: {CLASS_NAME}
"""
        with open(f"{OUTPUT_DIR}/data.yaml", "w") as f:
            f.write(yaml_content.strip())
        print(f"Created data.yaml")

    def process_folder(self):
        image_exts = [".jpg", ".jpeg", ".png"]
        all_images = [f for f in os.listdir(INPUT_DIR) if Path(f).suffix.lower() in image_exts]
        
        # Shuffle for random train/val split
        random.shuffle(all_images)
        split_idx = int(len(all_images) * TRAIN_SPLIT)
        
        for i, filename in enumerate(all_images):
            split = 'train' if i < split_idx else 'val'
            img_path = os.path.join(INPUT_DIR, filename)
            
            image = Image.open(img_path).convert("RGB")
            result = self.detect(image, DETECTION_CONFIG)
            
            if result:
                box_xyxy = result["box_xyxy"]
                yolo_box = result["box_yolo"]
                
                image_output_path = os.path.join(OUTPUT_DIR, split, "images", filename)
                label_output_path = os.path.join(OUTPUT_DIR, split, "labels", Path(filename).stem + ".txt")

                Path(os.path.dirname(image_output_path)).mkdir(parents=True, exist_ok=True)
                Path(os.path.dirname(label_output_path)).mkdir(parents=True, exist_ok=True)
                

                image.save(image_output_path)
                
   
                with open(label_output_path, "w") as f:
                    f.write(f"{CLASS_ID} {' '.join([f'{coord:.6f}' for coord in yolo_box])}\n")

                debug_path = os.path.join(OUTPUT_DIR, "visual_verification")
                os.makedirs(debug_path, exist_ok=True)

                box_xyxy = result["box_xyxy"]
                debug_image = image.copy() 
                self.draw_label(debug_image, box_xyxy, result["label"], result["confidence"], result["method"], os.path.join(debug_path, filename))
                                
                print(f"Processed {filename} → {split}")
        
        self.create_yaml()
        
    def draw_label(self, image: Image.Image, box: List[float], label: str, confidence: float, method: str, save_path: str):
        draw = ImageDraw.Draw(image)
        x1, y1, x2, y2 = box
        draw.rectangle([x1, y1, x2, y2], outline="red", width=3)

        label_text = f"{label} ({confidence:.2f}, {method})"
        
        try:
            font = ImageFont.truetype("arial.ttf", 20)
        except:
            font = ImageFont.load_default()
        
        text_bbox = draw.textbbox((0, 0), label_text, font=font)
        text_width = text_bbox[2] - text_bbox[0]
        text_height = text_bbox[3] - text_bbox[1]
        
        text_bg_x1 = x1
        text_bg_y1 = y1 - text_height - 5
        text_bg_x2 = x1 + text_width + 5
        text_bg_y2 = y1
        
        draw.rectangle([text_bg_x1, text_bg_y1, text_bg_x2, text_bg_y2], fill="red")
        draw.text((x1 + 2, y1 - text_height - 3), label_text, fill="white", font=font)
        
        image.save(save_path)

if __name__ == "__main__":
    labeler = YOLOLabeler()
    labeler.process_folder()