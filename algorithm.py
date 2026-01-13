from transformers import OwlViTProcessor, OwlViTForObjectDetection
from PIL import Image, ImageDraw, ImageFont
import torch
import cv2
import numpy as np
from typing import Optional, Tuple, List, Dict
import os
import glob


class ModularDetector:
    def __init__(self):
        self.processor = OwlViTProcessor.from_pretrained(
            "google/owlvit-base-patch32"
        )
        self.model = OwlViTForObjectDetection.from_pretrained(
            "google/owlvit-base-patch32"
        )
    
    def detect_with_owlvit(
        self,
        image: Image.Image,
        prompt: str,
        threshold: float = 0.1
    ) -> Optional[Tuple[List[float], float]]:
        inputs = self.processor(
            text=[prompt],
            images=image,
            return_tensors="pt"
        )

        with torch.no_grad():
            outputs = self.model(**inputs)

        target_sizes = torch.tensor([image.size[::-1]])
        results = self.processor.post_process_object_detection(
            outputs,
            target_sizes=target_sizes,
            threshold=threshold
        )[0]

        if len(results["scores"]) == 0:
            return None

        idx = torch.argmax(results["scores"]).item()
        box = results["boxes"][idx].tolist()
        score = results["scores"][idx].item()

        return [round(x, 2) for x in box], score

    def detect_by_color(
        self,
        image: Image.Image,
        color_name: str
    ) -> Optional[Tuple[List[float], float]]:

        color_ranges = {
            "yellow": ((20, 100, 100), (40, 255, 255)),
            "red": ((0, 100, 100), (10, 255, 255)),
            "blue": ((100, 100, 100), (130, 255, 255)),
            "green": ((40, 100, 100), (80, 255, 255)),
            "orange": ((10, 100, 100), (25, 255, 255)),
            "purple": ((130, 100, 100), (160, 255, 255)),
        }

        lower, upper = color_ranges.get(color_name.lower(), None)
        if lower is None:
            return None

        cv_image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        hsv = cv2.cvtColor(cv_image, cv2.COLOR_BGR2HSV)

        mask = cv2.inRange(
            hsv,
            np.array(lower),
            np.array(upper)
        )

        contours, _ = cv2.findContours(
            mask,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE
        )

        if not contours:
            return None

        x, y, w, h = cv2.boundingRect(
            max(contours, key=cv2.contourArea)
        )

        return [x, y, x + w, y + h], 0.95

    def detect_most_different(
        self,
        image: Image.Image,
        grid_size: int = 16
    ) -> Optional[Tuple[List[float], float]]:

        cv_image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        gray = cv2.cvtColor(cv_image, cv2.COLOR_BGR2GRAY)

        lap = np.abs(cv2.Laplacian(gray, cv2.CV_64F))

        h, w = gray.shape
        cell_h, cell_w = h // grid_size, w // grid_size

        best_var = 0
        best_box = None

        for i in range(grid_size):
            for j in range(grid_size):
                y1, y2 = i * cell_h, (i + 1) * cell_h
                x1, x2 = j * cell_w, (j + 1) * cell_w
                var = np.var(lap[y1:y2, x1:x2])

                if var > best_var:
                    best_var = var
                    best_box = [x1, y1, x2, y2]

        if best_box is None:
            return None

        pad = min(cell_w, cell_h) // 2
        x1, y1, x2, y2 = best_box

        return [
            max(0, x1 - pad),
            max(0, y1 - pad),
            min(w, x2 + pad),
            min(h, y2 + pad)
        ], 0.8

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

    def detect(
        self,
        image: Image.Image,
        config: Dict
    ) -> Optional[Dict]:

        if prompt := config.get("text_prompt"):
            if result := self.detect_with_owlvit(
                image,
                prompt,
                config.get("owlvit_threshold", 0.1)
            ):
                box, score = result
                method = "owlvit"
            else:
                box, score, method = None, None, None
        else:
            box, score, method = None, None, None

        if box is None and (color := config.get("color_fallback")):
            if result := self.detect_by_color(image, color):
                box, score = result
                method = "color"

        if box is None and config.get("use_saliency", True):
            if result := self.detect_most_different(image):
                box, score = result
                method = "saliency"

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
            "label": config.get("label", "object")
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
            
            image = Image.open(image_path)
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


if __name__ == "__main__":
    input_folder = "input_meat"
    output_folder = "output_meat"
    
    config = {
        "text_prompt": "a blue and yellow box or can",
        "label": "hero_meat_can",
        "owlvit_threshold": 0.1,
        "color_fallback": "yellow",
        "use_saliency": True
    }
    
    process_images(input_folder, output_folder, config)