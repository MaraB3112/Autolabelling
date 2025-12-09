from transformers import OwlViTProcessor, OwlViTForObjectDetection
from PIL import Image, ImageDraw, ImageFont
import torch
import cv2
import numpy as np
from typing import Optional, Tuple, List, Dict

class ModularDetector:
    """Modular object detector with multiple fallback strategies."""
    
    def __init__(self):
        self.processor = OwlViTProcessor.from_pretrained("google/owlvit-base-patch32")
        self.model = OwlViTForObjectDetection.from_pretrained("google/owlvit-base-patch32")
    
    def detect_with_owlvit(self, image: Image.Image, prompt: str, threshold: float = 0.1) -> Optional[Tuple[List[float], float]]:
        """Strategy 1: OWL-ViT text-based detection."""
        inputs = self.processor(text=[prompt], images=image, return_tensors="pt")
        with torch.no_grad():
            outputs = self.model(**inputs)

        target_sizes = torch.tensor([image.size[::-1]])
        results = self.processor.post_process_object_detection(
            outputs=outputs,
            target_sizes=target_sizes,
            threshold=threshold
        )[0]

        scores = results["scores"]
        boxes = results["boxes"]

        if len(scores) == 0:
            return None

        best_idx = torch.argmax(scores).item()
        best_box = boxes[best_idx].tolist()
        best_score = scores[best_idx].item()
        best_box = [round(x, 2) for x in best_box]

        return best_box, best_score
    
    def detect_by_color(self, image: Image.Image, color_name: str = "yellow") -> Optional[Tuple[List[float], float]]:
        """Strategy 2: Color-based detection with predefined color ranges."""
        color_ranges = {
            "yellow": ((20, 100, 100), (40, 255, 255)),
            "red": ((0, 100, 100), (10, 255, 255)),
            "blue": ((100, 100, 100), (130, 255, 255)),
            "green": ((40, 100, 100), (80, 255, 255)),
            "orange": ((10, 100, 100), (25, 255, 255)),
            "purple": ((130, 100, 100), (160, 255, 255)),
        }
        
        lower_hsv, upper_hsv = color_ranges.get(color_name.lower(), color_ranges["yellow"])
        
        cv_image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        hsv = cv2.cvtColor(cv_image, cv2.COLOR_BGR2HSV)

        mask = cv2.inRange(hsv, np.array(lower_hsv), np.array(upper_hsv))
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if len(contours) == 0:
            return None

        largest = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(largest)
        return [x, y, x + w, y + h], 0.95  # High confidence for color detection
    
    def detect_most_different(self, image: Image.Image, grid_size: int = 16) -> Optional[Tuple[List[float], float]]:
        """Strategy 3: Find the most visually different region using saliency."""
        cv_image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        gray = cv2.cvtColor(cv_image, cv2.COLOR_BGR2GRAY)
        
        # Use Laplacian variance to find high-detail areas
        laplacian = cv2.Laplacian(gray, cv2.CV_64F)
        laplacian_abs = np.abs(laplacian)
        
        # Divide image into grid and find most salient region
        h, w = gray.shape
        cell_h, cell_w = h // grid_size, w // grid_size
        max_variance = 0
        best_region = None
        
        for i in range(grid_size):
            for j in range(grid_size):
                y1, y2 = i * cell_h, (i + 1) * cell_h
                x1, x2 = j * cell_w, (j + 1) * cell_w
                region = laplacian_abs[y1:y2, x1:x2]
                variance = np.var(region)
                
                if variance > max_variance:
                    max_variance = variance
                    best_region = [x1, y1, x2, y2]
        
        if best_region is None:
            return None
        
        # Expand the region slightly for better coverage
        padding = min(cell_w, cell_h) // 2
        x1, y1, x2, y2 = best_region
        x1 = max(0, x1 - padding)
        y1 = max(0, y1 - padding)
        x2 = min(w, x2 + padding)
        y2 = min(h, y2 + padding)
        
        return [x1, y1, x2, y2], 0.8  # Medium confidence for saliency
    
    def detect(self, image: Image.Image, config: Dict) -> Optional[Tuple[List[float], float, str]]:
        """
        Run detection with fallback strategies.
        
        Args:
            image: PIL Image
            config: Dictionary with detection configuration:
                - 'text_prompt': Text prompt for OWL-ViT (optional)
                - 'color_fallback': Color name for color-based detection (optional)
                - 'use_saliency': Whether to use saliency as final fallback (default: True)
                - 'owlvit_threshold': Confidence threshold for OWL-ViT (default: 0.1)
        
        Returns:
            Tuple of (box, score, method_used) or None
        """
        text_prompt = config.get('text_prompt')
        color_fallback = config.get('color_fallback')
        use_saliency = config.get('use_saliency', True)
        owlvit_threshold = config.get('owlvit_threshold', 0.1)
        
        # Strategy 1: Try OWL-ViT if text prompt provided
        if text_prompt:
            print(f"Trying OWL-ViT with prompt: '{text_prompt}'...")
            result = self.detect_with_owlvit(image, text_prompt, owlvit_threshold)
            if result:
                print(f"✓ OWL-ViT detected object (score: {result[1]:.3f})")
                return (*result, "owlvit")
            print("✗ OWL-ViT found nothing")
        
        # Strategy 2: Try color-based detection if color specified
        if color_fallback:
            print(f"Trying color-based detection for '{color_fallback}'...")
            result = self.detect_by_color(image, color_fallback)
            if result:
                print(f"✓ Color detection found region (score: {result[1]:.3f})")
                return (*result, "color")
            print("✗ Color detection found nothing")
        
        # Strategy 3: Try saliency-based detection as last resort
        if use_saliency:
            print("Trying saliency-based detection (most different area)...")
            result = self.detect_most_different(image)
            if result:
                print(f"✓ Saliency detection found region (score: {result[1]:.3f})")
                return (*result, "saliency")
            print("✗ Saliency detection found nothing")
        
        print("All detection strategies failed.")
        return None

def draw_label(image: Image.Image, box: List[float], label: str, 
               score: float, method: str, image_path: str):
    """Draw bounding box and label on image."""
    draw = ImageDraw.Draw(image)
    x1, y1, x2, y2 = box
    draw.rectangle([x1, y1, x2, y2], outline="red", width=3)

    try:
        font = ImageFont.truetype("arial.ttf", 20)
    except:
        font = ImageFont.load_default()

    text = f"{label} ({score:.2f}) [{method}]"
    draw.text((x1, y1 - 25), text, fill="red", font=font)
    
    # Create output filename: "image.png" -> "image_labeled.png"
    from pathlib import Path
    path = Path(image_path)
    output_path = path.stem + "_labeled" + path.suffix
    
    image.save(output_path)
    print(f"Labeled image saved to: {output_path}")

def batch_detect(image_paths: List[str], configs: List[Dict], labels: List[str]):
    """
    Run detection on multiple images with different configurations.
    
    Args:
        image_paths: List of image file paths
        configs: List of detection configs (one per image)
        labels: List of labels to assign (one per image)
    """
    detector = ModularDetector()
    
    for idx, (img_path, config, label) in enumerate(zip(image_paths, configs, labels)):
        print(f"\n{'='*60}")
        print(f"Processing image {idx+1}/{len(image_paths)}: {img_path}")
        print(f"Config: {config}")
        print(f"Label: {label}")
        print('='*60)
        
        image = Image.open(img_path).convert("RGB")
        result = detector.detect(image, config)
        
        if result:
            box, score, method = result
            output_path = f"output_{idx+1}_{label.replace(' ', '_')}.png"
            draw_label(image, box, label, score, method, output_path)
        else:
            print(f"Failed to detect anything in {img_path}")

# --------------------------
# Example Usage
# --------------------------
if __name__ == "__main__":
    # Single image detection
    detector = ModularDetector()
    
    # Example 1: Try text prompt, then color, then saliency
    config1 = {
        'text_prompt': 'a can',
        'color_fallback': 'grey',
        'use_saliency': True
    }

    label = "tomato-can"
    image_path = "tomato_can.png"
    
    image = Image.open(image_path).convert("RGB")
    result = detector.detect(image, config1)
    
    if result:
        box, score, method = result
        draw_label(image, box, label, score, method,image_path)
    
   