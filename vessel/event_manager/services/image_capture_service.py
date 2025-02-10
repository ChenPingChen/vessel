import base64
import numpy as np
import cv2
from typing import Dict, List

class ImageCaptureService:
    def __init__(self):
        self.min_size = 150  # 最小像素尺寸
        
    def should_capture_image(self, detection: Dict) -> bool:
        """判斷是否需要擷取圖片"""
        width = detection['bbox'][2] - detection['bbox'][0]
        height = detection['bbox'][3] - detection['bbox'][1]
        return width >= self.min_size and height >= self.min_size
        
    def capture_and_encode(self, frame: np.ndarray, bbox: List[int]) -> str:
        """擷取並編碼圖片"""
        x1, y1, x2, y2 = map(int, bbox)
        vessel_image = frame[y1:y2, x1:x2]
        _, buffer = cv2.imencode('.jpg', vessel_image)
        return base64.b64encode(buffer).decode('utf-8')