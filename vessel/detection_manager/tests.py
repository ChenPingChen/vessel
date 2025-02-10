from django.test import TestCase
import unittest
import cv2
import os
import numpy as np
from pathlib import Path
from detection_manager.services.vessel_mcmot import VesselMCMOT
from detection_manager.config.config import CAMERA_1_ID

# Create your tests here.

class TestVesselMCMOT(unittest.TestCase):
    def setUp(self):
        """測試初始化"""
        # 設定模型路徑
        self.object_model_path = "/home/mycena/專案/new/vessel/asiabay_demo_20250114_T2_yolo11s.pt"
        self.reid_model_path = "/home/mycena/專案/new/vessel/osnet_x0_25_msmt17.pt"
        
        # 初始化 VesselMCMOT
        self.vessel_mcmot = VesselMCMOT(
            object_model_ckpt=self.object_model_path,
            reid_model_ckpt=self.reid_model_path
        )
        
        # 設定測試影片路徑
        self.video_path = '/home/mycena/mcmot/250113_mergev_13027_middle_10x.mp4'
        
        # 確保測試影片存在
        if not os.path.exists(self.video_path):
            raise FileNotFoundError(f"測試影片不存在：{self.video_path}")

    def test_process_camera_frame(self):
        """測試影片處理功能"""
        cap = cv2.VideoCapture(self.video_path)
        
        try:
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                    
                # 處理幀
                detected_objects = self.vessel_mcmot.process_camera_frame(
                    frame=frame,
                    cameraId=CAMERA_1_ID
                )
                
                print(detected_objects)
            
        finally:
            cap.release()

if __name__ == '__main__':
    unittest.main()
