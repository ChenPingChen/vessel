import unittest
import cv2
import os
import time
import logging
from ultralytics import YOLO
from django.conf import settings
from camera_manager.services.stream_manager_service import StreamManagerService
from camera_manager.services.utils.visualization import VisualizationService
from camera_manager.services.utils.grid_location_service import GridLocationService
from camera_manager.services.utils.measurement_service import MeasurementService
from detection_manager.services.vessel_mcmot import VesselMCMOT


class StreamSynchronizationTest(unittest.TestCase):
    def setUp(self):
        # 設定日誌
        logging.basicConfig(level=logging.INFO)
        
        # 設定配置文件路徑
        self.config_path = os.path.join(settings.CONFIG_DIR, 'camera_config.yaml')
        
        # 初始化服務
        self.stream_manager = StreamManagerService(config_path=self.config_path)
        self.visualizer = VisualizationService()
        
        self.mcmot = VesselMCMOT(object_model_ckpt='/home/mycena/專案/new/vessel/asiabay_demo_20250114_T2_yolo11s.pt', reid_model_ckpt='/home/mycena/專案/new/vessel/osnet_x0_25_msmt17.pt')
        
        # 初始化相機和GridLocationService
        self.stream_manager.initialize_cameras()
        self.grid_services = {}
        self.measurement_services = {}
        for camera_id, camera in self.stream_manager.config_service.cameras.items():
            self.grid_services[camera_id] = GridLocationService(camera)
            # 只為 camera4 初始化測量服務
            if camera_id == 'camera4':
                measurement_config = self.stream_manager.config_service.get_camera_measurement_config(camera_id)
                if measurement_config:
                    self.measurement_services[camera_id] = MeasurementService()
                    self.measurement_services[camera_id].initialize_from_config(measurement_config)
        
        # 設定視窗參數
        self.window_names = {
            'camera1': 'Camera 1 - 高雄港-右',
            'camera2': 'Camera 2 - 高雄港-中',
            'camera3': 'Camera 3 - 高雄港-左一',
            'camera4': 'Camera 4 - 高雄港-左二'
        }
        
        # 預先獲取所有相機的 channel_region 配置
        self.camera_regions = {
            camera_id: camera.channel_region 
            for camera_id, camera in self.stream_manager.config_service.cameras.items()
        }
        
        
    def process_frame(self, frame, camera_id):
        """處理單一幀"""
        result_frame = frame.copy()
        
        # 1. 繪製航道區域
        channel_region = self.camera_regions.get(camera_id)
        if channel_region:
            result_frame = self.visualizer.draw_channel_regions(result_frame, channel_region)
            result_frame = self.visualizer.add_region_labels(result_frame, channel_region)
        
        # 2. 繪製GCP參考點和連線
        grid_service = self.grid_services[camera_id]
        reference_points = grid_service.get_reference_points()
        result_frame = self.visualizer.draw_gcp_points(result_frame, reference_points)
        result_frame = self.visualizer.draw_reference_lines(result_frame, reference_points)
        
        # 3. 如果是 camera4，繪製測量區域和比例尺
        if camera_id == 'camera4' and camera_id in self.measurement_services:
            measurement_config = self.stream_manager.config_service.get_camera_measurement_config(camera_id)
            if measurement_config:
                result_frame = self.visualizer.draw_measurement_zone(result_frame, measurement_config)
                result_frame = self.visualizer.draw_scale_reference(result_frame, measurement_config)
        
        # 4. 使用 MCMOT 進行物件檢測和追蹤
        detected_objects = self.mcmot.process_camera_frame(frame, camera_id)
        
        # 5. 處理每個檢測到的物件
        for obj in detected_objects:
            x1, y1, x2, y2 = obj['bbox']
            class_name = obj['class_name']
            local_id = obj['local_id']
            global_id = obj.get('global_id')
            score = obj['score']
            
            # 繪製邊界框
            cv2.rectangle(result_frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
            
            # 準備顯示資訊
            info_text = [
                f"{class_name}: {score:.2f}",
                f"Local ID: {local_id}"
            ]
            
            if global_id is not None:
                info_text.append(f"Global ID: {global_id}")
            
            # 如果是 camera4，添加測量資訊
            if camera_id == 'camera4' and camera_id in self.measurement_services:
                measurement_info = self.measurement_services[camera_id].measure_vessel_dimensions(obj)
                if measurement_info:
                    info_text.extend([
                        f"L: {measurement_info['length']:.1f}m",
                        f"W: {measurement_info['width']:.1f}m",
                        f"H: {measurement_info['height']:.1f}m"
                    ])
            
            # 添加文字資訊
            y_offset = int(y1)
            for i, text in enumerate(info_text):
                y = y_offset + (i+1)*20
                (text_width, text_height), _ = cv2.getTextSize(
                    text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2
                )
                
                # 添加黑色背景
                cv2.rectangle(result_frame,
                            (int(x1), y - text_height - 5),
                            (int(x1) + text_width, y + 5),
                            (0, 0, 0), -1)
                
                # 添加文字
                cv2.putText(result_frame, text,
                          (int(x1), y),
                          cv2.FONT_HERSHEY_SIMPLEX,
                          0.6, (0, 255, 0), 2)
        
        return result_frame

    def test_synchronized_frames(self):
        """測試同步幀獲取功能"""
        try:
            # 等待串流初始化
            time.sleep(5)
            
            # 創建視窗
            for camera_id, window_name in self.window_names.items():
                cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
                cv2.resizeWindow(window_name, 960, 540)
                
                # 設定視窗位置
                if camera_id == 'camera1':
                    cv2.moveWindow(window_name, 0, 0)
                elif camera_id == 'camera2':
                    cv2.moveWindow(window_name, 960, 0)
                elif camera_id == 'camera3':
                    cv2.moveWindow(window_name, 0, 540)
                elif camera_id == 'camera4':
                    cv2.moveWindow(window_name, 960, 540)
            
            while True:
                frames = self.stream_manager.get_synchronized_frames()
                if frames:
                    for camera_id, frame in frames.items():
                        if camera_id in self.window_names:
                            # 處理每一幀
                            processed_frame = self.process_frame(frame, camera_id)
                            
                            # 顯示影像
                            cv2.imshow(self.window_names[camera_id], processed_frame)
                
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
                
                time.sleep(0.001)
                
        except Exception as e:
            logging.error(f"Test error: {str(e)}")
            raise
            
        finally:
            self.stream_manager.stop_streaming()
            cv2.destroyAllWindows()
            time.sleep(1)

if __name__ == '__main__':
    unittest.main()