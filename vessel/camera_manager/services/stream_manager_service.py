from collections import deque
import threading
import time
import logging
from typing import Dict, Optional
import numpy as np
from camera_manager.services.config_service import ConfigService
from camera_manager.services.rtsp_service import RTSPService
import queue

class StreamManagerService:
    _instance = None
    _lock = threading.Lock()

    def __new__(cls, *args, **kwargs):
        if not cls._instance:
            with cls._lock:
                if not cls._instance:
                    cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self, config_path: str = None):
        if not hasattr(self, 'initialized'):
            self.config_service = ConfigService(config_path) if config_path else None
            self.cameras = {}
            self.frame_buffers = {}
            self.is_running = False
            self.sync_interval = 0.033  # 約 30 FPS
            self.last_sync_time = time.time()
            self.sync_lock = threading.Lock()
            self.frame_queue = queue.Queue(maxsize=30)  # 新增幀隊列
            self.initialized = True

    def initialize_cameras(self):
        """初始化所有啟用的相機"""
        try:
            active_cameras = self.config_service.get_active_cameras()
            for camera_id, config in active_cameras.items():
                camera = RTSPService(camera_id, config)
                if camera.connect():
                    self.cameras[camera_id] = camera
                    self.frame_buffers[camera_id] = deque(maxlen=5)
                else:
                    logging.error(f"Failed to initialize camera {camera_id}")

            if not self.cameras:
                raise RuntimeError("No cameras were successfully initialized")

        except Exception as e:
            logging.error(f"Camera initialization failed: {str(e)}")
            raise

    def get_synchronized_frames(self) -> Optional[Dict[str, np.ndarray]]:
        """獲取同步的影像幀"""
        with self.sync_lock:
            current_time = time.time()
            frames = {}
            
            if current_time - self.last_sync_time >= self.sync_interval:
                for camera_id, camera in self.cameras.items():
                    frame = camera.get_frame()
                    if frame is not None:
                        frames[camera_id] = frame
                    else:
                        logging.warning(f"Camera {camera_id} frame not available")
                        continue
                
                # 如果至少有一個攝影機有幀，就更新時間並返回
                if frames:
                    self.last_sync_time = current_time
                    if not self.frame_queue.full():
                        self.frame_queue.put(frames)
                    return frames
                    
            return None

    def stop_streaming(self):
        """停止所有串流"""
        self.is_running = False
        for camera in self.cameras.values():
            camera.disconnect()