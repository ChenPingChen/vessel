import cv2
import numpy as np
import threading
import time
import logging
import subprocess
from typing import Optional, Tuple
from camera_manager.services.config_service import CameraConfig


class RTSPService:
    def __init__(self, camera_id: str, config: CameraConfig):
        self.camera_id = camera_id
        self.config = config
        self.stream = None
        self.frame = None
        self.process = None
        self.last_frame_time = time.time()
        self.is_running = False
        self.lock = threading.Lock()
        self.connection_retry_count = 0
        self.max_retries = 3
        self.retry_interval = 5
        self.frame_timeout = 3.0
        self.frame_width = 3840  # 設定實際的幀寬度
        self.frame_height = 2160  # 設定實際的幀高度
        
    def connect(self) -> bool:
        """建立RTSP連接"""
        try:
            command = [
                'ffmpeg',
                '-rtsp_transport', 'tcp',
                '-i', self.config.rtsp_url,
                '-f', 'rawvideo',
                '-pix_fmt', 'bgr24',
                '-preset', 'ultrafast',
                '-tune', 'zerolatency',
                '-vf', f'scale={self.frame_width}:{self.frame_height}',  # 確保輸出解析度
                '-'
            ]
            
            self.process = subprocess.Popen(
                command,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                bufsize=10**8
            )
            
            self.is_running = True
            threading.Thread(target=self._stream_capture, daemon=True).start()
            return True
            
        except Exception as e:
            logging.error(f"Camera {self.camera_id} connection error: {str(e)}")
            return False
    
    def _stream_capture(self):
        """串流捕捉循環"""
        while self.is_running:
            try:
                # 從 FFmpeg 進程讀取原始影像數據
                raw_image = self.process.stdout.read(self.get_frame_size())
                if len(raw_image) == 0:
                    self._handle_stream_error()
                    continue
                    
                # 轉換為 numpy 陣列
                frame = np.frombuffer(raw_image, dtype=np.uint8)
                frame = frame.reshape(self.get_frame_shape())
                
                with self.lock:
                    self.frame = frame
                    self.last_frame_time = time.time()
                    
            except Exception as e:
                logging.error(f"Stream capture error: {str(e)}")
                self._handle_stream_error()
            
            time.sleep(0.001)

    def get_frame_size(self) -> int:
        """獲取每幀的大小（以字節為單位）"""
        return self.frame_width * self.frame_height * 3  # 3 表示 BGR 三個通道

    def get_frame_shape(self) -> Tuple[int, int, int]:
        """獲取幀的形狀"""
        return (self.frame_height, self.frame_width, 3)

    def _handle_stream_error(self):
        """處理串流錯誤"""
        if time.time() - self.last_frame_time > self.frame_timeout:
            self._reconnect()

    def _reconnect(self):
        """重新連接"""
        if self.connection_retry_count < self.max_retries:
            logging.info(f"Attempting to reconnect camera {self.camera_id}")
            if self.process:
                self.process.terminate()
                self.process.wait()
            if self.connect():
                self.connection_retry_count = 0
            else:
                self.connection_retry_count += 1

    def get_frame(self) -> Optional[np.ndarray]:
        """獲取當前幀"""
        with self.lock:
            return self.frame.copy() if self.frame is not None else None

    def disconnect(self):
        """斷開連接"""
        self.is_running = False
        if self.process:
            self.process.terminate()
            self.process.wait()