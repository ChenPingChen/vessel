import os
import yaml
from typing import Dict, Optional

class CameraConfig:
    def __init__(self, name: str, rtsp_url: str, status: str, gcp: Dict, channel_region: Dict):
        self.name = name
        self.rtsp_url = rtsp_url
        self.status = status
        self.gcp = gcp
        self.channel_region = channel_region
        self.measurement = {}  # 初始化空的測量配置

class MeasurementConfig:
    def __init__(self, enabled: bool, zone: Dict, scale_reference: Dict, vessel_size_calibration: Dict):
        self.enabled = enabled
        self.zone = zone
        self.scale_reference = scale_reference
        self.vessel_size_calibration = vessel_size_calibration

class ConfigService:
    def __init__(self, config_path: str):
        self.config_path = config_path
        self.cameras: Dict[str, CameraConfig] = {}
        self._load_config()
        self._load_measurement_config()
    
    def _load_config(self):
        """載入相機配置"""
        try:
            with open(self.config_path, 'r') as f:
                config = yaml.safe_load(f)
                for cam_id, cam_config in config['cameras'].items():
                    self.cameras[cam_id] = CameraConfig(
                        name=cam_config['name'],
                        rtsp_url=cam_config['rtsp_url'],
                        status=cam_config['status'],
                        gcp=cam_config['gcp'],
                        channel_region=cam_config['channel_region']
                    )
        except Exception as e:
            raise RuntimeError(f"Failed to load camera config: {str(e)}")

    def _load_measurement_config(self):
        """載入測量配置"""
        try:
            measurement_path = os.path.join(
                os.path.dirname(self.config_path),
                'measurement_config.yaml'
            )
            
            if not os.path.exists(measurement_path):
                return
            
            with open(measurement_path, 'r') as f:
                config = yaml.safe_load(f)
                if 'camera4_measurement' in config:
                    measurement = MeasurementConfig(
                        enabled=config['camera4_measurement']['enabled'],
                        zone=config['camera4_measurement']['zone'],
                        scale_reference=config['camera4_measurement']['scale_reference'],
                        vessel_size_calibration=config['camera4_measurement']['vessel_size_calibration']
                    )
                    if 'camera4' in self.cameras:
                        self.cameras['camera4'].measurement = {
                            'enabled': measurement.enabled,
                            'zone': measurement.zone,
                            'scale_reference': measurement.scale_reference,
                            'vessel_size_calibration': measurement.vessel_size_calibration
                        }

        except Exception as e:
            raise RuntimeError(f"Failed to load measurement config: {str(e)}")

    def get_active_cameras(self) -> Dict[str, CameraConfig]:
        """獲取所有啟用的相機"""
        return {k: v for k, v in self.cameras.items() if v.status == 'active'}

    def get_camera_measurement_config(self, camera_id: str) -> Optional[Dict]:
        """獲取特定相機的測量配置"""
        camera = self.cameras.get(camera_id)
        if camera and camera.measurement:
            return camera.measurement
        return None