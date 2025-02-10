import logging
from typing import Dict,Optional
from math import sqrt

class MeasurementService:
    def __init__(self):
        self.image_size = (3840, 2160)
        self.measurement_zone = None
        self.scale_line = None
        self.pixel_to_meter_ratio = None
        self.width_ratio = 3.5
        self.measurement_history = {}
        self.final_measurements = {}
        
        # 設置日誌
        self.logger = logging.getLogger("MeasurementService")
        self._setup_logging()
    
    def _setup_logging(self):
        self.logger.setLevel(logging.INFO)
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)
        self.logger.addHandler(console_handler)

    def initialize_from_config(self, measurement_config: Dict):
        """從配置初始化測量服務"""
        if not measurement_config or not measurement_config.get('enabled'):
            return False
            
        # 設置測量區域
        zone = measurement_config['zone']
        if 'pixel_coord' in zone:
            x1, y1, x2, y2 = zone['pixel_coord']
            self.set_measurement_zone(x1, y1, x2, y2)
        
        # 設置比例尺
        scale_ref = measurement_config['scale_reference']
        if all(k in scale_ref for k in ['point1', 'point2', 'real_distance']):
            self.set_scale_line(
                x1=scale_ref['point1'][0],
                y1=scale_ref['point1'][1],
                x2=scale_ref['point2'][0],
                y2=scale_ref['point2'][1],
                real_distance=scale_ref['real_distance']
            )
        
        # 設置船隻尺寸校準參數
        if 'vessel_size_calibration' in measurement_config:
            self.width_ratio = measurement_config['vessel_size_calibration'].get('width_ratio', 3.5)
        
        return True

    def set_measurement_zone(self, x1: int, y1: int, x2: int, y2: int):
        """設置測量區域"""
        self.measurement_zone = (x1, y1, x2, y2)
        self.logger.info(f"設置測量區域: ({x1}, {y1}, {x2}, {y2})")

    def set_scale_line(self, x1: int, y1: int, x2: int, y2: int, real_distance: float):
        """設置比例尺"""
        self.scale_line = {
            'start': (x1, y1),
            'end': (x2, y2),
            'distance': real_distance
        }
        
        pixel_distance = sqrt((x2 - x1)**2 + (y2 - y1)**2)
        self.pixel_to_meter_ratio = real_distance / pixel_distance
        self.logger.info(f"設置比例尺: {self.pixel_to_meter_ratio:.4f} 公尺/像素")

    def is_in_measurement_zone(self, detection: Dict) -> bool:
        """檢查物件是否在測量區域內"""
        if not self.measurement_zone:
            return False
        
        # 計算物件中心點
        x1, y1, x2, y2 = detection['bbox']
        center_x = (x1 + x2) / 2
        center_y = (y1 + y2) / 2
        
        # 檢查中心點是否在測量區域內
        zone_x1, zone_y1, zone_x2, zone_y2 = self.measurement_zone
        return (zone_x1 <= center_x <= zone_x2 and 
                zone_y1 <= center_y <= zone_y2)

    def measure_vessel_dimensions(self, detection: Dict) -> Optional[Dict]:
        """測量船隻尺寸"""
        if not self.pixel_to_meter_ratio or not self.is_in_measurement_zone(detection):
            return None
            
        x1, y1, x2, y2 = detection['bbox']
        
        # 計算像素尺寸
        pixel_length = abs(x2 - x1)
        pixel_width = abs(y2 - y1)
        
        # 轉換為實際尺寸（公尺）
        length = pixel_length * self.pixel_to_meter_ratio
        width = pixel_width * self.pixel_to_meter_ratio
        height = width * self.width_ratio  # 使用寬度比例估算高度
        
        return {
            'length': length,
            'width': width,
            'height': height,
        }

    def update_measurement_history(self, track_id: int, measurement: Dict):
        """更新測量歷史"""
        if track_id not in self.measurement_history:
            self.measurement_history[track_id] = []
        self.measurement_history[track_id].append(measurement)

    def finalize_measurement(self, track_id: int):
        """完成測量並計算平均值"""
        if track_id not in self.measurement_history:
            return
            
        measurements = self.measurement_history[track_id]
        if not measurements:
            return
            
        # 計算平均值
        avg_length = sum(m['length'] for m in measurements) / len(measurements)
        avg_width = sum(m['width'] for m in measurements) / len(measurements)
        avg_height = sum(m['height'] for m in measurements) / len(measurements)
        
        self.final_measurements[track_id] = {
            'length': avg_length,
            'width': avg_width,
            'height': avg_height,
        }
        
        self.logger.info(f"完成船隻 {track_id} 的測量: "
                        f"長={avg_length:.1f}m, "
                        f"寬={avg_width:.1f}m, "
                        f"高={avg_height:.1f}m")