from typing import Tuple, Dict, List
import numpy as np
from datetime import datetime
from math import radians, sin, cos, atan2, sqrt
from rasterio.transform import from_gcps, AffineTransformer
from rasterio.control import GroundControlPoint
from ..config_service import CameraConfig

class GridLocationService:
    def __init__(self, camera_config: CameraConfig):
        """
        初始化 GridLocationService
        Args:
            config: CameraConfig 實例，包含所有相機配置
        """
        self.gcps: List[GroundControlPoint] = []
        self.transformer = None
        self.inverse_transformer = None
        self.reference_points = {}
        self.direction_vector = None
        self.scale_factor = None
        self.earth_radius = 6371000  # 地球半徑（公尺）
        
        # 初始化配置
        self._initialize_from_config(camera_config)
    
    def _initialize_from_config(self, config: CameraConfig):
        """從配置初始化服務"""
        if not config.gcp:
            raise ValueError("配置中缺少 GCP 點位資訊")
            
        # 載入GCP點位
        for point_id, point_data in config.gcp.items():
            self.add_reference_point(
                point_id=point_id,
                pixel_coord=point_data['pixel_coord'],
                geo_coord=point_data['geo_coord']
            )

    def add_reference_point(self, point_id: str, 
                          pixel_coord: Tuple[int, int],
                          geo_coord: Tuple[float, float]):
        """添加參考點並更新轉換器"""
        gcp = GroundControlPoint(
            row=pixel_coord[1],
            col=pixel_coord[0],
            x=geo_coord[1],  # 經度
            y=geo_coord[0],  # 緯度
            z=0,
            id=point_id
        )
        self.gcps.append(gcp)
        
        if len(self.gcps) >= 3:
            self._update_transformers()
        
        self.reference_points[point_id] = {
            'pixel_coord': pixel_coord,
            'geo_coord': geo_coord
        }
        
        if len(self.reference_points) == 2:
            self.calculate_reference_line()

    def _update_transformers(self):
        """更新坐標轉換器"""
        transform = from_gcps(self.gcps)
        self.transformer = AffineTransformer(transform)
        self.inverse_transformer = AffineTransformer(~transform)

    def pixel_to_geo(self, pixel_x: int, pixel_y: int) -> Tuple[float, float]:
        """將像素坐標轉換為地理坐標"""
        if self.transformer is None:
            raise ValueError("需要至少3個參考點才能進行坐標轉換")
        
        lon, lat = self.transformer.xy(pixel_y, pixel_x)
        return (lat, lon)  # 返回 (緯度, 經度)

    def geo_to_pixel(self, lat: float, lon: float) -> Tuple[int, int]:
        """將地理坐標轉換為像素坐標"""
        if self.inverse_transformer is None:
            raise ValueError("需要至少3個參考點才能進行坐標轉換")
        
        row, col = self.inverse_transformer.rowcol(lon, lat)
        return (int(col), int(row))  # 返回 (x, y)

    def calculate_reference_line(self):
        """計算參考線方向向量和比例尺"""
        points = list(self.reference_points.values())
        p1, p2 = points[0], points[1]

        dx_pixel = p2['pixel_coord'][0] - p1['pixel_coord'][0]
        dy_pixel = p2['pixel_coord'][1] - p1['pixel_coord'][1]
        pixel_distance = np.sqrt(dx_pixel**2 + dy_pixel**2)

        real_distance = self.haversine_distance(
            p1['geo_coord'][0], p1['geo_coord'][1],
            p2['geo_coord'][0], p2['geo_coord'][1]
        )

        self.direction_vector = (dx_pixel/pixel_distance, dy_pixel/pixel_distance)
        self.scale_factor = real_distance / pixel_distance

    def haversine_distance(self, lat1: float, lon1: float, 
                         lat2: float, lon2: float) -> float:
        """計算兩點間的實際地理距離（公尺）"""
        lat1, lon1, lat2, lon2 = map(radians, [lat1, lon1, lat2, lon2])
        
        dlat = lat2 - lat1
        dlon = lon2 - lon1
        a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
        c = 2 * atan2(sqrt(a), sqrt(1-a))
        
        return self.earth_radius * c

    def process_detection(self, detection: dict) -> dict:
        """處理檢測結果，轉換座標並添加時間戳"""
        center_x = int((detection['bbox'][0] + detection['bbox'][2]) / 2)
        center_y = int((detection['bbox'][1] + detection['bbox'][3]) / 2)
        
        lat, lon = self.pixel_to_geo(center_x, center_y)
            
        return {
            'timestamp': datetime.now().isoformat(),
            'pixel_coord': (center_x, center_y),
            'geo_coord': (lat, lon),
            'confidence': detection.get('confidence', 0.0),
            'object_type': detection.get('class', 'unknown')
        }

    def get_reference_points(self) -> Dict:
        """獲取所有參考點資訊"""
        return self.reference_points.copy()