import cv2
import numpy as np
from typing import Dict, List, Tuple

class VisualizationService:
    def __init__(self):
        self.colors = {
            'exterior_channel': (255, 0, 0),    # 藍色
            'interior_channel': (0, 255, 0),    # 綠色
            'VTS_channel': (0, 165, 255),       # 橙色
            'gcp_point': (0, 255, 255),         # 黃色
            'reference_line': (255, 255, 0),    # 青色
            'measurement_zone': (0, 0, 255),    # 紅色
            'scale_line': (0, 255, 0)           # 綠色
        }
        
    def draw_channel_regions(self, frame: np.ndarray, channel_regions: Dict) -> np.ndarray:
        """繪製所有通道區域"""
        result_frame = frame.copy()
        
        for region_name, region_data in channel_regions.items():
            if region_data['pixel_coord'] and len(region_data['pixel_coord']) > 0:
                color = self.colors.get(region_name)
                points = np.array(region_data['pixel_coord'], dtype=np.int32)
                
                # 繪製填充多邊形（半透明）
                overlay = result_frame.copy()
                cv2.fillPoly(overlay, [points], color)
                cv2.addWeighted(overlay, 0.3, result_frame, 0.7, 0, result_frame)
                
                # 繪製邊界線
                cv2.polylines(result_frame, [points], True, color, 2)
                
        return result_frame

    def draw_gcp_points(self, frame: np.ndarray, gcp_points: Dict) -> np.ndarray:
        """繪製GCP參考點"""
        result_frame = frame.copy()
        
        for point_id, point_data in gcp_points.items():
            if 'pixel_coord' in point_data:
                pixel_x, pixel_y = map(int, point_data['pixel_coord'])
                geo_lat, geo_lon = point_data['geo_coord']
                
                # 繪製參考點
                cv2.circle(result_frame, (pixel_x, pixel_y), 5, self.colors['gcp_point'], -1)
                
                # 添加參考點資訊
                info_text = [
                    f"{point_id}",
                    f"Lat: {geo_lat:.6f}",
                    f"Lon: {geo_lon:.6f}"
                ]
                
                y_offset = pixel_y
                for i, text in enumerate(info_text):
                    y = y_offset + (i * 30)
                    (text_width, text_height), _ = cv2.getTextSize(
                        text, cv2.FONT_HERSHEY_SIMPLEX, 1.0, 2
                    )
                    
                    # 添加黑色背景
                    cv2.rectangle(result_frame,
                                (pixel_x, y - text_height - 5),
                                (pixel_x + text_width, y + 5),
                                (0, 0, 0), -1)
                    
                    # 顯示文字
                    cv2.putText(result_frame, text,
                              (pixel_x, y),
                              cv2.FONT_HERSHEY_SIMPLEX,
                              1.0, self.colors['gcp_point'], 2)
        
        return result_frame

    def draw_reference_lines(self, frame: np.ndarray, gcp_points: Dict) -> np.ndarray:
        """繪製參考點之間的連線和距離"""
        result_frame = frame.copy()
        points_list = list(gcp_points.items())
        
        for i in range(len(points_list)):
            for j in range(i + 1, len(points_list)):
                point1_id, point1 = points_list[i]
                point2_id, point2 = points_list[j]
                
                if 'pixel_coord' in point1 and 'pixel_coord' in point2:
                    # 繪製連線
                    pt1 = tuple(map(int, point1['pixel_coord']))
                    pt2 = tuple(map(int, point2['pixel_coord']))
                    cv2.line(result_frame, pt1, pt2, self.colors['reference_line'], 2)
                    
                    # 計算中點位置
                    mid_x = int((pt1[0] + pt2[0]) / 2)
                    mid_y = int((pt1[1] + pt2[1]) / 2)
                    
                    # 計算距離（使用 GridLocationService 提供的距離）
                    distance = self._calculate_distance(
                        point1['geo_coord'][0], point1['geo_coord'][1],
                        point2['geo_coord'][0], point2['geo_coord'][1]
                    )
                    
                    # 顯示距離
                    distance_text = f"{distance:.1f}m"
                    (text_width, text_height), _ = cv2.getTextSize(
                        distance_text, cv2.FONT_HERSHEY_SIMPLEX, 1.0, 2
                    )
                    
                    # 添加黑色背景
                    cv2.rectangle(result_frame,
                                (mid_x - text_width//2, mid_y - text_height - 5),
                                (mid_x + text_width//2, mid_y + 5),
                                (0, 0, 0), -1)
                    
                    # 顯示距離文字
                    cv2.putText(result_frame, distance_text,
                              (mid_x - text_width//2, mid_y),
                              cv2.FONT_HERSHEY_SIMPLEX,
                              1.0, self.colors['reference_line'], 2)
        
        return result_frame

    def _calculate_distance(self, lat1: float, lon1: float, lat2: float, lon2: float) -> float:
        """計算兩點間的距離（公尺）"""
        earth_radius = 6371000  # 地球半徑（公尺）
        
        lat1, lon1 = map(np.radians, [lat1, lon1])
        lat2, lon2 = map(np.radians, [lat2, lon2])
        
        dlat = lat2 - lat1
        dlon = lon2 - lon1
        
        a = np.sin(dlat/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2)**2
        c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1-a))
        
        return earth_radius * c

    def add_region_labels(self, frame: np.ndarray, channel_regions: Dict) -> np.ndarray:
        """添加區域標籤"""
        result_frame = frame.copy()
        
        for region_name, region_data in channel_regions.items():
            if region_data['pixel_coord'] and len(region_data['pixel_coord']) > 0:
                points = np.array(region_data['pixel_coord'], dtype=np.int32)
                # 計算區域中心點
                center = np.mean(points, axis=0, dtype=np.int32)
                
                # 添加標籤文字
                cv2.putText(result_frame, 
                          region_name.replace('_', ' ').title(),
                          tuple(center),
                          cv2.FONT_HERSHEY_SIMPLEX,
                          1.0,
                          self.colors[region_name],
                          2)
                
        return result_frame

    def draw_measurement_zone(self, frame: np.ndarray, measurement_config: Dict) -> np.ndarray:
        """繪製測量區域（紅框）"""
        if not measurement_config or not measurement_config.get('enabled'):
            return frame
        
        result_frame = frame.copy()
        zone = measurement_config['zone']
        if 'pixel_coord' in zone:
            x1, y1, x2, y2 = zone['pixel_coord']
            
            # 繪製測量區域框
            cv2.rectangle(result_frame, (x1, y1), (x2, y2), 
                         self.colors['measurement_zone'], 2)
            
            # 添加區域標籤
            label = 'Measurement Zone'
            (text_width, text_height), _ = cv2.getTextSize(
                label, cv2.FONT_HERSHEY_SIMPLEX, 1.0, 2
            )
            
            # 添加黑色背景
            cv2.rectangle(result_frame,
                         (x1, y1 - text_height - 5),
                         (x1 + text_width, y1 + 5),
                         (0, 0, 0), -1)
            
            # 顯示標籤文字
            cv2.putText(result_frame, label,
                       (x1, y1),
                       cv2.FONT_HERSHEY_SIMPLEX,
                       1.0, self.colors['measurement_zone'], 2)
        
        return result_frame

    def draw_scale_reference(self, frame: np.ndarray, measurement_config: Dict) -> np.ndarray:
        """繪製比例尺（綠線）"""
        if not measurement_config or not measurement_config.get('enabled'):
            return frame
        
        result_frame = frame.copy()
        scale_ref = measurement_config['scale_reference']
        if all(k in scale_ref for k in ['point1', 'point2', 'real_distance']):
            pt1 = tuple(map(int, scale_ref['point1']))
            pt2 = tuple(map(int, scale_ref['point2']))
            
            # 繪製比例尺線
            cv2.line(result_frame, pt1, pt2, self.colors['scale_line'], 2)
            
            # 計算中點位置
            mid_x = int((pt1[0] + pt2[0]) / 2)
            mid_y = int((pt1[1] + pt2[1]) / 2) - 10
            
            # 顯示實際距離
            distance_text = f"{scale_ref['real_distance']:.1f}m"
            (text_width, text_height), _ = cv2.getTextSize(
                distance_text, cv2.FONT_HERSHEY_SIMPLEX, 1.0, 2
            )
            
            # 添加黑色背景
            cv2.rectangle(result_frame,
                         (mid_x - text_width//2, mid_y - text_height - 5),
                         (mid_x + text_width//2, mid_y + 5),
                         (0, 0, 0), -1)
            
            # 顯示距離文字
            cv2.putText(result_frame, distance_text,
                       (mid_x - text_width//2, mid_y),
                       cv2.FONT_HERSHEY_SIMPLEX,
                       1.0, self.colors['scale_line'], 2)
        
        return result_frame
    
    def draw_detection_results(self, frame: np.ndarray, detected_objects: List[Dict]) -> np.ndarray:
        """繪製物件偵測結果"""
        result_frame = frame.copy()
        
        for obj in detected_objects:
            x1, y1, x2, y2 = map(int, obj['bbox'])
            class_name = obj['class_name']
            local_id = obj['local_id']
            global_id = obj.get('global_id')
            score = obj['score']
            
            # 繪製邊界框
            cv2.rectangle(result_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            
            # 準備顯示資訊
            info_text = [
                f"{class_name}: {score:.2f}",
                f"Local ID: {local_id}"
            ]
            
            if global_id is not None:
                info_text.append(f"Global ID: {global_id}")
                
            # 添加文字資訊
            self._add_text_with_background(
                result_frame, 
                info_text, 
                (x1, y1), 
                color=(0, 255, 0)
            )
        
        return result_frame

    def _add_text_with_background(self, frame: np.ndarray, 
                                text_list: List[str], 
                                position: Tuple[int, int],
                                color: Tuple[int, int, int]) -> None:
        """添加帶背景的文字（內部方法）"""
        x, y = position
        for i, text in enumerate(text_list):
            y_pos = y + (i+1)*20
            (text_width, text_height), _ = cv2.getTextSize(
                text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2
            )
            
            # 添加黑色背景
            cv2.rectangle(frame,
                        (x, y_pos - text_height - 5),
                        (x + text_width, y_pos + 5),
                        (0, 0, 0), -1)
            
            # 添加文字
            cv2.putText(frame, text,
                    (x, y_pos),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6, color, 2)