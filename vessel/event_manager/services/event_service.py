from datetime import datetime
from typing import Dict, List
from event_manager.models import Vessel, VesselEvent
import time


class EventService:
    def __init__(self):
        self.active_events = {}  # {global_id: event_data}
        self.last_track_time = {}  # {global_id: last_tracking_time}
        self.tracking_interval = 10  # 追蹤點記錄間隔（秒）

    def process_detection(self, detection: Dict, camera_id: str, 
                         current_time: datetime, coordinates: Dict,
                         channel_type: str = None) -> None:
        """
        處理檢測結果並更新事件
        """
        global_id = detection.get('global_id')
        if not global_id:
            return
            
        if global_id not in self.active_events:
            # 創建新事件
            self._create_new_event(global_id, detection, camera_id, 
                                 current_time, coordinates)
        else:
            # 更新現有事件
            self._update_event(global_id, camera_id, current_time, 
                             coordinates, channel_type)
    
    def check_event_completion(self, camera_id: str, 
                             current_time: datetime) -> None:
        """
        檢查並結束完成的事件（從 camera4 消失）
        """
        if camera_id != 'camera4':
            return
            
        for global_id in list(self.active_events.keys()):
            event_data = self.active_events[global_id]
            if 'camera4' in event_data['camera_positions']:
                # 結束事件
                self._complete_event(global_id, current_time) 

    def _create_new_event(self, global_id: int, detection: Dict, 
                        camera_id: str, current_time: datetime,
                        coordinates: Dict) -> None:
        """創建新事件"""
        vessel = Vessel.objects.create(
            first_seen=current_time,
            last_seen=current_time,
            vessel_features=detection.get('feature')
        )
        
        event = VesselEvent.objects.create(
            vessel=vessel,
            start_time=current_time,
            duration_metadata={
                'tracking_points': [{
                    'timestamp': current_time.isoformat(),
                    'longitude': coordinates.get('longitude'),
                    'latitude': coordinates.get('latitude'),
                    'channel_type': None
                }]
            }
        )
        
        self.active_events[global_id] = {
            'vessel': vessel,
            'event': event,
            'camera_positions': {camera_id: coordinates}
        }
        
    def _calculate_average_position(self, positions: List[Dict]) -> Dict:
        """計算多個相機預測位置的平均值"""
        if not positions:
            return None
            
        total_lat = sum(pos['latitude'] for pos in positions)
        total_lon = sum(pos['longitude'] for pos in positions)
        return {
            'latitude': total_lat / len(positions),
            'longitude': total_lon / len(positions)
        }
    
    def _should_record_tracking_point(self, global_id: int) -> bool:
        """判斷是否需要記錄追蹤點"""
        current_time = time.time()
        last_time = self.last_track_time.get(global_id, 0)
        
        if current_time - last_time >= self.tracking_interval:
            self.last_track_time[global_id] = current_time
            return True
        return False
    

    def _update_event(self, global_id: int, camera_id: str,
                    current_time: datetime, coordinates: Dict,
                    channel_type: str = None) -> None:
        """更新現有事件"""
        event_data = self.active_events[global_id]
        event_data['camera_positions'][camera_id] = coordinates
        
        # 更新最後看到的時間
        event_data['vessel'].last_seen = current_time
        event_data['vessel'].save()
        
        # 計算平均位置
        all_positions = list(event_data['camera_positions'].values())
        avg_position = self._calculate_average_position(all_positions)
        
        # 檢查是否需要記錄追蹤點
        if self._should_record_tracking_point(global_id):
            tracking_points = event_data['event'].duration_metadata['tracking_points']
            tracking_points.append({
                'timestamp': current_time.isoformat(),
                'longitude': avg_position['longitude'],
                'latitude': avg_position['latitude'],
                'channel_type': channel_type
            })
            event_data['event'].save()

    def _complete_event(self, global_id: int, current_time: datetime) -> None:
        """完成事件並保存到資料庫"""
        event_data = self.active_events[global_id]
        event = event_data['event']
        
        # 更新結束時間
        event.end_time = current_time
        event.save()
        
        # 清理活動事件記錄
        del self.active_events[global_id]
        if global_id in self.last_track_time:
            del self.last_track_time[global_id]