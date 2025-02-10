import requests
from typing import Dict, Optional
import logging
from django.conf import settings
from datetime import datetime, timedelta
from django.utils import timezone

class AISService:
    def __init__(self):
        self.api_url = settings.AIS_API_URL
        self.headers = {
            "Ocp-Apim-Subscription-Key": settings.OCP_APIM_SUBSCRIPTION_KEY
        }
        self.logger = logging.getLogger(__name__)
        self.cache = {}  # 用於儲存 AIS 查詢結果
        self.query_interval = 5  # 查詢間隔（秒）
        
    def query_ais_data(self, query_params: Dict) -> Optional[Dict]:
        """
        查詢指定範圍內的 AIS 資訊
        Args:
            query_params: Dict 包含查詢參數
                {
                    "port": str,
                    "lat1": float,
                    "lng1": float,
                    "lat2": float,
                    "lng2": float,
                    "lat3": float,
                    "lng3": float,
                    "lat4": float,
                    "lng4": float
                }
        """
        # 生成快取金鑰
        cache_key = f"{query_params['lat1']},{query_params['lng1']}"
        
        # 檢查快取
        if cache_key in self.cache:
            cached_data = self.cache[cache_key]
            if (datetime.now() - cached_data['timestamp']) < timedelta(seconds=self.query_interval):
                return cached_data['data']
        
        # 進行 API 查詢
        try:
            response = requests.post(
                self.api_url,
                headers=self.headers,
                json=query_params,
                timeout=10
            )
            
            if response.status_code == 200:
                result = response.json()
                # 更新快取
                self.cache[cache_key] = {
                    'timestamp': datetime.now(),
                    'data': result
                }
                return self.filter_ais_data(result)
            else:
                self.logger.error(f"AIS API 錯誤: {response.status_code} - {response.text}")
                return None
                
        except Exception as e:
            self.logger.error(f"查詢 AIS 資料時發生錯誤: {str(e)}")
            return None
        

    # def filter_ais_data(self, ais_data: Dict, current_time: Optional[datetime] = None) -> Dict:
    #     """
    #     過濾 AIS 資料，只保留指定時間範圍內的資料
    #     Args:
    #         ais_data: 原始 AIS 資料
    #         current_time: 當前時間，如果未提供則使用系統當前時間
    #     Returns:
    #         Dict: 過濾後的 AIS 資料
    #     """
    #     if not ais_data or 'aisDatas' not in ais_data:
    #         return {}

    #     if current_time is None:
    #         current_time = timezone.now()
    #     elif timezone.is_naive(current_time):
    #         current_time = timezone.make_aware(current_time)

    #     # 設定時間範圍（前後5分鐘）
    #     time_range = timedelta(minutes=5)
    #     start_time = current_time - time_range
    #     end_time = current_time + time_range

    #     filtered_vessels = []
        
    #     for vessel in ais_data['aisDatas']:
    #         try:
    #             # 解析 AIS 資料的時間戳
    #             datetime_str = vessel['datetimeOp']
                
    #             # 處理時間字串
    #             if '+' in datetime_str:
    #                 datetime_str = datetime_str.split('+')[0]
                
    #             # 確保時間字串格式正確（處理毫秒部分）
    #             if '.' in datetime_str:
    #                 base_time, ms = datetime_str.split('.')
    #                 ms = ms[:6].ljust(6, '0')  # 確保毫秒部分有6位
    #                 datetime_str = f"{base_time}.{ms}"
                
    #             # 解析時間
    #             vessel_time = datetime.fromisoformat(datetime_str)
                
    #             # 加入台北時區
    #             vessel_time = timezone.make_aware(vessel_time, timezone.get_current_timezone())
                
    #             # 檢查是否在時間範圍內
    #             if start_time <= vessel_time <= end_time:
    #                 filtered_vessels.append(vessel)
                    
    #         except (ValueError, KeyError) as e:
    #             self.logger.warning(f"解析 AIS 資料時間戳時發生錯誤: {str(e)}, 原始時間戳: {vessel.get('datetimeOp', 'unknown')}")
    #             continue

    #     return {'aisDatas': filtered_vessels}
    
    