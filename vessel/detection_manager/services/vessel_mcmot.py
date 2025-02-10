import numpy as np
from .mcmot_service import MCMOT
from ..config.config import CAMERA_1_ID, CAMERA_2_ID, CAMERA_3_ID, CAMERA_4_ID, VESSEL_SIZE_THRESHOLD, GALLERY_MATCH_THRESHOLD

class VesselMCMOT(MCMOT):
    def __init__(self, object_model_ckpt, reid_model_ckpt, object_types=["vessel", "truck", "person"], size_threshold=VESSEL_SIZE_THRESHOLD):
        """
        繼承 MCMOT，並添加：
        - 進港方向判斷
        - 限制只查詢相鄰相機
        - 動態更新 Gallery 特徵
        """
        super().__init__(object_model_ckpt, reid_model_ckpt, object_types)
        self.size_threshold = size_threshold  # 設定船舶大小閾值
        self.vessel_trajectory = {} # {local_id: bbox} 記錄船舶上一幀位置
        self.max_size_record = {}  # {global_id: {cam1: area, cam2: area, ...}} 記錄船舶最大面積
        self.camera_query_scope = {  
            CAMERA_1_ID: [],         # Camera 1 無法查詢任何攝影機
            CAMERA_2_ID: [CAMERA_1_ID],      # Camera 2 可查詢 Camera 1
            CAMERA_3_ID: [CAMERA_2_ID], # Camera 3 可查詢 Camera 2, Camera 1
            CAMERA_4_ID: [CAMERA_3_ID] # Camera 4 可查詢 Camera 3, Camera 2, Camera 1
        }
        
    def is_moving_towards_port(self, last_position, current_position, cameraId):
        """
        根據相機 ID 判斷船舶是否朝向港口移動
        """
        last_x = (last_position[0] + last_position[2]) / 2
        last_y = (last_position[1] + last_position[3]) / 2
        current_x = (current_position[0] + current_position[2]) / 2
        current_y = (current_position[1] + current_position[3]) / 2

        if cameraId in [CAMERA_1_ID, CAMERA_2_ID]:
            return current_x < last_x and current_y > last_y  # 左下移動
        elif cameraId in [CAMERA_3_ID, CAMERA_4_ID]:
            return current_x < last_x  # 左移動
        return False
    
    def match_with_gallery(self, obj_type, query_feature, cameraId, obj_area):
        """
        查詢所有前序攝影機的數據，並確保 FAISS 檢索符合需求
        """
        if obj_area < self.size_threshold:
            return None  # 物件過小，不做 Gallery 查詢

        # ✅ 取得可查詢的所有前序攝影機
        valid_cameras = self.camera_query_scope.get(cameraId, [])
        print(f"Camera {cameraId} is querying previous cameras: {valid_cameras}")

        # ✅ 過濾 Gallery，只匹配來自前序攝影機的 global_id
        filtered_gallery = {
            gid: data for gid, data in self.gallery[obj_type].items()
            if any(cam in data["camera_features"] for cam in valid_cameras)
        }

        if len(filtered_gallery) == 0:
            print(f"Camera {cameraId}: No valid gallery data from previous cameras")
            return None

        # ✅ 確保每個 global_id 只使用來自前序相機的最佳特徵
        gallery_features = []
        gallery_ids = []

        for gid, data in filtered_gallery.items():
            best_cam = None
            best_feature = None

            # 找到該 global_id 在前序相機中最新的特徵
            for cam in valid_cameras:
                if cam in data["camera_features"]:
                    best_cam = cam
                    best_feature = data["camera_features"][cam]

            if best_feature is not None:
                gallery_features.append(best_feature)
                gallery_ids.append(gid)
                print(f"Using feature from Camera {best_cam} for Global ID {gid}")

        if len(gallery_features) == 0:
            print(f"Camera {cameraId}: No valid features from previous cameras")
            return None

        # ✅ 準備 FAISS 查詢
        gallery_features_np = np.array(gallery_features).astype("float32")
        query_np = np.array(query_feature).astype("float32").reshape(1, -1)

        distances, idx = self.faiss_indexes[obj_type].search(query_np, 1)

        for dist, i in zip(distances[0], idx[0]):
            print(f"FAISS Match - Distance: {dist}, Index: {i}, Camera {cameraId}")

        matched_id = gallery_ids[idx[0][0]]

        return matched_id if distances[0][0] < GALLERY_MATCH_THRESHOLD else None

    def register_object_in_gallery(self, cameraId, obj_type, feature_vector, area):
        """
        註冊新物件到 FAISS Gallery，並記錄相機位置
        """
        global_id = self.global_id_counter[obj_type]
        self.global_id_counter[obj_type] += 1

        self.gallery[obj_type][global_id] = {"camera_features": {cameraId: feature_vector}}
        self.faiss_indexes[obj_type].add(np.array(feature_vector).astype("float32").reshape(1, -1))

        self.max_size_record[global_id] = {cameraId: area}
        
        return global_id

    def update_gallery_feature(self, cameraId, obj_type, global_id, new_feature, new_area):
        """
        若新偵測到的物件面積大於該相機內的歷史記錄，則更新 Gallery 特徵
        """
        target_max_size_record = self.max_size_record.get(global_id, {})
        max_area_per_camera = target_max_size_record.get(cameraId, 0)
        
        if new_area > max_area_per_camera:
            target_max_size_record[cameraId] = new_area  # 更新該相機的最大面積
            self.gallery[obj_type][global_id]["camera_features"][cameraId] = new_feature  # 更新該相機的特徵
            
        return 

    def process_camera_frame(self, frame, cameraId):
        """
        改寫 MCMOT 方法：
        - 方向判斷
        - 限制相鄰相機匹配
        - 動態更新特徵
        """
        detected_objects = self.detect_objects(frame, cameraId)
        for obj in detected_objects:
            obj_type = obj["class_name"]
            local_id = obj["local_id"]
            feature_embedding = obj["feature"]
            bbox = obj["bbox"]
            obj_area = (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])
                
            # **方向篩選**
            # if local_id in self.vessel_trajectory:
            #     last_position = self.vessel_trajectory[local_id]
            #     if not self.is_moving_towards_port(last_position, bbox, cameraId):
            #         continue
            
            self.vessel_trajectory[local_id] = bbox

            # **查詢 Gallery 或 註冊全局 ID**
            if cameraId in self.local_to_global_id[obj_type] and local_id in self.local_to_global_id[obj_type][cameraId]:
                global_id = self.local_to_global_id[obj_type][cameraId][local_id]
            else:
                matched_id = self.match_with_gallery(obj_type, feature_embedding, cameraId, obj_area)
                if matched_id is not None:
                    global_id = matched_id  # 匹配成功
                else:
                    if obj_area >= self.size_threshold:  # ✅ 只有當大小超過閾值時才註冊
                        global_id = self.register_object_in_gallery(cameraId, obj_type, feature_embedding, obj_area)
                    else:    
                        self.update_info(object=obj, camera_id=cameraId, global_id=None)
                        continue  # 物件過小且無法匹配，不註冊 

                self.local_to_global_id[obj_type].setdefault(cameraId, {})[local_id] = global_id

            # **動態更新 Gallery 特徵**
            self.update_gallery_feature(cameraId, obj_type, global_id, feature_embedding, obj_area)            
            self.update_info(
                object=obj,
                camera_id=cameraId,
                global_id=global_id,
                )
            
            print(f"Camera {cameraId}: {obj_type} {local_id} → Global ID {global_id}")

        # **Camera 4: 清除離開的 ID**
        # if cameraId == CAMERA_4_ID:
        #     for obj_type in self.object_types:
        #         for global_id in list(self.gallery[obj_type].keys()):
        #             if global_id not in self.local_to_global_id[obj_type].get(cameraId, {}).values():
        #                 del self.gallery[obj_type][global_id]
        #                 print(f"Cleared {obj_type} ID {global_id} from gallery.")

        
        return detected_objects

    def update_info(self, camera_id, global_id, object):
        object.update({
            "camera_id": camera_id,
            "global_id": global_id,
            })
        del object["feature"]
    
