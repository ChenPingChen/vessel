import faiss
import numpy as np
from .detection_service import DetectionService

class MCMOT:
    def __init__(self, object_model_ckpt: str, reid_model_ckpt: str, object_types: list=["vessel"]):
        """
        初始化多物件多相機追蹤系統
        """
        self.object_types = object_types  # 支持的物件類型
        self.gallery = {obj_type: {} for obj_type in object_types}  # {object_type: {global_id: feature_embedding}}
        self.local_to_global_id = {obj_type: {} for obj_type in object_types}  # {cameraId: {object_type: {local_id: global_id}}}
        self.global_id_counter = {obj_type: 0 for obj_type in object_types}  # 獨立計數器

        # 為每種物件類別創建 FAISS 索引
        self.faiss_indexes = {obj_type: faiss.IndexFlatL2(512) for obj_type in object_types}
        self.object_detector = DetectionService(object_model_ckpt, reid_model_ckpt)

    def detect_objects(self, frame, cameraId):
        """
        模擬物件檢測，返回物件列表，包括類別、位置和局部 ID
        """
        detected_objects = self.object_detector.detect(cameraId=cameraId, image=frame)  # 假設物件檢測器已存在
        return detected_objects  # 例如：[{ "type": "ship", "local_id": 1, "bbox": [x1, y1, x2, y2] }, {...}]

    def match_with_gallery(self, obj_type, query_feature):
        """
        使用 FAISS 進行最近鄰搜索
        """
        if len(self.gallery[obj_type]) == 0:
            return None  # Gallery 為空，直接返回
        
        query_np = np.array(query_feature).astype('float32').reshape(1, -1)
        distances, idx = self.faiss_indexes[obj_type].search(query_np, 1)  # 找最近鄰

        matched_id = list(self.gallery[obj_type].keys())[idx[0][0]]
        return matched_id if distances[0][0] < 0.7 else None  # 設定閾值 0.7

    def register_object_in_gallery(self, obj_type, feature_vector):
        """
        註冊新物件到 FAISS Gallery
        """
        global_id = self.global_id_counter[obj_type]
        self.global_id_counter[obj_type] += 1

        self.gallery[obj_type][global_id] = feature_vector
        self.faiss_indexes[obj_type].add(np.array(feature_vector).astype('float32').reshape(1, -1))

        return global_id

    def process_camera_frame(self, frame, cameraId):
        """
        處理相機影像，執行 MCMOT 流程
        """
        detected_objects = self.detect_objects(frame, cameraId)
        for obj in detected_objects:
            global_id = None
            obj_type = obj["class_name"]  # 物件類型
            local_id = obj["local_id"] # 局部 ID
            feature_embedding = obj["feature"]  # 特徵嵌入
            obj_area = (obj["bbox"][2]-obj["bbox"][0])*(obj["bbox"][3]-obj["bbox"][1])

            # 若當前局部 ID 已有對應全局 ID，則直接使用
            if cameraId in self.local_to_global_id[obj_type] and local_id in self.local_to_global_id[obj_type][cameraId]:
                global_id = self.local_to_global_id[obj_type][cameraId][local_id]
                
            elif obj_area>10000:
                # 查詢 Gallery 進行匹配
                matched_id = self.match_with_gallery(obj_type, feature_embedding)
                if matched_id is not None:
                    print("匹配成功，沿用原 ID")
                    global_id = matched_id  # 若匹配成功，沿用原 ID
                else:
                    global_id = self.register_object_in_gallery(obj_type, feature_embedding)  # 註冊新 ID
                    print("註冊新 ID")

                # 記錄局部 ID 與全局 ID 的映射
                self.local_to_global_id[obj_type].setdefault(cameraId, {})[local_id] = global_id
            obj.update({"global_id": global_id})
            print(f"Camera {cameraId}: {obj_type} {local_id} → Global ID {global_id}")

        # Camera 4: 清除離開的 ID
        if cameraId == 4:
            for obj_type in self.object_types:
                for global_id in list(self.gallery[obj_type].keys()):
                    if global_id not in self.local_to_global_id[obj_type].get(cameraId, {}).values():
                        del self.gallery[obj_type][global_id]
                        print(f"Cleared {obj_type} ID {global_id} from gallery.")

        return detected_objects