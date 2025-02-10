import numpy as np
from .models.object_detect import ObjectDetect
from .models.reId import ReID

class DetectionService:
    def __init__(self, object_model_ckpt: str, reid_model_ckpt: str):
        self.reid_model_ckpt = reid_model_ckpt
        self.object_model = self._create_model(model_class=ObjectDetect, model_ckpt=object_model_ckpt)
        self.reid_model_dict = dict()
        
    def detect(self, cameraId: str, image: np.ndarray):
        persons = self.detect_object(cameraId=cameraId, image=image)
        return persons

    def _create_model(self, model_class, model_ckpt: str):
        model = None
        try: 
            model = model_class(ckpt=model_ckpt)
            print(f"權重載入成功！！")
            
        except Exception as e:
            print("權重載入失敗，原因：{e}")

        return model
        
    def getReidModel(self, cameraId: str):
        if cameraId not in self.reid_model_dict:
            reid_model = self._create_model(model_class=ReID, model_ckpt=self.reid_model_ckpt)
            self.reid_model_dict.update({
                cameraId: reid_model
            })
        return self.reid_model_dict.get(cameraId)


    def detect_object(self, cameraId: str, image: np.ndarray):
        objects = []
        preds = self.object_model.detect(image)
        
        reid_model = self.getReidModel(cameraId=cameraId)
        outputs, features = reid_model.detect(preds, image)
        if len(outputs):
            for output, feature in zip(outputs, features):
                bbox = [int(pt) for pt in output[:4]]  # x1, y1, x2, y2
                score, label = round(float(output[4]), 3), int(output[5])
                id = int(output[6]) if len(output)>6 else None
                if label==0:
                    objects.append({
                        "class_name": self.object_model.names[label],
                        "local_id": id,
                        "global_id": None,
                        "bbox": bbox, 
                        "score": score, 
                        "feature": feature  
                        })
        return objects

        

