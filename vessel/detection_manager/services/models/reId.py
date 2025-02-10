import torch 
import numpy as np
from typing import List
from third_party.strong_sort.utils.parser import get_config
from third_party.strong_sort.strong_sort import StrongSORT

class ReID:
    def __init__(self, ckpt: str, gpu_id: int=0, yaml: str="third_party/strong_sort/configs/strong_sort.yaml"):
        self.cfg = get_config()
        self.map_location = None
        self.device = None
        self.model = None
        self.initialize(ckpt=ckpt, gpu_id=gpu_id, yaml=yaml)
        
    def initialize(self, ckpt: str, gpu_id: int, yaml: str):
        try:
            cfg = self.cfg
            self.map_location = (
                "cuda"
                if torch.cuda.is_available() and gpu_id is not None
                else "cpu"
            )   
            self.device = torch.device(
                self.map_location + ":" + str(gpu_id)
                if torch.cuda.is_available() and gpu_id is not None
                else self.map_location
            )
            cfg.merge_from_file(yaml)
            self.model = [StrongSORT(
                            ckpt,
                            self.device,
                            max_dist=cfg.STRONGSORT.MAX_DIST,
                            max_iou_distance=cfg.STRONGSORT.MAX_IOU_DISTANCE,
                            max_age=cfg.STRONGSORT.MAX_AGE,
                            n_init=cfg.STRONGSORT.N_INIT,
                            nn_budget=cfg.STRONGSORT.NN_BUDGET,
                            mc_lambda=cfg.STRONGSORT.MC_LAMBDA,
                            ema_alpha=cfg.STRONGSORT.EMA_ALPHA,
                        )]
            print(f"DEEPSORT MODEL CREATE SUCCESSFULLY, USE {self.device} ")
            
        except Exception as e:
            print(f"DEEPSORT MODEL CREATION FAILED, REASON: {str(e)}")

    def detect(self, data: List[torch.Tensor], image_ori: np.ndarray):
        outputs = []
        if isinstance(data, (list, torch.Tensor)):
            data_preprocess = self.preprocess(data)
            outputs, features = self.inference(data_preprocess, image_ori)
            outputs = self.postprocess(outputs)

        return outputs, features 

    def preprocess(self, data):
        if isinstance(data, torch.Tensor):
            det = data
        else:
            print(data)
            bboxes, labels, scores = data
            det = []
            for  bbox, label, score in zip(bboxes, labels, scores):
                det.append(bbox+[score]+[label])
            det= torch.FloatTensor(det) if self.device == torch.device("cpu") \
                    else torch.Tensor(det)
        return det

    def inference(self, data, image_ori):
        outputs, features = [[]], []
        if len(data) > 0:
            with torch.no_grad():
                xywhs = self.xyxy2xywh(data[:, 0:4])
                confs = data[:, 4]; clss = data[:, 5]
                for i, data in enumerate([data]):
                    outputs[i], features = self.model[i].update(xywhs.cpu(), confs.cpu(), clss.cpu(), image_ori)
        return outputs, features

    def postprocess(self, data):
        output = []
        if isinstance(data[0], np.ndarray):
            output = data[0].tolist()
        return output
    
    def xyxy2xywh(self, x):
        # Convert nx4 boxes from [x1, y1, x2, y2] to [x, y, w, h] where xy1=top-left, xy2=bottom-right
        y = x.clone() if isinstance(x, torch.Tensor) else np.copy(x)
        y[:, 0] = (x[:, 0] + x[:, 2]) / 2  # x center
        y[:, 1] = (x[:, 1] + x[:, 3]) / 2  # y center
        y[:, 2] = x[:, 2] - x[:, 0]  # width
        y[:, 3] = x[:, 3] - x[:, 1]  # height
        return y

    def getContextDict(self, context):
        return {
            "model_name": context.model_name, 
            "model_dir": context.model_dir, 
            "model_file": context.model_file, 
            "gpu_id": context.gpu_id,
        }

