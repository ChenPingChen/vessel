from src.dao.context import Context

reid_context = Context(
    model_name='REID',
    model_dir='weights',
    model_file='osnet_x0_25_msmt17.pt',
    gpu_id=0
)

vessel_context = Context(    
    model_name='Vessel',
    model_dir='weights',
    model_file='vessel_yolo11n_v2.pt',
    threshold=0.5
)
