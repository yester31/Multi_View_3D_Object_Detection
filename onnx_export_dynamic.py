# by yhpark 2025-10-16
import os
import sys
import torch
from GenONNX import *

sys.path.insert(1, os.path.join(sys.path[0], "..", "Monocular_Depth_Estimation_TRT", "Depth_Anything_V2", "Depth-Anything-V2"))
# sys.path.insert(1, os.path.join(sys.path[0], "Depth-Anything-V2"))

CUR_DIR = os.path.dirname(os.path.abspath(__file__))

# Model Config
model_configs = {
    'vits': {'encoder': 'vits', 'features': 64, 'out_channels': [48, 96, 192, 384]},
    'vitb': {'encoder': 'vitb', 'features': 128, 'out_channels': [96, 192, 384, 768]},
    'vitl': {'encoder': 'vitl', 'features': 256, 'out_channels': [256, 512, 1024, 1024]},
    'vitg': {'encoder': 'vitg', 'features': 384, 'out_channels': [1536, 1536, 1536, 1536]}
}

def main ():
    print('[MDET] Generate Save Directory')
    save_path = os.path.join(CUR_DIR, 'onnx')
    os.makedirs(save_path, exist_ok=True)

    print('[MDET] Load model')
    input_h = 518 
    input_w = 518 
    encoder = 'vits'    # 'vits', 'vitb', 'vitg' 
    metric_model = True # True or False
    dataset = 'hypersim'# 'hypersim' for indoor model, 'vkitti' for outdoor model

    checkpoints_dir_path = "/home/workspace/Monocular_Depth_Estimation_TRT/Depth_Anything_V2/Depth-Anything-V2/checkpoints"
    if metric_model:
        from metric_depth.depth_anything_v2.dpt import DepthAnythingV2
        max_depth = 20 # 20 for indoor model,
        if dataset == 'vkitti': # 'hypersim' for indoor model, 'vkitti' for outdoor model
            max_depth = 80 #  80 for outdoor model
        model = DepthAnythingV2(**{**model_configs[encoder], 'max_depth': max_depth})
        checkpoints_path = f'{checkpoints_dir_path}/depth_anything_v2_metric_{dataset}_{encoder}.pth'
        model.load_state_dict(torch.load(checkpoints_path, map_location='cpu'))
    else:
        from depth_anything_v2.dpt import DepthAnythingV2
        model = DepthAnythingV2(**model_configs[encoder])
        checkpoints_path = f'{checkpoints_dir_path}/depth_anything_v2_{encoder}.pth'
        model.load_state_dict(torch.load(checkpoints_path, map_location='cpu'))
    model = model.eval()

    dynamo = False      # False only
    onnx_sim = True     # True or False
    model_name = f"depth_anything_v2_{encoder}_{input_h}x{input_w}_dynamic_batch"
    model_name = f"{model_name}_metric_{dataset}" if metric_model else model_name
    model_name = f"{model_name}_dynamo" if dynamo else model_name
    export_model_path = os.path.join(save_path, f'{model_name}.onnx')

    print('[MDET] dummy input model')
    dummy_input = torch.randn((1, 3, input_h, input_w), requires_grad=False)
    
    print('[MDET] Export Onnx')
    dynamic_axes = None 
    dynamic_shapes = None 
    if dynamo:
        dynamic_shapes = {"x": {0: "batch"}}
    else:
        dynamic_axes={"input": {0: "batch"},} 

    with torch.no_grad():
        torch.onnx.export(
            model, 
            dummy_input, 
            export_model_path, 
            opset_version=20, 
            input_names=["input"],
            output_names=["output"],
            dynamic_axes=dynamic_axes,
            dynamo=dynamo,
            dynamic_shapes=dynamic_shapes
        )
    print(f"[MDET] Done export onnx ({export_model_path})")

    print("[MDET] Validate exported Onnx")
    checker_onnx(export_model_path)

    print('[MDET] Simplify Onnx')
    if onnx_sim :
        export_model_sim_path = os.path.join(save_path, f'{model_name}_sim.onnx')
        simplify_onnx(export_model_path, export_model_sim_path)
    print(f"[MDET] Done simplify onnx ({export_model_sim_path})")

if __name__ == "__main__":
    main()