from GenTRT import *
import cv2
import torch
import torch.nn.functional as F
from matplotlib import pyplot as plt

CUR_DIR = os.path.dirname(os.path.abspath(__file__))

def constrain_to_multiple_of(x, min_val=0, max_val=None, ensure_multiple_of=14):
    y = (np.round(x / ensure_multiple_of) * ensure_multiple_of).astype(int)

    if max_val is not None and y > max_val:
        y = (np.floor(x / ensure_multiple_of) * ensure_multiple_of).astype(int)

    if y < min_val:
        y = (np.ceil(x / ensure_multiple_of) * ensure_multiple_of).astype(int)

    return y

def preprocess_image(raw_image, input_size=518):

    image = cv2.cvtColor(raw_image, cv2.COLOR_BGR2RGB) / 255.0
    
    width, height = image.shape[1], image.shape[0]
    scale_height = input_size / height
    scale_width = input_size / width 

    # scale such that output size is lower bound
    if scale_width > scale_height:
        # fit width
        scale_height = scale_width
    else:
        # fit height
        scale_width = scale_height

    new_height = constrain_to_multiple_of(scale_height * height, min_val=input_size)
    new_width = constrain_to_multiple_of(scale_width * width, min_val=input_size)

    # resize sample
    image = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_CUBIC)

    # NormalizeImage
    image = (image - [0.485, 0.456, 0.406]) / [0.229, 0.224, 0.225]

    # PrepareForNet
    image = np.transpose(image, (2, 0, 1))
    image = np.ascontiguousarray(image).astype(np.float32)

    # [C, H, W] -> [1, C, H, W]
    image = np.expand_dims(image, axis=0)

    return image

if __name__ == "__main__":

    input_h = 518
    input_w = 518

    # Model and engine paths
    precision = "fp16"  # 'fp32' or 'fp16'
    encoder = 'vits'    # 'vits' or 'vitb' or 'vitg'
    metric_model = True # True or False
    dataset = 'hypersim'# 'hypersim' for indoor model, 'vkitti' for outdoor model
    dynamo = False       # True or False
    onnx_sim = True     # True or False
    dynamic = True     
    model_name = f"depth_anything_v2_{encoder}_{input_h}x{input_w}"
    model_name = f"{model_name}_dynamic_batch" if dynamic else f"{model_name}_static"
    model_name = f"{model_name}_metric_{dataset}" if metric_model else model_name
    model_name = f"{model_name}_dynamo" if dynamo else model_name
    model_name = f"{model_name}_sim" if onnx_sim else model_name
    onnx_path = os.path.join(CUR_DIR, 'onnx', f'{model_name}.onnx')
    engine_path = os.path.join(CUR_DIR, 'engine', f'{model_name}_{precision}.engine')
    os.makedirs(os.path.dirname(engine_path), exist_ok=True)


    # Input
    image_file_name = 'example.jpg'
    image_path = os.path.join(CUR_DIR, '..', 'Monocular_Depth_Estimation_TRT', 'data', image_file_name)
    raw_img = cv2.imread(image_path)
    h, w = raw_img.shape[:2]
    print(f'[MDET] original shape : {raw_img.shape}')
    raw_img = cv2.resize(raw_img, (input_w, input_h))
    inp = preprocess_image(raw_img, input_h)  # Preprocess image

    input_profiles = None
    if dynamic :
        input_profiles={
                "input": ((1,3,518,518), (2,3,518,518), (4,3,518,518))  # min,opt,max shape
            }
    

    trt_dav2 = TensorRTRunner(onnx_path, engine_path, fp16=True, input_profiles=input_profiles)
    trt_dav2.build_or_load_engine()
    out = trt_dav2.infer({"input": inp}, return_device=False)
    

    print("Latency (ms):", out["infer_time_ms"])
    print("Output tensor:", [k for k in out.keys() if k != "infer_time_ms"])

    print('[MDET] Post process')
    depth = torch.from_numpy(out["output"].reshape((1, input_w, input_h)))
    depth = F.interpolate(depth[:, None], (h, w), mode="bilinear", align_corners=True)[0, 0]
    depth = torch.clamp(depth, min=1e-3, max=1e3)
    depth = torch.squeeze(depth).numpy()
    print(f'[MDET] max : {depth.max():0.5f} , min : {depth.min():0.5f}')

    # ===================================================================
    print('[MDET] Generate color depth image')
    save_dir_path = os.path.join(CUR_DIR, 'results')
    os.makedirs(save_dir_path, exist_ok=True)

    # save colored depth image 
    if metric_model :
        output_file_depth_bar = os.path.join(save_dir_path, f'{os.path.splitext(image_file_name)[0]}_{model_name}_trt_new_depth_bar.jpg')
        plt.figure(figsize=(8, 6))
        inverse_depth = 1 / depth
        inverse_depth_normalized = (inverse_depth - inverse_depth.min()) / (inverse_depth.max() - inverse_depth.min())
        img = plt.imshow(inverse_depth_normalized, cmap='turbo')  
        plt.axis('off')
        cbar = plt.colorbar(img, fraction=0.046, pad=0.04)
        num_ticks = 5
        cbar_ticks = np.linspace(0, 1, num_ticks)
        cbar_ticklabels = np.linspace(depth.max(), depth.min(), num_ticks)
        cbar.set_ticks(cbar_ticks)
        cbar.set_ticklabels([f'{v:.2f} m' for v in cbar_ticklabels])
        cbar.set_label('Depth (m)', fontsize=12)
        plt.tight_layout()
        plt.savefig(output_file_depth_bar, bbox_inches='tight', pad_inches=0.1, dpi=300)
        plt.close()
    else:
        depth_normalized = (depth - depth.min()) / (depth.max() - depth.min()) * 255.0
        depth_normalized = depth_normalized.astype(np.uint8)
        output_file_depth = os.path.join(save_dir_path, os.path.splitext(image_file_name)[0] + f'_{model_name}_trt_new.jpg')
        cmap = plt.get_cmap("turbo")
        color_depth = (cmap(depth_normalized)[..., :3] * 255).astype(np.uint8)
        color_depth_bgr = cv2.cvtColor(color_depth, cv2.COLOR_RGB2BGR)    
        cv2.imwrite(output_file_depth, color_depth_bgr)

    trt_dav2.destroy()