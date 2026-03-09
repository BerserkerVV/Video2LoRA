import os
import sys
import argparse

import numpy as np
import torch
from diffusers import (CogVideoXDDIMScheduler, DDIMScheduler,
                       DPMSolverMultistepScheduler,
                       EulerAncestralDiscreteScheduler, EulerDiscreteScheduler,
                       PNDMScheduler)
from PIL import Image

current_file_path = os.path.abspath(__file__)
project_roots = [os.path.dirname(current_file_path), os.path.dirname(os.path.dirname(current_file_path)), os.path.dirname(os.path.dirname(os.path.dirname(current_file_path)))]
for project_root in project_roots:
    sys.path.insert(0, project_root) if project_root not in sys.path else None

from videox_fun.dist import set_multi_gpus_devices, shard_model
from videox_fun.models import (AutoencoderKLCogVideoX,
                              CogVideoXTransformer3DModel, T5EncoderModel,
                              T5Tokenizer)
from videox_fun.pipeline import (CogVideoXFunInpaintPipeline,
                                CogVideoXFunPipeline)
from videox_fun.utils.fp8_optimization import (convert_model_weight_to_float8, replace_parameters_by_name,
                                              convert_weight_dtype_wrapper)
from videox_fun.utils.lora_utils import create_network
from videox_fun.utils.utils import get_image_to_video_latent, save_videos_grid
from videox_fun.utils.hypernet import VideoHyperDream
import json


def parse_args():
    parser = argparse.ArgumentParser(description="CogVideoX-Fun I2V Inference with LiLoRA and HyperNetwork")
    
    # GPU memory mode
    parser.add_argument("--gpu_memory_mode", type=str, default="model_full_load",
                       choices=["model_full_load", "model_full_load_and_qfloat8", "model_cpu_offload", 
                               "model_cpu_offload_and_qfloat8", "sequential_cpu_offload"],
                       help="GPU memory mode")
    
    # Multi GPUs config
    parser.add_argument("--ulysses_degree", type=int, default=1, help="Ulysses degree for multi-GPU inference")
    parser.add_argument("--ring_degree", type=int, default=1, help="Ring degree for multi-GPU inference")
    parser.add_argument("--fsdp_dit", action="store_true", help="Use FSDP for transformer")
    parser.add_argument("--fsdp_text_encoder", action="store_true", default=True, help="Use FSDP for text encoder")
    parser.add_argument("--compile_dit", action="store_true", help="Compile transformer for speedup")
    
    # Model paths
    parser.add_argument("--model_name", type=str, default="models/Diffusion_Transformer/CogVideoX-Fun-V1.1-2b-InP",
                       help="Path to pretrained model")
    parser.add_argument("--sampler_name", type=str, default="DDIM_Origin",
                       choices=["Euler", "Euler A", "DPM++", "PNDM", "DDIM_Cog", "DDIM_Origin"],
                       help="Sampler name")
    parser.add_argument("--transformer_path", type=str, default=None, help="Path to transformer checkpoint")
    parser.add_argument("--vae_path", type=str, default=None, help="Path to VAE checkpoint")
    
    # LiLoRA and HyperNetwork paths
    parser.add_argument("--lilora_path", type=str, default=None, help="Path to LiLoRA weights (.safetensors)")
    parser.add_argument("--hypernetwork_path", type=str, default=None, help="Path to HyperNetwork weights (.safetensors)")
    
    # LiLoRA params (should match training config)
    parser.add_argument("--lora_rank", type=int, default=1, help="LiLoRA rank")
    parser.add_argument("--lora_down_dim", type=int, default=200, help="LiLoRA down dimension")
    parser.add_argument("--lora_up_dim", type=int, default=100, help="LiLoRA up dimension")
    parser.add_argument("--lora_weight", type=float, default=0.55, help="LiLoRA weight multiplier")
    
    # Input data paths
    parser.add_argument("--first_frame_path", type=str, default=None, help="Path to first frame image")
    parser.add_argument("--ref_video_path", type=str, default=None, help="Path to reference video for HyperNetwork")
    
    # Video generation params
    parser.add_argument("--video_length", type=int, default=49, help="Number of frames to generate")
    parser.add_argument("--fps", type=int, default=8, help="FPS for output video")
    parser.add_argument("--partial_video_length", type=int, default=None, 
                       help="Length of each sub video segment for ultra long video generation")
    parser.add_argument("--overlap_video_length", type=int, default=4, 
                       help="Overlap length between segments for ultra long video generation")
    
    # Weight dtype
    parser.add_argument("--weight_dtype", type=str, default="bfloat16", choices=["bfloat16", "float16", "float32"],
                       help="Weight dtype")
    
    # Prompts
    parser.add_argument("--prompt", type=str, default=None, help="Prompt for video generation")
    parser.add_argument("--negative_prompt", type=str, 
                       default="The video is not of a high quality, it has a low resolution. "
                               "Watermark present in each frame. The background is solid. "
                               "Strange body and strange trajectory. Distortion.",
                       help="Negative prompt")
    
    # Inference params
    parser.add_argument("--guidance_scale", type=float, default=6.0, help="Guidance scale")
    parser.add_argument("--seed", type=int, default=43, help="Random seed")
    parser.add_argument("--num_inference_steps", type=int, default=50, help="Number of inference steps")
    
    # Output
    parser.add_argument("--save_path", type=str, default="samples/cogvideox-fun-videos_i2v", 
                       help="Path to save generated videos")
    
    # Test set (JSON)
    parser.add_argument("--test_json_path", type=str, default=None, 
                       help="Path to JSON file containing test samples")
    
    return parser.parse_args()

def load_ref_video_frames(ref_video_path, target_size, num_frames=None):
    """
    Load reference video frames from video file or image directory
    
    Args:
        ref_video_path: Path to video file or directory containing image frames
        target_size: Target size [height, width] for resizing
        num_frames: Number of frames to load (if None, load all available)
    
    Returns:
        ref_pixel_values: Tensor of shape [B, F, C, H, W] in 0~1 range
    """
    import cv2
    
    target_h, target_w = target_size
    
    ref_frames = []
    
    if os.path.isfile(ref_video_path):
        cap = cv2.VideoCapture(ref_video_path)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        if num_frames is None:
            num_frames = frame_count
        
        for i in range(num_frames):
            ret, frame = cap.read()
            if not ret:
                break
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = Image.fromarray(frame)
            frame = frame.resize((target_w, target_h))
            ref_frames.append(np.array(frame))
        
        cap.release()
    elif os.path.isdir(ref_video_path):
        image_files = sorted([f for f in os.listdir(ref_video_path) if f.endswith(('.png', '.jpg', '.jpeg'))])
        
        if num_frames is None:
            num_frames = len(image_files)
        
        for i in range(num_frames):
            if i >= len(image_files):
                break
            frame = Image.open(os.path.join(ref_video_path, image_files[i])).convert('RGB')
            frame = frame.resize((target_w, target_h))
            ref_frames.append(np.array(frame))
    
    if len(ref_frames) == 0:
        raise ValueError(f"No frames loaded from {ref_video_path}")
    
    ref_frames_np = np.array(ref_frames)
    ref_frames_tensor = torch.from_numpy(ref_frames_np).float()
    ref_frames_tensor = ref_frames_tensor / 255.0
    
    ref_frames_tensor = ref_frames_tensor.unsqueeze(0)
    
    print(f"DEBUG: ref_frames_tensor shape after unsqueeze(0): {ref_frames_tensor.shape}")
    
    return ref_frames_tensor


def get_model_device(model):
    """安全获取模型实际设备，处理 CPU offload 模式"""
    try:
        params = list(model.parameters())
        if len(params) > 0:
            return params[0].device
        buffers = list(model.buffers())
        if len(buffers) > 0:
            return buffers[0].device
    except Exception:
        pass
    return torch.device("cpu")


def process_sample(pipeline, vae, device, weight_dtype, hypernetwork, lilora_network, sample_info, sample_size, args):
    """
    Process a single test sample
    
    Args:
        pipeline: CogVideoX pipeline
        vae: VAE model
        device: device
        weight_dtype: weight dtype
        hypernetwork: VideoHyperDream model (or None)
        lilora_network: LoRANetwork model (or None)
        sample_info: dict with first_frame_path, ref_video_path, text, video_name
        sample_size: [height, width]
        args: other arguments
    
    Returns:
        sample: generated video tensor
    """
    first_frame_path = sample_info.get('first_frame_path')
    ref_video_path = sample_info.get('ref_video_path')
    prompt = sample_info.get('text', '')
    video_name = sample_info.get('video_name', 'sample')
    
    # 如果 JSON 中没有提供 prompt，使用命令行参数中的默认值
    if not prompt or not prompt.strip():
        prompt = args.get('prompt', '')
        if prompt:
            print(f"Using prompt from command line: {prompt[:50]}...")
        else:
            print(f"Warning: No prompt provided for sample {video_name}")
            return None
    
    if first_frame_path is None or not os.path.exists(first_frame_path):
        print(f"Warning: first_frame_path not found: {first_frame_path}")
        return None
    
    input_video, input_video_mask, clip_image = get_image_to_video_latent(
        first_frame_path, None, video_length=args['video_length'], sample_size=sample_size
    )
    
    original_image = Image.open(first_frame_path).convert('RGB')
    original_width, original_height = original_image.size
    print(f"\n[Debug Input Video Processing]")
    print(f"[Debug Input] first_frame_path: {first_frame_path}")
    print(f"[Debug Input] Original image size: ({original_width}, {original_height})")
    print(f"[Debug Input] sample_size (after resize): {sample_size}")
    print(f"[Debug Input] input_video type: {type(input_video)}")
    if isinstance(input_video, torch.Tensor):
        print(f"[Debug Input] input_video shape: {input_video.shape}")
        if input_video.dim() == 5:
            B, C, F, H, W = input_video.shape
            print(f"[Debug Input] Input tensor: B={B}, C={C}, F={F}, H={H}, W={W}")
            print(f"[Debug Input] Height matches sample_size[0]: {H == sample_size[0]}, Width matches sample_size[1]: {W == sample_size[1]}")
    if input_video_mask is not None and isinstance(input_video_mask, torch.Tensor):
        print(f"[Debug Input] input_video_mask shape: {input_video_mask.shape}")
    print("-" * 40)
    
    if hypernetwork is not None and lilora_network is not None and ref_video_path is not None and os.path.exists(ref_video_path):
        print(f"Generating weights from reference video: {ref_video_path}")
        ref_frames_tensor = load_ref_video_frames(ref_video_path, sample_size)
        B, F, H, W, C = ref_frames_tensor.shape
        ref_pixel_values = ref_frames_tensor.permute(0, 4, 1, 2, 3).contiguous()
        ref_pixel_values = ref_pixel_values.to(dtype=weight_dtype)

        with torch.no_grad():
            vae_device = get_model_device(vae)
            ref_pixel_values = ref_pixel_values.to(device=vae_device, dtype=weight_dtype)
            ref_video_features = vae.encode(ref_pixel_values).latent_dist.sample()
            ref_video_features = ref_video_features * vae.config.scaling_factor
        
        _, weight_list = hypernetwork(ref_video_features)
        
        for i, (weight, lora_layer) in enumerate(zip(weight_list, lilora_network.unet_loras)):
            if weight.dim() == 3:
                weight = weight.view(weight.size(0), -1)
            elif weight.dim() == 2:
                if weight.size(0) == 1:
                    weight = weight.view(-1)
                else:
                    weight = weight.view(weight.size(0), -1)
            elif weight.dim() == 1:
                weight = weight.view(-1)
            lora_layer.update_weight(weight)
            lora_layer.down_aux.data = lora_layer.down_aux.data.to(dtype=weight.dtype)
            lora_layer.up_aux.data = lora_layer.up_aux.data.to(dtype=weight.dtype)
        
        print(f"Updated {len(weight_list)} LiLoRA layers with hypernetwork weights from {F} reference frames.")
    
    generator = torch.Generator(device=device).manual_seed(args.get('seed', 43))
    
    with torch.no_grad():
        sample = pipeline(
            prompt,
            num_frames=args['video_length'],
            negative_prompt=args.get('negative_prompt', ''),
            height=sample_size[0],
            width=sample_size[1],
            generator=generator,
            guidance_scale=args.get('guidance_scale', 6.0),
            num_inference_steps=args.get('num_inference_steps', 50),
            video=input_video,
            mask_video=input_video_mask
        ).videos

    print(f"\n[Debug Pipeline Output @ Inference]")
    print(f"[Debug Pipeline] Original image size: ({original_width}, {original_height})")
    print(f"[Debug Pipeline] Sample size requested: ({sample_size[0]}, {sample_size[1]})")
    print(f"[Debug Pipeline] Pipeline output shape: {sample.shape}")
    
    if isinstance(sample, torch.Tensor):
        if sample.dim() == 5:
            B, C, F, H, W = sample.shape
            print(f"[Debug Pipeline] Output tensor shape: B={B}, C={C}, F={F}, H={H}, W={W}")
            
            expected_H = original_height
            expected_W = original_width
            print(f"\n[Debug Size Verification]")
            print(f"[Debug] Comparing: Original image ({original_height}, {original_width}) vs Output video ({H}, {W})")
            
            if H == expected_H and W == expected_W:
                print(f"[Debug] ✓ Size MATCH! Output matches original image size")
            else:
                print(f"[Debug] ✗ Size MISMATCH!")
                print(f"[Debug]   Expected: ({expected_H}, {expected_W})")
                print(f"[Debug]   Got:      ({H}, {W})")
                print(f"[Debug]   Difference: H={H-expected_H} ({((H-expected_H)/expected_H*100):.1f}%), W={W-expected_W} ({((W-expected_W)/expected_W*100):.1f}%)")
            
            if H != expected_H or W != expected_W:
                print(f"\n[Debug Analysis]")
                print(f"[Debug] VAE config - scaling_factor: {vae.config.scaling_factor}")
                print(f"[Debug] VAE config - latent_channels: {vae.config.latent_channels}")
                if hasattr(vae, 'config') and hasattr(vae.config, 'decoder'):
                    print(f"[Debug] VAE decoder will upsample by factor of 4x or 8x")
        else:
            print(f"[Debug Pipeline] Unexpected tensor dimensions: {sample.dim()}")
    
    print("-" * 80)
    
    return sample

def main():
    args = parse_args()
    
    # Convert weight dtype string to torch dtype
    weight_dtype_map = {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32
    }
    weight_dtype = weight_dtype_map.get(args.weight_dtype, torch.bfloat16)
    
    # GPU memory mode, which can be chosen in [model_full_load, model_full_load_and_qfloat8, model_cpu_offload, model_cpu_offload_and_qfloat8, sequential_cpu_offload].
    # model_full_load means that the entire model will be moved to the GPU.
    # 
    # model_full_load_and_qfloat8 means that the entire model will be moved to the GPU,
    # and the transformer model has been quantized to float8, which can save more GPU memory. 
    # 
    # model_cpu_offload means that the entire model will be moved to the CPU after use, which can save some GPU memory.
    # 
    # model_cpu_offload_and_qfloat8 indicates that the entire model will be moved to the CPU after use, 
    # and the transformer model has been quantized to float8, which can save more GPU memory. 
    # 
    # sequential_cpu_offload means that each layer of the model will be moved to the CPU after use, 
    # resulting in slower speeds but saving a large amount of GPU memory.
    GPU_memory_mode     = args.gpu_memory_mode
    # Multi GPUs config
    # Please ensure that the product of ulysses_degree and ring_degree equals the number of GPUs used. 
    # For example, if you are using 8 GPUs, you can set ulysses_degree = 2 and ring_degree = 4.
    # If you are using 1 GPU, you can set ulysses_degree = 1 and ring_degree = 1.
    ulysses_degree      = args.ulysses_degree
    ring_degree         = args.ring_degree
    # Use FSDP to save more GPU memory in multi gpus.
    fsdp_dit            = args.fsdp_dit
    fsdp_text_encoder   = args.fsdp_text_encoder
    # Compile will give a speedup in fixed resolution and need a little GPU memory. 
    # The compile_dit is not compatible with the fsdp_dit and sequential_cpu_offload.
    compile_dit         = args.compile_dit
    
    # Config and model path
    model_name          = args.model_name
    
    # Choose the sampler in "Euler" "Euler A" "DPM++" "PNDM" "DDIM_Cog" and "DDIM_Origin"
    sampler_name        = args.sampler_name
    
    # Load pretrained model if need
    transformer_path    = args.transformer_path
    vae_path            = args.vae_path
    
    # LiLoRA and HyperNetwork paths
    lilora_path         = args.lilora_path
    hypernetwork_path   = args.hypernetwork_path
    
    # Input data paths
    first_frame_path    = args.first_frame_path
    ref_video_path      = args.ref_video_path
    
    # LiLoRA params (should match training config)
    lora_rank           = args.lora_rank
    lora_down_dim       = args.lora_down_dim
    lora_up_dim         = args.lora_up_dim
    lora_weight         = args.lora_weight
    
    # Other params
    sample_size         = [384, 672]
    # V1.0 and V1.1 support up to 49 frames of video generation,
    # while V1.5 supports up to 85 frames.  
    video_length        = args.video_length
    fps                 = args.fps
    
    # If you want to generate ultra long videos, please set partial_video_length as the length of each sub video segment
    partial_video_length = args.partial_video_length
    overlap_video_length = args.overlap_video_length
    
    # If you want to generate from text, please set the validation_image_start = None and validation_image_end = None
    validation_image_start  = None  # Will be set from first_frame_path below
    validation_image_end    = None
    
    # Use first_frame_path for image-to-video generation
    # If first_frame_path is provided, it will be used as the starting frame
    # and sample_size will be set to match the first frame dimensions
    # Note: If test_json_path is provided, first_frame_path will be read from JSON
    if args.test_json_path is None:  # Only check when NOT using test JSON
        if first_frame_path is not None and os.path.exists(first_frame_path):
            first_frame = Image.open(first_frame_path).convert('RGB')
            W, H = first_frame.size
            sample_size = [H, W]  # PIL image.size is (W, H), sample_size should be [H, W]
            validation_image_start = first_frame_path
            validation_image_end = None
            print(f"Using first frame from: {first_frame_path}, sample_size: {sample_size}")
        else:
            raise ValueError("first_frame_path must be provided and exist for direct I2V inference")
    
    # HyperNetwork reference video path (required when using hypernetwork)
    # Reference video should be a video file path or a list of image frames
    # The reference video will be encoded by VAE and used by HyperNetwork to generate LiLoRA weights
    ref_video_path      = args.ref_video_path
    
    # prompts
    prompt                  = args.prompt
    negative_prompt         = args.negative_prompt
    guidance_scale          = args.guidance_scale
    seed                    = args.seed
    num_inference_steps     = args.num_inference_steps
    lora_weight             = args.lora_weight
    save_path               = args.save_path

    # 初始化多 GPU 设备
    device = set_multi_gpus_devices(ulysses_degree, ring_degree)

    transformer = CogVideoXTransformer3DModel.from_pretrained(
        model_name, 
        subfolder="transformer",
        low_cpu_mem_usage=True,
        torch_dtype=weight_dtype,
    ).to(weight_dtype)

    if transformer_path is not None:
        print(f"From checkpoint: {transformer_path}")
        if transformer_path.endswith("safetensors"):
            from safetensors.torch import load_file, safe_open
            state_dict = load_file(transformer_path)
        else:
            state_dict = torch.load(transformer_path, map_location="cpu")
        state_dict = state_dict["state_dict"] if "state_dict" in state_dict else state_dict

        m, u = transformer.load_state_dict(state_dict, strict=False)
        print(f"missing keys: {len(m)}, unexpected keys: {len(u)}")

    # Get Vae
    vae = AutoencoderKLCogVideoX.from_pretrained(
        model_name, 
        subfolder="vae"
    ).to(weight_dtype)
    
    vae.enable_tiling()

    if vae_path is not None:
        print(f"From checkpoint: {vae_path}")
        if vae_path.endswith("safetensors"):
            from safetensors.torch import load_file, safe_open
            state_dict = load_file(vae_path)
        else:
            state_dict = torch.load(vae_path, map_location="cpu")
        state_dict = state_dict["state_dict"] if "state_dict" in state_dict else state_dict

        m, u = vae.load_state_dict(state_dict, strict=False)
        print(f"missing keys: {len(m)}, unexpected keys: {len(u)}")

    # Get tokenizer and text_encoder
    tokenizer = T5Tokenizer.from_pretrained(
        model_name, subfolder="tokenizer"
    )
    text_encoder = T5EncoderModel.from_pretrained(
        model_name, subfolder="text_encoder", torch_dtype=weight_dtype
    )

    # Get Scheduler
    Chosen_Scheduler = scheduler_dict = {
        "Euler": EulerDiscreteScheduler,
        "Euler A": EulerAncestralDiscreteScheduler,
        "DPM++": DPMSolverMultistepScheduler, 
        "PNDM": PNDMScheduler,
        "DDIM_Cog": CogVideoXDDIMScheduler,
        "DDIM_Origin": DDIMScheduler,
    }[sampler_name]
    scheduler = Chosen_Scheduler.from_pretrained(
        model_name, 
        subfolder="scheduler"
    )

    if transformer.config.in_channels != vae.config.latent_channels:
        pipeline = CogVideoXFunInpaintPipeline(
            vae=vae,
            tokenizer=tokenizer,
            text_encoder=text_encoder,
            transformer=transformer,
            scheduler=scheduler,
        )
    else:
        pipeline = CogVideoXFunPipeline(
            vae=vae,
            tokenizer=tokenizer,
            text_encoder=text_encoder,
            transformer=transformer,
            scheduler=scheduler,
        )
    if ulysses_degree > 1 or ring_degree > 1:
        from functools import partial
        transformer.enable_multi_gpus_inference()
        if fsdp_dit:
            shard_fn = partial(shard_model, device_id=device, param_dtype=weight_dtype)
            pipeline.transformer = shard_fn(pipeline.transformer)
            print("Add FSDP DIT")
        if fsdp_text_encoder:
            shard_fn = partial(shard_model, device_id=device, param_dtype=weight_dtype)
            pipeline.text_encoder = shard_fn(pipeline.text_encoder)
            print("Add FSDP TEXT ENCODER")

    if compile_dit:
        for i in range(len(pipeline.transformer.transformer_blocks)):
            pipeline.transformer.transformer_blocks[i] = torch.compile(pipeline.transformer.transformer_blocks[i])
        print("Add Compile")

    if GPU_memory_mode == "sequential_cpu_offload":
        pipeline.enable_sequential_cpu_offload(device=device)
    elif GPU_memory_mode == "model_cpu_offload_and_qfloat8":
        convert_model_weight_to_float8(transformer, exclude_module_name=[], device=device)
        convert_weight_dtype_wrapper(transformer, weight_dtype)
        pipeline.enable_model_cpu_offload(device=device)
    elif GPU_memory_mode == "model_cpu_offload":
        pipeline.enable_model_cpu_offload(device=device)
    elif GPU_memory_mode == "model_full_load_and_qfloat8":
        convert_model_weight_to_float8(transformer, exclude_module_name=[], device=device)
        convert_weight_dtype_wrapper(transformer, weight_dtype)
        pipeline.to(device=device)
    else:
        pipeline.to(device=device)

    generator = torch.Generator(device=device).manual_seed(seed)

    lilora_network = None
    if lilora_path is not None:
        lora_weight_dim = (lora_down_dim + lora_up_dim) * lora_rank
        
        lilora_network = create_network(
            multiplier=lora_weight,
            network_dim=lora_rank,
            network_alpha=lora_weight,
            text_encoder=pipeline.text_encoder,
            transformer=pipeline.transformer,
            down_dim=lora_down_dim,
            up_dim=lora_up_dim,
            is_train=False,
        )
        lilora_network.apply_to(pipeline.text_encoder, pipeline.transformer, apply_text_encoder=False, apply_unet=True)
        
        from safetensors.torch import load_file
        lora_state_dict = load_file(lilora_path)
        info = lilora_network.load_weights(lilora_path)
        print(f"LiLoRA weights loaded from {lilora_path} (missing: {len(info.missing_keys)}, unexpected: {len(info.unexpected_keys)})")
        
        print(f"Total LoRA modules in UNet: {len(lilora_network.unet_loras)}")

    hypernetwork = None
    if hypernetwork_path is not None and lilora_network is not None:
        from safetensors.torch import load_file
        
        lora_weight_dim = (lora_down_dim + lora_up_dim) * lora_rank
        weight_num = len(lilora_network.unet_loras)
        
        print(f"Creating VideoHyperDream with weight_num={weight_num}, weight_dim={lora_weight_dim}")
        
        hypernetwork = VideoHyperDream(
            video_feat_dim=16,
            weight_num=weight_num,
            weight_dim=lora_weight_dim,
        ).to(device=device, dtype=weight_dtype)
        
        hyper_state_dict = load_file(hypernetwork_path)
        hypernetwork.load_state_dict(hyper_state_dict, strict=False)
        hypernetwork.eval()
        
        if lilora_network is not None and len(lilora_network.unet_loras) > 0:
            hypernetwork.set_lilora(lilora_network.unet_loras)
            hypernetwork.set_device(device)
            print(f"HyperNetwork loaded. Connected to {len(lilora_network.unet_loras)} LiLoRA layers from create_network.")
        else:
            print("Warning: No LiLoRA layers found from create_network!")
            hypernetwork = None

    def save_results(sample, video_length, video_name=None):
        if not os.path.exists(save_path):
            os.makedirs(save_path, exist_ok=True)

        if video_name is None:
            index = len([path for path in os.listdir(save_path)]) + 1
            prefix = str(index).zfill(8)
            video_name = prefix

        if video_length == 1:
            video_path = os.path.join(save_path, video_name + ".png")

            image = sample[0, :, 0]
            image = image.transpose(0, 1).transpose(1, 2)
            image = (image * 255).numpy().astype(np.uint8)
            image = Image.fromarray(image)
            image.save(video_path)
        else:
            video_path = os.path.join(save_path, video_name + ".mp4")
            save_videos_grid(sample, video_path, fps=fps)
        
        print(f"Results saved to: {video_path}")
        return video_path

    def process_test_set():
        if args.test_json_path is None or not os.path.exists(args.test_json_path):
            print("No test JSON path provided or file not found.")
            return
        
        json_dir = os.path.dirname(os.path.abspath(args.test_json_path))
        
        with open(args.test_json_path, 'r') as f:
            test_data = json.load(f)
        
        if isinstance(test_data, list):
            samples = test_data
        elif isinstance(test_data, dict):
            samples = test_data.get('samples', [])
        else:
            print(f"Unknown test data format in {args.test_json_path}")
            return
        
        print(f"Processing {len(samples)} test samples...")
        
        for idx, sample_info in enumerate(samples):
            print(f"\n{'='*60}")
            print(f"Processing sample {idx + 1}/{len(samples)}: {sample_info.get('video_name', f'sample_{idx}')}")
            print(f"{'='*60}")
            
            try:
                sample_first_frame = sample_info.get('first_frame_path')
                ref_video_path = sample_info.get('ref_video_path')
                
                # Fix first_frame_path - JSON paths are relative to Omni-Effects-main base directory
                if sample_first_frame:
                    if not os.path.isabs(sample_first_frame):
                        # Simply join with base directory (Omni-Effects-main)
                        sample_first_frame = os.path.join('/home/xhm001/Desktop/Code/Omni-Effects-main', sample_first_frame)
                    
                    if os.path.exists(sample_first_frame):
                        first_frame = Image.open(sample_first_frame).convert('RGB')
                        W, H = first_frame.size
                        current_sample_size = [H, W]  # PIL image.size is (W, H), sample_size should be [H, W]
                        print(f"Dynamic sample_size: {current_sample_size}")
                        
                        # Update sample_info with the fixed path
                        sample_info_copy = sample_info.copy()
                        sample_info_copy['first_frame_path'] = sample_first_frame
                    else:
                        raise FileNotFoundError(f"first_frame_path not found: {sample_first_frame}")
                else:
                    raise ValueError("first_frame_path must be provided for I2V inference")
                
                # Fix ref_video_path if it's a relative path
                if ref_video_path and not os.path.isabs(ref_video_path):
                    ref_video_path = os.path.join('/home/xhm001/Desktop/Code/Omni-Effects-main', ref_video_path)
                    sample_info_copy['ref_video_path'] = ref_video_path
                
                sample_args = {
                    'video_length': args.video_length,
                    'seed': args.seed + idx,
                    'negative_prompt': args.negative_prompt,
                    'guidance_scale': args.guidance_scale,
                    'num_inference_steps': args.num_inference_steps,
                }
                
                sample = process_sample(
                    pipeline=pipeline,
                    vae=vae,
                    device=device,
                    weight_dtype=weight_dtype,
                    hypernetwork=hypernetwork,
                    lilora_network=lilora_network,
                    sample_info=sample_info_copy,
                    sample_size=current_sample_size,
                    args=sample_args
                )
                
                if sample is not None:
                    video_name = sample_info.get('video_name', f'sample_{idx}')
                    save_results(sample, sample_args['video_length'], video_name)
                else:
                    print(f"Skipping sample {idx + 1} due to missing first frame.")
                
            except Exception as e:
                print(f"Error processing sample {idx + 1}: {str(e)}")
                import traceback
                traceback.print_exc()
        
        print(f"\nAll {len(samples)} samples processed.")

    if args.test_json_path is not None:
        process_test_set()
    else:
        if partial_video_length is not None:
            partial_video_length = int((partial_video_length - 1) // vae.config.temporal_compression_ratio * vae.config.temporal_compression_ratio) + 1 if video_length != 1 else 1
            latent_frames = (partial_video_length - 1) // vae.config.temporal_compression_ratio + 1
            if partial_video_length != 1 and transformer.config.patch_size_t is not None and latent_frames % transformer.config.patch_size_t != 0:
                additional_frames = transformer.config.patch_size_t - latent_frames % transformer.config.patch_size_t
                partial_video_length += additional_frames * vae.config.temporal_compression_ratio
                
            init_frames = 0
            last_frames = init_frames + partial_video_length
            while init_frames < video_length:
                if last_frames >= video_length:
                    _partial_video_length = video_length - init_frames
                    _partial_video_length = int((_partial_video_length - 1) // vae.config.temporal_compression_ratio * vae.config.temporal_compression_ratio) + 1
                    latent_frames = (_partial_video_length - 1) // vae.config.temporal_compression_ratio + 1
                    if _partial_video_length != 1 and transformer.config.patch_size_t is not None and latent_frames % transformer.config.patch_size_t != 0:
                        additional_frames = transformer.config.patch_size_t - latent_frames % transformer.config.patch_size_t
                        _partial_video_length += additional_frames * vae.config.temporal_compression_ratio

                    if _partial_video_length <= 0:
                        break
                else:
                    _partial_video_length = partial_video_length

                input_video, input_video_mask, clip_image = get_image_to_video_latent(validation_image, None, video_length=_partial_video_length, sample_size=sample_size)
                
                if hypernetwork is not None and ref_video_path is not None and init_frames == 0:
                    print(f"Loading reference video from {ref_video_path}...")
                    ref_frames_tensor = load_ref_video_frames(ref_video_path, sample_size)
                    B, F, H, W, C = ref_frames_tensor.shape
                    ref_pixel_values = ref_frames_tensor.permute(0, 4, 1, 2, 3).contiguous()

                    with torch.no_grad():
                        vae_device = get_model_device(vae)
                        ref_pixel_values = ref_pixel_values.to(device=vae_device, dtype=weight_dtype)
                        ref_video_features = vae.encode(ref_pixel_values).latent_dist.sample()
                        ref_video_features = ref_video_features * vae.config.scaling_factor
                    
                    _, weight_list = hypernetwork(ref_video_features)
                    
                    for i, (weight, lora_layer) in enumerate(zip(weight_list, lilora_network.unet_loras)):
                        if weight.dim() == 3:
                            weight = weight.view(weight.size(0), -1)
                        elif weight.dim() == 2:
                            if weight.size(0) == 1:
                                weight = weight.view(-1)
                            else:
                                weight = weight.view(weight.size(0), -1)
                        elif weight.dim() == 1:
                            weight = weight.view(-1)
                        lora_layer.update_weight(weight)
                    
                    print(f"Updated {len(weight_list)} LiLoRA layers with hypernetwork weights from {F} reference frames.")
                
                with torch.no_grad():
                    sample = pipeline(
                        args.prompt, 
                        num_frames = _partial_video_length,
                        negative_prompt = negative_prompt,
                        height      = sample_size[0],
                        width       = sample_size[1],
                        generator   = generator,
                        guidance_scale = guidance_scale,
                        num_inference_steps = num_inference_steps,

                        video        = input_video,
                        mask_video   = input_video_mask
                    ).videos
                
                if init_frames != 0:
                    mix_ratio = torch.from_numpy(
                        np.array([float(_index) / float(overlap_video_length) for _index in range(overlap_video_length)], np.float32)
                    ).unsqueeze(0).unsqueeze(0).unsqueeze(-1).unsqueeze(-1)
                    
                    new_sample[:, :, -overlap_video_length:] = new_sample[:, :, -overlap_video_length:] * (1 - mix_ratio) + \
                        sample[:, :, :overlap_video_length] * mix_ratio
                    new_sample = torch.cat([new_sample, sample[:, :, overlap_video_length:]], dim = 2)

                    sample = new_sample
                else:
                    new_sample = sample

                if last_frames >= video_length:
                    break

                validation_image = [
                    Image.fromarray(
                        (sample[0, :, _index].transpose(0, 1).transpose(1, 2) * 255).numpy().astype(np.uint8)
                    ) for _index in range(-overlap_video_length, 0)
                ]

                init_frames = init_frames + _partial_video_length - overlap_video_length
                last_frames = init_frames + _partial_video_length
        else:
            video_length = int((video_length - 1) // vae.config.temporal_compression_ratio * vae.config.temporal_compression_ratio) + 1 if video_length != 1 else 1
        latent_frames = (video_length - 1) // vae.config.temporal_compression_ratio + 1
        if video_length != 1 and transformer.config.patch_size_t is not None and latent_frames % transformer.config.patch_size_t != 0:
            additional_frames = transformer.config.patch_size_t - latent_frames % transformer.config.patch_size_t
            video_length += additional_frames * vae.config.temporal_compression_ratio
        input_video, input_video_mask, clip_image = get_image_to_video_latent(validation_image_start, validation_image_end, video_length=video_length, sample_size=sample_size)

        if hypernetwork is not None and ref_video_path is not None:
            print(f"Loading reference video from {ref_video_path}...")
            ref_frames_tensor = load_ref_video_frames(ref_video_path, sample_size)
            B, F, H, W, C = ref_frames_tensor.shape
            ref_pixel_values = ref_frames_tensor.permute(0, 4, 1, 2, 3).contiguous()

            with torch.no_grad():
                vae_device = get_model_device(vae)
                ref_pixel_values = ref_pixel_values.to(device=vae_device, dtype=weight_dtype)
                ref_video_features = vae.encode(ref_pixel_values).latent_dist.sample()
                ref_video_features = ref_video_features * vae.config.scaling_factor
            
            _, weight_list = hypernetwork(ref_video_features)
            
            for i, (weight, lora_layer) in enumerate(zip(weight_list, lilora_network.unet_loras)):
                if weight.dim() == 3:
                    weight = weight.view(weight.size(0), -1)
                elif weight.dim() == 2:
                    if weight.size(0) == 1:
                        weight = weight.view(-1)
                    else:
                        weight = weight.view(weight.size(0), -1)
                elif weight.dim() == 1:
                    weight = weight.view(-1)
                lora_layer.update_weight(weight)
            
            print(f"Updated {len(weight_list)} LiLoRA layers with hypernetwork weights from {F} reference frames.")

        with torch.no_grad():
            sample = pipeline(
                args.prompt, 
                num_frames = video_length,
                negative_prompt = negative_prompt,
                height      = sample_size[0],
                width       = sample_size[1],
                generator   = generator,
                guidance_scale = guidance_scale,
                num_inference_steps = num_inference_steps,

                video        = input_video,
                mask_video   = input_video_mask
            ).videos

    if args.test_json_path is not None:
        process_test_set()
    else:
        if ulysses_degree * ring_degree > 1:
            import torch.distributed as dist
            if dist.get_rank() == 0:
                save_results(sample, video_length)
        else:
            save_results(sample, video_length)

if __name__ == "__main__":
    main()