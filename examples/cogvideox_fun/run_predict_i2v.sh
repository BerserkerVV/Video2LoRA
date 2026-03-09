#!/bin/bash

# CogVideoX-Fun I2V Inference Script
# Usage: bash run_predict_i2v.sh [OPTIONS]

# =============================================================================
# 基础设置 (Basic Settings)
# =============================================================================
export CUDA_VISIBLE_DEVICES=6
export PYTHONPATH="/home/xhm001/Desktop/Code/VideoX-Fun-main:${PYTHONPATH}"

# 模型路径 (Model Path)
MODEL_NAME="/home/xhm001/Desktop/Code/VideoX-Fun-main/models/cogvideoxfun/models--alibaba-pai--CogVideoX-Fun-V1.1-5b-InP"

# =============================================================================
# LiLoRA 和 HyperNetwork 路径 (LiLoRA & HyperNetwork Paths)
# =============================================================================
# LiLoRA 权重路径
LILORA_PATH="/home/xhm001/Desktop/Code/VideoX-Fun-main/Hyper_150/checkpoint-14800/checkpoint-checkpoint-14800.safetensors"

# HyperNetwork 权重路径
HYPERNETWORK_PATH="/home/xhm001/Desktop/Code/VideoX-Fun-main/Hyper_150/checkpoint-14800/hypernetwork.safetensors"

# LiLoRA 训练参数 (应与训练配置匹配)
LORA_RANK=1
LORA_DOWN_DIM=100
LORA_UP_DIM=50
LORA_WEIGHT=1


# =============================================================================
# 输入数据路径 (Input Data Paths)
# =============================================================================
# 注意：当使用 --test_json_path 时，以下参数会从 JSON 中读取
# 单样本推理时，可以手动指定以下参数

# 首帧图像路径 (用于图像到视频生成)
FIRST_FRAME_PATH=None

# 参考视频路径 (用于 HyperNetwork 生成权重)
REF_VIDEO_PATH=None

# 测试集 JSON 路径 (可选，批量处理时使用)
TEST_JSON_PATH=/home/xhm001/Desktop/Code/VideoX-Fun-main/Open_VFX_for_evl/cake.json

# =============================================================================
# 视频生成参数 (Video Generation Parameters)
# =============================================================================
VIDEO_LENGTH=49
FPS=8
PARTIAL_VIDEO_LENGTH=None
OVERLAP_VIDEO_LENGTH=4

# =============================================================================
# 推理参数 (Inference Parameters)
# =============================================================================
# 注意：当使用 --test_json_path 时，以下参数会从 JSON 中读取
PROMPT=None
NEGATIVE_PROMPT="The video is not of a high quality, it has a low resolution. Watermark present in each frame. The background is solid. Strange body and strange trajectory. Distortion."
GUIDANCE_SCALE=6.0
SEED=43
NUM_INFERENCE_STEPS=50

# =============================================================================
# GPU 设置 (GPU Settings)
# =============================================================================
# GPU 内存模式: model_full_load, model_full_load_and_qfloat8, model_cpu_offload, model_cpu_offload_and_qfloat8, sequential_cpu_offload
GPU_MEMORY_MODE="model_full_load"

# 多 GPU 设置
ULYSSES_DEGREE=1
RING_DEGREE=1
FSPD_DIT=False
FSPD_TEXT_ENCODER=False
COMPILE_DIT=False

# 权重数据类型: bfloat16, float16, float32
WEIGHT_DTYPE="bfloat16"

# =============================================================================
# 采样器选择 (Sampler Selection)
# =============================================================================
# 可选: Euler, Euler A, DPM++, PNDM, DDIM_Cog, DDIM_Origin
SAMPLER_NAME="DDIM_Origin"

# =============================================================================
# 输出路径 (Output Path)
# =============================================================================
SAVE_PATH="samples/ALL"

# =============================================================================
# 解析命令行参数 (Parse Command Line Arguments)
# =============================================================================
while [[ $# -gt 0 ]]; do
    case $1 in
        --model_name)
            MODEL_NAME="$2"
            shift 2
            ;;
        --lilora_path)
            LILORA_PATH="$2"
            shift 2
            ;;
        --hypernetwork_path)
            HYPERNETWORK_PATH="$2"
            shift 2
            ;;
        --first_frame_path)
            FIRST_FRAME_PATH="$2"
            shift 2
            ;;
        --ref_video_path)
            REF_VIDEO_PATH="$2"
            shift 2
            ;;
        --test_json_path)
            TEST_JSON_PATH="$2"
            shift 2
            ;;
        --video_length)
            VIDEO_LENGTH="$2"
            shift 2
            ;;
        --fps)
            FPS="$2"
            shift 2
            ;;
        --prompt)
            PROMPT="$2"
            shift 2
            ;;
        --negative_prompt)
            NEGATIVE_PROMPT="$2"
            shift 2
            ;;
        --guidance_scale)
            GUIDANCE_SCALE="$2"
            shift 2
            ;;
        --seed)
            SEED="$2"
            shift 2
            ;;
        --num_inference_steps)
            NUM_INFERENCE_STEPS="$2"
            shift 2
            ;;
        --lora_rank)
            LORA_RANK="$2"
            shift 2
            ;;
        --lora_down_dim)
            LORA_DOWN_DIM="$2"
            shift 2
            ;;
        --lora_up_dim)
            LORA_UP_DIM="$2"
            shift 2
            ;;
        --lora_weight)
            LORA_WEIGHT="$2"
            shift 2
            ;;
        --save_path)
            SAVE_PATH="$2"
            shift 2
            ;;
        --gpu_memory_mode)
            GPU_MEMORY_MODE="$2"
            shift 2
            ;;
        --weight_dtype)
            WEIGHT_DTYPE="$2"
            shift 2
            ;;
        --sampler_name)
            SAMPLER_NAME="$2"
            shift 2
            ;;
        --ulysses_degree)
            ULYSSES_DEGREE="$2"
            shift 2
            ;;
        --ring_degree)
            RING_DEGREE="$2"
            shift 2
            ;;
        --partial_video_length)
            PARTIAL_VIDEO_LENGTH="$2"
            shift 2
            ;;
        --overlap_video_length)
            OVERLAP_VIDEO_LENGTH="$2"
            shift 2
            ;;
        --help|-h)
            echo "Usage: bash run_predict_i2v.sh [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  --model_name            模型路径 (默认: models/Diffusion_Transformer/CogVideoX-Fun-V1.1-2b-InP)"
            echo "  --lilora_path           LiLoRA 权重路径"
            echo "  --hypernetwork_path     HyperNetwork 权重路径"
            echo "  --first_frame_path      首帧图像路径"
            echo "  --ref_video_path        参考视频路径 (用于 HyperNetwork)"
            echo "  --test_json_path        测试集 JSON 路径"
            echo "  --video_length          生成视频帧数 (默认: 49)"
            echo "  --fps                   输出视频 FPS (默认: 8)"
            echo "  --prompt                生成提示词"
            echo "  --negative_prompt       负向提示词"
            echo "  --guidance_scale        引导 scale (默认: 6.0)"
            echo "  --seed                  随机种子 (默认: 43)"
            echo "  --num_inference_steps   推理步数 (默认: 50)"
            echo "  --lora_rank             LoRA rank (默认: 1)"
            echo "  --lora_down_dim         LoRA down dimension (默认: 200)"
            echo "  --lora_up_dim           LoRA up dimension (默认: 100)"
            echo "  --lora_weight           LoRA 权重 (默认: 0.55)"
            echo "  --save_path             输出路径 (默认: samples/cogvideox-fun-videos_i2v)"
            echo "  --gpu_memory_mode       GPU 内存模式"
            echo "  --weight_dtype          权重数据类型"
            echo "  --sampler_name          采样器名称"
            echo "  --help                  显示帮助信息"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

# =============================================================================
# 构建命令行参数 (Build Command Line Arguments)
# =============================================================================
PYTHON_ARGS=()

# GPU 设置
PYTHON_ARGS+=("--gpu_memory_mode" "${GPU_MEMORY_MODE}")
PYTHON_ARGS+=("--ulysses_degree" "${ULYSSES_DEGREE}")
PYTHON_ARGS+=("--ring_degree" "${RING_DEGREE}")

[[ "${FSPD_DIT}" == "True" ]] && PYTHON_ARGS+=("--fsdp_dit")
[[ "${FSPD_TEXT_ENCODER}" == "True" ]] && PYTHON_ARGS+=("--fsdp_text_encoder")
[[ "${COMPILE_DIT}" == "True" ]] && PYTHON_ARGS+=("--compile_dit")

# 模型设置
PYTHON_ARGS+=("--model_name" "${MODEL_NAME}")
PYTHON_ARGS+=("--sampler_name" "${SAMPLER_NAME}")

# LiLoRA 和 HyperNetwork
[[ -n "${LILORA_PATH}" && "${LILORA_PATH}" != "None" ]] && PYTHON_ARGS+=("--lilora_path" "${LILORA_PATH}")
[[ -n "${HYPERNETWORK_PATH}" && "${HYPERNETWORK_PATH}" != "None" ]] && PYTHON_ARGS+=("--hypernetwork_path" "${HYPERNETWORK_PATH}")

# LiLoRA 参数
PYTHON_ARGS+=("--lora_rank" "${LORA_RANK}")
PYTHON_ARGS+=("--lora_down_dim" "${LORA_DOWN_DIM}")
PYTHON_ARGS+=("--lora_up_dim" "${LORA_UP_DIM}")
[[ -n "${LORA_WEIGHT}" && "${LORA_WEIGHT}" != "None" ]] && PYTHON_ARGS+=("--lora_weight" "${LORA_WEIGHT}")

# 视频参数
PYTHON_ARGS+=("--video_length" "${VIDEO_LENGTH}")
PYTHON_ARGS+=("--fps" "${FPS}")
[[ -n "${PARTIAL_VIDEO_LENGTH}" && "${PARTIAL_VIDEO_LENGTH}" != "None" ]] && PYTHON_ARGS+=("--partial_video_length" "${PARTIAL_VIDEO_LENGTH}")
PYTHON_ARGS+=("--overlap_video_length" "${OVERLAP_VIDEO_LENGTH}")

# 权重类型
PYTHON_ARGS+=("--weight_dtype" "${WEIGHT_DTYPE}")

# 推理参数 (单样本时使用，批量时从 JSON 读取)
if [[ -z "${TEST_JSON_PATH}" || "${TEST_JSON_PATH}" == "None" ]]; then
    [[ -n "${FIRST_FRAME_PATH}" && "${FIRST_FRAME_PATH}" != "None" ]] && PYTHON_ARGS+=("--first_frame_path" "${FIRST_FRAME_PATH}")
    [[ -n "${REF_VIDEO_PATH}" && "${REF_VIDEO_PATH}" != "None" ]] && PYTHON_ARGS+=("--ref_video_path" "${REF_VIDEO_PATH}")
    [[ -n "${PROMPT}" && "${PROMPT}" != "None" ]] && PYTHON_ARGS+=("--prompt" "${PROMPT}")
    PYTHON_ARGS+=("--negative_prompt" "${NEGATIVE_PROMPT}")
    PYTHON_ARGS+=("--guidance_scale" "${GUIDANCE_SCALE}")
    PYTHON_ARGS+=("--seed" "${SEED}")
    PYTHON_ARGS+=("--num_inference_steps" "${NUM_INFERENCE_STEPS}")
else
    PYTHON_ARGS+=("--test_json_path" "${TEST_JSON_PATH}")
fi

# 输出路径
PYTHON_ARGS+=("--save_path" "${SAVE_PATH}")

# =============================================================================
# 运行推理脚本 (Run Inference Script)
# =============================================================================
echo "========================================"
echo "CogVideoX-Fun I2V Inference"
echo "========================================"
echo "Model: ${MODEL_NAME}"
echo "LiLoRA: ${LILORA_PATH}"
echo "HyperNetwork: ${HYPERNETWORK_PATH}"
echo "Video Length: ${VIDEO_LENGTH}"
if [[ -n "${TEST_JSON_PATH}" && "${TEST_JSON_PATH}" != "None" ]]; then
    echo "Test Set: ${TEST_JSON_PATH}"
else
    echo "First Frame: ${FIRST_FRAME_PATH}"
    echo "Prompt: ${PROMPT}"
fi
echo "========================================"

torchrun --nproc-per-node=1 --master_port=29501\
    /home/xhm001/Desktop/Code/VideoX-Fun-main/examples/cogvideox_fun/predict_i2v.py \
    "${PYTHON_ARGS[@]}"


echo ""
echo "Inference completed!"
