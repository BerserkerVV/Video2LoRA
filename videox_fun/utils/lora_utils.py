# LoRA network module
# reference:
# https://github.com/microsoft/LoRA/blob/main/loralib/layers.py
# https://github.com/cloneofsimo/lora/blob/master/lora_diffusion/lora.py
# https://github.com/bmaltais/kohya_ss

import hashlib
import math
import os
from collections import defaultdict
from io import BytesIO
from typing import List, Optional, Type, Union

import safetensors.torch
import torch
import torch.utils.checkpoint
from diffusers.models.lora import LoRACompatibleConv, LoRACompatibleLinear
from safetensors.torch import load_file
from transformers import T5EncoderModel
import torch.nn.functional as F


class LoRAModule(torch.nn.Module):
    """
    replaces forward method of the original Linear, instead of replacing the original Linear module.
    """

    def __init__(
        self,
        lora_name,
        org_module: torch.nn.Module,
        multiplier=1.0,
        rank=1,
        alpha=1,
        down_dim: int = 200,
        up_dim: int = 100,
        dropout=None,
        rank_dropout=None,
        module_dropout=None,
        is_train: bool = False,
    ):
        """if alpha == 0 or None, alpha is rank (no scaling)."""
        super().__init__()
        self.lora_name = lora_name

        if org_module.__class__.__name__ == "Conv2d":
            in_features = org_module.in_channels
            out_features = org_module.out_channels
        else:
            in_features = org_module.in_features
            out_features = org_module.out_features

        self.rank = rank

            
        down_aux = torch.empty(down_dim, in_features)
        up_aux = torch.empty(out_features, up_dim)
        torch.nn.init.orthogonal_(down_aux, gain=1)
        torch.nn.init.orthogonal_(up_aux, gain=1)
        
        # learnable parameters
        self.down_aux = torch.nn.Parameter(down_aux)
        self.up_aux = torch.nn.Parameter(up_aux) 
        self.in_features = in_features
        self.out_features = out_features
        self.down_dim = down_dim
        self.up_dim = up_dim
        self.down = self.up = None
        self.network_alpha = alpha
        self.rank = rank
        self.is_train = is_train
        self.org_module = org_module
        self.multiplier = multiplier  # 添加 multiplier 属性
        
        if is_train:
            # weight initialization
            down_weight = torch.empty(rank, down_dim)
            up_weight = torch.empty(up_dim, rank)
            torch.nn.init.xavier_normal_(down_weight)
            torch.nn.init.zeros_(up_weight)
            weight = torch.concat([torch.flatten(down_weight), torch.flatten(up_weight)])
            self.weight_embedding = torch.nn.Parameter(weight)
        
    def update_weight(self, weight_embedding):
        """
           Update the weights of the LoRa model. Only called outside.
           This function is used to replace the weights of the LoRA model with pre-optimized weights or weights predicted
           by a hypernetwork.
           """
        # Check if the shape is [batch_size, (up_dim+down_dim)*r] or [(up_dim+down_dim)*r]
        expected_dim = (self.up_dim + self.down_dim) * self.rank

        if len(weight_embedding.shape) > 2 or (
                len(weight_embedding.shape) == 2 and weight_embedding.shape[1] != expected_dim) or (
                len(weight_embedding.shape) == 1 and weight_embedding.shape != (expected_dim,)):
            raise ValueError(
                "The shape of weight_embedding must be [batch_size, (up_dim+down_dim)*r] or [(up_dim+down_dim)*r], "
                "and the dimension of weight_embedding must not be more than 2.")
        self._parameters.pop("weight_embedding", None)
        self.weight_embedding = weight_embedding
        
    def update_aux(self, down_aux=None, up_aux=None):
        """
        update up_aux and down_aux based on aux_seed, using orthonogal initialization
        """
        self.down_aux.data = down_aux
        self.up_aux.data = up_aux
              


    def apply_to(self):
        self.org_forward = self.org_module.forward
        self.org_module.forward = self.forward
        del self.org_module

    def forward(self, hidden_states):
        orig_dtype = hidden_states.dtype
        # 确保计算在 embedding 的精度下进行 (通常是 float32)
        dtype = self.weight_embedding.dtype

        # 1. 获取原始层的输出 (这一步保证了不丢失预训练模型的信息)
        # 注意：必须调用 org_forward 而不是 self.org_module()
        org_out = self.org_forward(hidden_states)

        # 2. 计算 LiLoRA 的增量部分
        down_aux = self.down_aux.to(self.weight_embedding.device)
        up_aux = self.up_aux.to(self.weight_embedding.device)
        
        # 分割与 Reshape (与 LoRALinearLayer 逻辑一致)
        down_weight, up_weight = self.weight_embedding.split([self.down_dim * self.rank, self.up_dim * self.rank], dim=-1)
        
        if self.weight_embedding.dim() == 1:
            down_weight = down_weight.reshape(self.rank, -1)
            up_weight = up_weight.reshape(-1, self.rank)
            
            # 重建矩阵
            down = down_weight @ down_aux
            up = up_aux @ up_weight
            
            # 计算增量
            delta_out = F.linear(hidden_states.to(dtype), down)
            delta_out = F.linear(delta_out, up)
        else:
            # Batch 模式 (用于 HyperNetwork)
            down_weight = down_weight.reshape(self.weight_embedding.size(0), self.rank, -1)
            up_weight = up_weight.reshape(self.weight_embedding.size(0), -1, self.rank)
            
            down = down_weight @ down_aux
            up = up_aux @ up_weight
            
            delta_out = torch.einsum('b r i, b ... i -> b ... r', down, hidden_states.to(dtype))
            delta_out = torch.einsum('b o r, b ... r -> b ... o', up, delta_out)

        # 3. 缩放与合并
        if self.network_alpha is not None:
            delta_out *= self.network_alpha / self.rank

        # 最终结果 = 原始输出 + (增量 * 乘数)
        return org_out + (delta_out * self.multiplier).to(orig_dtype)
def addnet_hash_legacy(b):
    """Old model hash used by sd-webui-additional-networks for .safetensors format files"""
    m = hashlib.sha256()

    b.seek(0x100000)
    m.update(b.read(0x10000))
    return m.hexdigest()[0:8]


def addnet_hash_safetensors(b):
    """New model hash used by sd-webui-additional-networks for .safetensors format files"""
    hash_sha256 = hashlib.sha256()
    blksize = 1024 * 1024

    b.seek(0)
    header = b.read(8)
    n = int.from_bytes(header, "little")

    offset = n + 8
    b.seek(offset)
    for chunk in iter(lambda: b.read(blksize), b""):
        hash_sha256.update(chunk)

    return hash_sha256.hexdigest()


def precalculate_safetensors_hashes(tensors, metadata):
    """Precalculate the model hashes needed by sd-webui-additional-networks to
    save time on indexing the model later."""

    # Because writing user metadata to the file can change the result of
    # sd_models.model_hash(), only retain the training metadata for purposes of
    # calculating the hash, as they are meant to be immutable
    metadata = {k: v for k, v in metadata.items() if k.startswith("ss_")}

    bytes = safetensors.torch.save(tensors, metadata)
    b = BytesIO(bytes)

    model_hash = addnet_hash_safetensors(b)
    legacy_hash = addnet_hash_legacy(b)
    return model_hash, legacy_hash


class LoRANetwork(torch.nn.Module):
    # 目标模块保持不变
    TRANSFORMER_TARGET_REPLACE_MODULE = [
        "CogVideoXTransformer3DModel", "WanTransformer3DModel", 
        "Wan2_2Transformer3DModel", "FluxTransformer2DModel", "QwenImageTransformer2DModel", 
        "Wan2_2Transformer3DModel_Animate", "Wan2_2Transformer3DModel_S2V", "FantasyTalkingTransformer3DModel",
    ]
    TEXT_ENCODER_TARGET_REPLACE_MODULE = ["T5LayerSelfAttention", "T5LayerFF", "BertEncoder", "T5SelfAttention", "T5CrossAttention"]
    
    LORA_PREFIX_TRANSFORMER = "lora_unet"
    LORA_PREFIX_TEXT_ENCODER = "lora_te"

    def __init__(
        self,
        text_encoder: Union[List[torch.nn.Module], torch.nn.Module],
        unet,
        multiplier: float = 1.0,
        rank: int = 1,
        alpha: float = 1,
        down_dim: int = 200,   # LiLoRA 辅助维度
        up_dim: int = 100,     # LiLoRA 辅助维度
        dropout: Optional[float] = None,
        module_class: Type[object] = LoRAModule,
        is_train: bool = False,
        skip_name: str = None,
        target_name: str = None,
        **kwargs
    ) -> None:
        super().__init__()
        self.multiplier = multiplier
        self.rank = rank
        self.alpha = alpha
        self.down_dim = down_dim
        self.up_dim = up_dim
        self.is_train = is_train

        print(f"Creating LiLoRA network. Rank: {rank}, Alpha: {alpha}, Aux Dims: {down_dim}/{up_dim}")

        def create_modules(
            is_unet: bool,
            root_module: torch.nn.Module,
            target_replace_modules: List[str],
        ) -> List[LoRAModule]:
            prefix = self.LORA_PREFIX_TRANSFORMER if is_unet else self.LORA_PREFIX_TEXT_ENCODER
            loras = []
            
            for name, module in root_module.named_modules():
                # 仅处理指定的 Transformer Block 模块
                if module.__class__.__name__ in target_replace_modules:
                    for child_name, child_module in module.named_modules():
                        # 只筛选 Linear 层
                        if child_module.__class__.__name__ not in ["Linear", "LoRACompatibleLinear"]:
                            continue
                        
                        # 过滤逻辑 (skip/target)
                        if skip_name and skip_name in child_name: continue
                        if target_name:
                            t_list = [target_name] if isinstance(target_name, str) else target_name
                            if not any(t in child_name for t in t_list): continue

                        # 生成 LoRA 命名的标准格式
                        lora_name = f"{prefix}_{name}_{child_name}".replace(".", "_")

                        lora = module_class(
                            lora_name=lora_name,
                            org_module=child_module,
                            multiplier=self.multiplier,
                            rank=self.rank,
                            alpha=self.alpha,
                            down_dim=self.down_dim,
                            up_dim=self.up_dim,
                            is_train=self.is_train,
                            dropout=dropout,
                        )
                        loras.append(lora)
            return loras

        # 1. 处理 Text Encoder
        text_encoders = text_encoder if isinstance(text_encoder, list) else [text_encoder]
        self.text_encoder_loras = []
        for te in text_encoders:
            if te is not None:
                self.text_encoder_loras.extend(create_modules(False, te, self.TEXT_ENCODER_TARGET_REPLACE_MODULE))

        # 2. 处理 Transformer (U-Net)
        self.unet_loras = create_modules(True, unet, self.TRANSFORMER_TARGET_REPLACE_MODULE)

        print(f"Total modules created: TE={len(self.text_encoder_loras)}, UNet={len(self.unet_loras)}")

        # 注册所有 LoRA 模块到当前 Network
        for lora in self.text_encoder_loras + self.unet_loras:
            self.add_module(lora.lora_name, lora)

    def apply_to(self, text_encoder, unet, apply_text_encoder=True, apply_unet=True):
        if apply_text_encoder:
            print("enable LoRA for text encoder")
        else:
            for lora in self.text_encoder_loras:
                if hasattr(self, lora.lora_name):
                    delattr(self, lora.lora_name)
            self.text_encoder_loras = []

        if apply_unet:
            print("enable LoRA for U-Net")
        else:
            self.unet_loras = []

        for lora in self.text_encoder_loras + self.unet_loras:
            lora.apply_to()
            self.add_module(lora.lora_name, lora)

    def set_multiplier(self, multiplier):
        self.multiplier = multiplier
        for lora in self.text_encoder_loras + self.unet_loras:
            lora.multiplier = self.multiplier

    def load_weights(self, file):
        if os.path.splitext(file)[1] == ".safetensors":
            from safetensors.torch import load_file

            weights_sd = load_file(file)
        else:
            weights_sd = torch.load(file, map_location="cpu")
        info = self.load_state_dict(weights_sd, False)
        return info

    def prepare_optimizer_params(self, text_encoder_lr, unet_lr, default_lr):
        self.requires_grad_(True)
        all_params = []

        def enumerate_params(loras):
            params = []
            for lora in loras:
                params.extend(lora.parameters())
            return params

        if self.text_encoder_loras:
            param_data = {"params": enumerate_params(self.text_encoder_loras)}
            if text_encoder_lr is not None:
                param_data["lr"] = text_encoder_lr
            all_params.append(param_data)

        if self.unet_loras:
            param_data = {"params": enumerate_params(self.unet_loras)}
            if unet_lr is not None:
                param_data["lr"] = unet_lr
            all_params.append(param_data)

        return all_params

    def enable_gradient_checkpointing(self):
        pass

    def get_trainable_params(self):
        return self.parameters()

    def save_weights(self, file, dtype, metadata):
        if metadata is not None and len(metadata) == 0:
            metadata = None

        # 1. 获取原始 state_dict
        original_sd = self.state_dict()
        state_dict = {}

        # 2. 过滤掉不需要保存的参数
        # 在 LiLoRA 中，只需要保存 down_aux 和 up_aux
        # weight_embedding 是 hypernetwork 生成的，不保存
        # org_module.weight 是原始模块的权重，不应该保存（避免文件过大）
        for key, value in original_sd.items():
            if "weight_embedding" in key:
                continue  # 跳过 hypernetwork 生成的权重
            if "org_module.weight" in key:
                continue  # 跳过原始模块权重（避免文件过大）
            
            # 转换数据类型并移至 CPU
            v = value.detach().clone().to("cpu")
            if dtype is not None:
                v = v.to(dtype)
            state_dict[key] = v

        # 3. 保存逻辑
        if os.path.splitext(file)[1] == ".safetensors":
            from safetensors.torch import save_file

            if metadata is None:
                metadata = {}
            
            # 计算哈希值（针对过滤后的 state_dict）
            model_hash, legacy_hash = precalculate_safetensors_hashes(state_dict, metadata)
            metadata["sshs_model_hash"] = model_hash
            metadata["sshs_legacy_hash"] = legacy_hash

            save_file(state_dict, file, metadata)
        else:
            torch.save(state_dict, file)
            
        print(f"Weights saved to {file}. (Only down_aux/up_aux saved, org_module.weight excluded)")

def create_network(
    multiplier: float,
    network_dim: Optional[int],
    network_alpha: Optional[float],
    text_encoder: Union[T5EncoderModel, List[T5EncoderModel]],
    transformer,
    neuron_dropout: Optional[float] = None,
    skip_name: str = None,
    target_name = None,
    **kwargs,
):
    if network_dim is None:
        network_dim = 1  # 这里的 network_dim 对应 LoRA 的 rank
    if network_alpha is None:
        network_alpha = 1.0

    # 从 kwargs 中提取 LiLoRA 特有的维度，并设置默认值
    # 这样可以确保在使用 HyperNetwork 时，投影维度是可控的
    down_dim = kwargs.get("down_dim", 200) 
    up_dim = kwargs.get("up_dim", 100)
    is_train = kwargs.get("is_train", False)

    network = LoRANetwork(
        text_encoder=text_encoder,
        unet=transformer,
        multiplier=multiplier,
        rank=network_dim,
        alpha=network_alpha,
        down_dim=down_dim,    # 传递给 LoRANetwork
        up_dim=up_dim,        # 传递给 LoRANetwork
        dropout=neuron_dropout,
        is_train=is_train,    # 传递训练状态
        skip_name=skip_name,
        target_name=target_name,
        # verbose=True, # 修正了单词拼写错误
    )
    return network

def merge_lora(pipeline, lora_path, multiplier, device='cpu', dtype=torch.float32, state_dict=None, transformer_only=False, sub_transformer_name="transformer"):
    LORA_PREFIX_TRANSFORMER = "lora_unet"
    LORA_PREFIX_TEXT_ENCODER = "lora_te"
    if state_dict is None:
        state_dict = load_file(lora_path)
    else:
        state_dict = state_dict
    updates = defaultdict(dict)
    for key, value in state_dict.items():
        if "diffusion_model" in key:
            key = key.replace("diffusion_model.", "lora_unet__")
            key = key.replace("blocks.", "blocks_")
            key = key.replace(".self_attn.", "_self_attn_")
            key = key.replace(".cross_attn.", "_cross_attn_")
            key = key.replace(".ffn.", "_ffn_")
        if "lora_A" in key or "lora_B" in key:
            key = "lora_unet__" + key
            key = key.replace("blocks.", "blocks_")
            key = key.replace(".self_attn.", "_self_attn_")
            key = key.replace(".cross_attn.", "_cross_attn_")
            key = key.replace(".ffn.", "_ffn_")
            key = key.replace(".lora_A.default.", ".lora_down.")
            key = key.replace(".lora_B.default.", ".lora_up.")
        layer, elem = key.split('.', 1)
        updates[layer][elem] = value

    sequential_cpu_offload_flag = False
    if pipeline.transformer.device == torch.device(type="meta"):
        pipeline.remove_all_hooks()
        sequential_cpu_offload_flag = True
        offload_device = pipeline._offload_device

    for layer, elems in updates.items():

        if "lora_te" in layer:
            if transformer_only:
                continue
            else:
                layer_infos = layer.split(LORA_PREFIX_TEXT_ENCODER + "_")[-1].split("_")
                curr_layer = pipeline.text_encoder
        else:
            layer_infos = layer.split(LORA_PREFIX_TRANSFORMER + "_")[-1].split("_")
            curr_layer = getattr(pipeline, sub_transformer_name)

        try:
            curr_layer = curr_layer.__getattr__("_".join(layer_infos[1:]))
        except Exception:
            temp_name = layer_infos.pop(0)
            try:
                while len(layer_infos) > -1:
                    try:
                        curr_layer = curr_layer.__getattr__(temp_name + "_" + "_".join(layer_infos))
                        break
                    except Exception:
                        try:
                            curr_layer = curr_layer.__getattr__(temp_name)
                            if len(layer_infos) > 0:
                                temp_name = layer_infos.pop(0)
                            elif len(layer_infos) == 0:
                                break
                        except Exception:
                            if len(layer_infos) == 0:
                                print(f'Error loading layer in front search: {layer}. Try it in back search.')
                            if len(temp_name) > 0:
                                temp_name += "_" + layer_infos.pop(0)
                            else:
                                temp_name = layer_infos.pop(0)
            except Exception:
                if "lora_te" in layer:
                    if transformer_only:
                        continue
                    else:
                        layer_infos = layer.split(LORA_PREFIX_TEXT_ENCODER + "_")[-1].split("_")
                        curr_layer = pipeline.text_encoder
                else:
                    layer_infos = layer.split(LORA_PREFIX_TRANSFORMER + "_")[-1].split("_")
                    curr_layer = getattr(pipeline, sub_transformer_name)

                len_layer_infos = len(layer_infos)
                start_index     = 0 if len_layer_infos >= 1 and len(layer_infos[0]) > 0 else 1
                end_indx        = len_layer_infos

                error_flag      = False if len_layer_infos >= 1 else True
                while start_index < len_layer_infos:
                    try:
                        if start_index >= end_indx:
                            print(f'Error loading layer in back search: {layer}')
                            error_flag = True
                            break
                        curr_layer = curr_layer.__getattr__("_".join(layer_infos[start_index:end_indx]))
                        start_index = end_indx
                        end_indx = len_layer_infos
                    except Exception:
                        end_indx -= 1
                if error_flag:
                    continue

        origin_dtype = curr_layer.weight.data.dtype
        origin_device = curr_layer.weight.data.device

        curr_layer = curr_layer.to(device, dtype)
        weight_up = elems['lora_up.weight'].to(device, dtype)
        weight_down = elems['lora_down.weight'].to(device, dtype)
        
        if 'alpha' in elems.keys():
            alpha = elems['alpha'].item() / weight_up.shape[1]
        else:
            alpha = 1.0

        if len(weight_up.shape) == 4:
            curr_layer.weight.data += multiplier * alpha * torch.mm(
                weight_up.squeeze(3).squeeze(2), weight_down.squeeze(3).squeeze(2)
            ).unsqueeze(2).unsqueeze(3)
        else:
            curr_layer.weight.data += multiplier * alpha * torch.mm(weight_up, weight_down)
        curr_layer = curr_layer.to(origin_device, origin_dtype)

    if sequential_cpu_offload_flag:
        pipeline.enable_sequential_cpu_offload(device=offload_device)
    return pipeline

# TODO: Refactor with merge_lora.
def unmerge_lora(pipeline, lora_path, multiplier=1, device="cpu", dtype=torch.float32, sub_transformer_name="transformer"):
    """Unmerge state_dict in LoRANetwork from the pipeline in diffusers."""
    LORA_PREFIX_UNET = "lora_unet"
    LORA_PREFIX_TEXT_ENCODER = "lora_te"
    state_dict = load_file(lora_path)

    updates = defaultdict(dict)
    for key, value in state_dict.items():
        if "diffusion_model" in key:
            key = key.replace("diffusion_model.", "lora_unet__")
            key = key.replace("blocks.", "blocks_")
            key = key.replace(".self_attn.", "_self_attn_")
            key = key.replace(".cross_attn.", "_cross_attn_")
            key = key.replace(".ffn.", "_ffn_")
        if "lora_A" in key or "lora_B" in key:
            key = "lora_unet__" + key
            key = key.replace("blocks.", "blocks_")
            key = key.replace(".self_attn.", "_self_attn_")
            key = key.replace(".cross_attn.", "_cross_attn_")
            key = key.replace(".ffn.", "_ffn_")
            key = key.replace(".lora_A.default.", ".lora_down.")
            key = key.replace(".lora_B.default.", ".lora_up.")
        layer, elem = key.split('.', 1)
        updates[layer][elem] = value

    sequential_cpu_offload_flag = False
    if pipeline.transformer.device == torch.device(type="meta"):
        pipeline.remove_all_hooks()
        sequential_cpu_offload_flag = True

    for layer, elems in updates.items():

        if "lora_te" in layer:
            layer_infos = layer.split(LORA_PREFIX_TEXT_ENCODER + "_")[-1].split("_")
            curr_layer = pipeline.text_encoder
        else:
            layer_infos = layer.split(LORA_PREFIX_UNET + "_")[-1].split("_")
            curr_layer = getattr(pipeline, sub_transformer_name)

        try:
            curr_layer = curr_layer.__getattr__("_".join(layer_infos[1:]))
        except Exception:
            temp_name = layer_infos.pop(0)
            try:
                while len(layer_infos) > -1:
                    try:
                        curr_layer = curr_layer.__getattr__(temp_name + "_" + "_".join(layer_infos))
                        break
                    except Exception:
                        try:
                            curr_layer = curr_layer.__getattr__(temp_name)
                            if len(layer_infos) > 0:
                                temp_name = layer_infos.pop(0)
                            elif len(layer_infos) == 0:
                                break
                        except Exception:
                            if len(layer_infos) == 0:
                                print(f'Error loading layer in front search: {layer}. Try it in back search.')
                            if len(temp_name) > 0:
                                temp_name += "_" + layer_infos.pop(0)
                            else:
                                temp_name = layer_infos.pop(0)
            except Exception:
                if "lora_te" in layer:
                    layer_infos = layer.split(LORA_PREFIX_TEXT_ENCODER + "_")[-1].split("_")
                    curr_layer = pipeline.text_encoder
                else:
                    layer_infos = layer.split(LORA_PREFIX_UNET + "_")[-1].split("_")
                    curr_layer = getattr(pipeline, sub_transformer_name)
                len_layer_infos = len(layer_infos)

                start_index     = 0 if len_layer_infos >= 1 and len(layer_infos[0]) > 0 else 1
                end_indx        = len_layer_infos

                error_flag      = False if len_layer_infos >= 1 else True
                while start_index < len_layer_infos:
                    try:
                        if start_index >= end_indx:
                            print(f'Error loading layer in back search: {layer}')
                            error_flag = True
                            break
                        curr_layer = curr_layer.__getattr__("_".join(layer_infos[start_index:end_indx]))
                        start_index = end_indx
                        end_indx = len_layer_infos
                    except Exception:
                        end_indx -= 1
                if error_flag:
                    continue

        origin_dtype = curr_layer.weight.data.dtype
        origin_device = curr_layer.weight.data.device

        curr_layer = curr_layer.to(device, dtype)
        weight_up = elems['lora_up.weight'].to(device, dtype)
        weight_down = elems['lora_down.weight'].to(device, dtype)
        
        if 'alpha' in elems.keys():
            alpha = elems['alpha'].item() / weight_up.shape[1]
        else:
            alpha = 1.0

        if len(weight_up.shape) == 4:
            curr_layer.weight.data -= multiplier * alpha * torch.mm(
                weight_up.squeeze(3).squeeze(2), weight_down.squeeze(3).squeeze(2)
            ).unsqueeze(2).unsqueeze(3)
        else:
            curr_layer.weight.data -= multiplier * alpha * torch.mm(weight_up, weight_down)
        curr_layer = curr_layer.to(origin_device, origin_dtype)

    if sequential_cpu_offload_flag:
        pipeline.enable_sequential_cpu_offload(device=device)
    return pipeline
