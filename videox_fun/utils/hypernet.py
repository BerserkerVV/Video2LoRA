import os
from typing import *
import numpy as np
import torch
import platform

import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint

from torchvision.transforms.functional import resize
from .attention import TransformerBlock
from timm import create_model
# model download: https://github.com/huggingface/pytorch-image-models

# from einops import rearrange

from .lora_utils import LoRAModule


def _get_sinusoid_encoding_table(n_position, d_hid):
    ''' Sinusoid position encoding table '''

    def get_position_angle_vec(position):
        # this part calculate the position In brackets
        return [position / np.power(10000, 2 * (hid_j // 2) / d_hid) for hid_j in range(d_hid)]

    sinusoid_table = np.array([get_position_angle_vec(pos_i) for pos_i in range(n_position)])
    # [:, 0::2] are all even subscripts, is dim_2i
    sinusoid_table[:, 0::2] = np.sin(sinusoid_table[:, 0::2])  # dim 2i
    sinusoid_table[:, 1::2] = np.cos(sinusoid_table[:, 1::2])  # dim 2i+1

    return torch.FloatTensor(sinusoid_table).unsqueeze(0)


class WeightDecoder(nn.Module):
    def __init__(
            self,
            weight_dim: int = 150,
            weight_num: int = 168,
            decoder_blocks: int = 4,
            add_constant: bool = False,
    ):
        super(WeightDecoder, self).__init__()
        self.weight_num = weight_num
        self.weight_dim = weight_dim

        self.register_buffer(
            'block_pos_emb',
            _get_sinusoid_encoding_table(weight_num * 2, weight_dim)
        )

        # calc heads for mem-eff or flash_attn
        heads = 1
        while weight_dim % heads == 0 and weight_dim // heads > 64:
            heads *= 2
        heads //= 2

        self.pos_emb_proj = nn.Linear(weight_dim, weight_dim, bias=False)
        self.decoder_model = nn.ModuleList(
            TransformerBlock(weight_dim, heads, weight_dim // heads, context_dim=weight_dim, gated_ff=False)
            for _ in range(decoder_blocks)
        )
        # self.delta_proj = nn.Linear(weight_dim, weight_dim, bias=False)
        self.delta_proj = nn.Sequential(
            nn.LayerNorm(weight_dim),
            nn.Linear(weight_dim, weight_dim, bias=False)
        )
        self.init_weights(add_constant)

    def init_weights(self, add_constant: bool = False):
        def basic_init(module):
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)

        self.apply(basic_init)

        # For no pre-optimized training, you should consider use the following init
        # with self.down = down@down_aux + 1 in LiLoRAAttnProcessor
        # if add_constant:
        # torch.nn.init.constant_(self.delta_proj[1].weight, 0)

        # advice from Nataniel Ruiz, looks like 1e-3 is small enough
        # else:
        #     torch.nn.init.normal_(self.delta_proj[1].weight, std=1e-3)
        torch.nn.init.normal_(self.delta_proj[1].weight, std=1e-3)

    def forward(self, weight, features):
        pos_emb = self.pos_emb_proj(self.block_pos_emb[:, :weight.size(1)].clone().detach())
        h = weight + pos_emb
        for decoder in self.decoder_model:
            h = decoder(h, context=features)
        weight = weight + self.delta_proj(h)
        return weight


class VideoWeightGenerator(nn.Module):
    def __init__(
        self,
        video_feat_dim: int,
        weight_dim: int = 240,
        weight_num: int = 176,
        decoder_blocks: int = 16,
        sample_iters: int = 1,
        add_constant: bool = False,
    ):
        super().__init__()
        self.weight_num = weight_num
        self.weight_dim = weight_dim
        self.sample_iters = sample_iters

        self.register_buffer(
            'block_pos_emb',
            _get_sinusoid_encoding_table(weight_num * 2, weight_dim)
        )

        self.feature_proj = nn.Linear(video_feat_dim, weight_dim, bias=False)
        self.decoder_model = WeightDecoder(weight_dim, weight_num, decoder_blocks, add_constant)

    def encode_features(self, video_features):
        assert video_features.ndim == 5, f"Expected (B, C, T, H, W), got {video_features.shape}"
        B, C, T, H, W = video_features.shape
        return video_features.view(B, C, -1).transpose(1, 2)  # (B, L, C)

    def decode_weight(self, features, iters=None, weight=None):
        # 确保feature_proj层在正确的设备上
        self.feature_proj = self.feature_proj.to(features.device)
        features = self.feature_proj(features)
        if weight is None:
            weight = torch.zeros(
                features.size(0), self.weight_num, self.weight_dim,
                device=features.device, dtype=features.dtype 
            )
        for i in range(iters or self.sample_iters):
            weight = self.decoder_model(weight, features)
        return weight

    def forward(self, video_features, iters=None, weight=None, ensure_grad=0):
        features = self.encode_features(video_features) + ensure_grad
        # 确保所有层都在输入张量的设备上
        self.to(video_features.device)
        return self.decode_weight(features, iters, weight)

class VideoHyperDream(nn.Module):
    def __init__(
        self,
        video_feat_dim: int,          # e.g., 4 for SD VAE latent
        weight_dim: int = 240,
        weight_num: int = 176,
        decoder_blocks: int = 4,
        sample_iters: int = 4,
        add_constant: bool = False,
    ):
        super(VideoHyperDream, self).__init__()
        self.img_weight_generator = VideoWeightGenerator(
            video_feat_dim=video_feat_dim,
            weight_dim=weight_dim,
            weight_num=weight_num,
            decoder_blocks=decoder_blocks,
            sample_iters=sample_iters,
            add_constant=add_constant,
        )
        self.weight_dim = weight_dim
        self.add_constant = add_constant
        self.liloras: Dict[str, LoRAModule] = {}
        self.liloras_keys: List[str] = []
        self.gradient_checkpointing = False

    def enable_gradient_checkpointing(self):
        self.gradient_checkpointing = True

    def train_params(self):
        return [p for p in self.parameters() if p.requires_grad]

    def set_lilora(self, liloras):
        self.liloras = liloras
        if isinstance(liloras, dict):
            self.liloras_keys = list(liloras.keys())
        else:
            self.liloras_keys = list(range(len(liloras)))
        length = len(self.liloras_keys)
        print(f"Total LiLoRAs: {length}, Hypernet params for each video: {length * self.weight_dim}")

    def gen_weight(self, video_features, iters, weight, ensure_grad=0):
        weights = self.img_weight_generator(video_features, iters, weight, ensure_grad)
        weight_list = weights.split(1, dim=1)  # [B, N, D] -> N × [B, 1, D]
        return weights, [w.squeeze(1) for w in weight_list]
    
    def set_device(self, device):
        self.device = device

    def forward(self, video_features: torch.Tensor, iters: int = None, weight: torch.Tensor = None):
        if self.training and self.gradient_checkpointing:
            ensure_grad = torch.zeros(1, device=video_features.device).requires_grad_(True)
            weights, weight_list = checkpoint.checkpoint(
                self.gen_weight, video_features, iters, weight, ensure_grad
            )
        else:
            weights, weight_list = self.gen_weight(video_features, iters, weight)

        # Optional: update LiLoRA weights (currently commented out)
        # for key, weight in zip(self.liloras_keys, weight_list):
        #     self.liloras[key].update_weight(weight, self.add_constant)

        return weights, weight_list


class PreOptHyperDream(nn.Module):
    def __init__(
        self,
        rank: int = 1,
        down_dim: int = 100,
        up_dim: int = 50,
    ):
        super(PreOptHyperDream, self).__init__()
        # 初始占位（后续在 set_lilora 中替换）
        self.weights = nn.Parameter(torch.tensor(0.0))
        self.rank = rank
        self.down_dim = down_dim
        self.up_dim = up_dim
        # 每个 lora 的参数维度（down + up）乘以 rank，用于报告与计算
        self.params_per_lora = (down_dim + up_dim) * rank

        # 外部传入的 liloras（字典或列表），以及其 key 顺序
        self.liloras: Dict[str, LoRAModule] = {}
        self.liloras_keys: List[str] = []

        # checkpointing 与 device
        self.gradient_checkpointing = False
        self.device = 'cpu'

        # class 映射（如果使用 class 模式）
        self.class_to_idx: Dict[str, int] = {}
        self.idx_to_class: Dict[int, str] = {}
        self.classes_list: List[str] = []
        self.num_classes = 0

        # length 表示每组 weights 中包含多少个 lora（即 len(self.liloras_keys)）
        self.length = 0

    def enable_gradient_checkpointing(self):
        self.gradient_checkpointing = True

    def train_params(self):
        return [p for p in self.parameters() if p.requires_grad]

    def set_device(self, device):
        self.device = device

    def set_lilora(self, liloras: Union[Dict[str, LoRAModule], List[LoRAModule]],
                   classes_list: List[str] = None):
        """
        设置 liloras，并预分配权重。
        - liloras: dict 或 list，表示所有需要被 hypernetwork 控制的 LoRA 元素
        - classes_list: 若提供（字符串列表），则按每个唯一 class 预分配权重，并建立 class->idx 映射
        """
        self.liloras = liloras
        if isinstance(liloras, dict):
            self.liloras_keys = list(liloras.keys())  # 固定顺序
        elif isinstance(liloras, list):
            self.liloras_keys = list(range(len(liloras)))
        else:
            raise TypeError("liloras only support dict and list!")

        # 每个 lora 条目的数量
        length = len(self.liloras_keys)
        self.length = length

        # 计算每个权重向量的最后一维大小（down + up）* rank
        dim_full = (self.down_dim * self.rank) + (self.up_dim * self.rank)

        # 如果提供 classes_list，则为每个 class 预分配权重并建立映射
        if classes_list is not None:
            # 去重并排序以保证顺序稳定（你可以按需改为不排序）
            self.classes_list = sorted(list(dict.fromkeys(classes_list)))
            self.num_classes = len(self.classes_list)
            self.class_to_idx = {cls: idx for idx, cls in enumerate(self.classes_list)}
            self.idx_to_class = {idx: cls for idx, cls in enumerate(self.classes_list)}

            print(f"Total LiLoRAs: {length}, Total Classes: {self.num_classes}")
            print(f"Pre-Optimized params for each class: {length * self.params_per_lora}")
            print(f"Pre-Optimized params: {length * self.params_per_lora * self.num_classes / 1e6:.1f}M")

            # 删除旧占位
            try:
                del self.weights
            except Exception:
                pass

            # 为每个 class 创建一个参数块，形状 (1, length, dim_full)
            self.weights = nn.ParameterList([
                nn.Parameter(
                    torch.cat([
                        torch.randn(1, length, self.down_dim * self.rank),
                        torch.zeros(1, length, self.up_dim * self.rank)
                    ], dim=-1)
                )
                for _ in range(self.num_classes)
            ])



    def _to_index_tensor(self, classes: Union[torch.Tensor, List[int], List[str]]):
        """
        将传入的 classes 转换为 long tensor 的索引（device 未设置）
        支持：
         - torch.Tensor（long 或 int）
         - list[int]
         - list[str]（需要 self.class_to_idx 已建立）
        返回 torch.LongTensor 形状 [B]
        """
        if isinstance(classes, torch.Tensor):
            cls_idx = classes.long()
        elif isinstance(classes, list):
            if len(classes) == 0:
                cls_idx = torch.tensor([], dtype=torch.long)
            elif isinstance(classes[0], str):
                # 字符串列表 -> 索引
                if not self.class_to_idx:
                    raise ValueError("class_to_idx mapping is empty; please call set_lilora with classes_list first.")
                idxs = [self.class_to_idx[c] for c in classes]
                cls_idx = torch.tensor(idxs, dtype=torch.long)
            else:
                # 假定是 list[int]
                cls_idx = torch.tensor(classes, dtype=torch.long)
        else:
            raise TypeError("classes must be torch.Tensor or list[int] or list[str]")

        return cls_idx

    def gen_weight(self, classes: Union[torch.Tensor, List[int], List[str]]):
        """
        为输入的 classes 生成权重。
        - 只为唯一 class 计算一次权重（加速）。
        - 返回：
            weights_per_sample: Tensor [B, length, dim_full]
            weight_list_per_sample: list(length) of Tensor [B, dim_per_lora]
        """
        # 转索引
        class_idx = self._to_index_tensor(classes).to(self.device)  # [B]

        if class_idx.numel() == 0:
            # 空 batch 保护
            return None, None

        # 找到 unique classes 及其 inverse mapping（用于把 unique 权重广播回每个样本）
        unique_idx, inverse_indices = torch.unique(class_idx, return_inverse=True)

        # 从 self.weights 中取出 unique classes 对应的参数（每个 entry 形状 (1, length, dim_full)）
        # 注意 self.weights 是 ParameterList，索引必须是 python int 或可迭代
        weights_unique = torch.cat([self.weights[int(i)].to(self.device) for i in unique_idx], dim=0)
        # weights_unique 形状：[U, length, dim_full]

        # 把 unique 权重映射回每个样本，得到按样本的 weights
        weights_per_sample = weights_unique[inverse_indices]  # -> [B, length, dim_full]

        # 按 lora 条目拆分：从 [B, length, dim_full] -> list of length `length` of [B, dim_per_lora]
        weight_splits = weights_per_sample.split(1, dim=1)  # length 个，每个 [B,1,dim_full]
        weight_list = [w.squeeze(1) for w in weight_splits]  # 每个 [B, dim_full]

        return weights_per_sample, weight_list

    def forward(self, classes: Union[torch.Tensor, List[int], List[str]]):
        """
        前向接口：
        - 接受 classes（tensor 或 list[int] 或 list[str]）
        - 内部会调用 gen_weight，返回 (weights, weight_list)；weights 形状 [B, length, dim_full]
        """
        if self.training and self.gradient_checkpointing:
            weights, weight_list = checkpoint.checkpoint(self.gen_weight, classes)
        else:
            weights, weight_list = self.gen_weight(classes)

        # for key, weight in zip(self.liloras_keys, weight_list):
        #     self.liloras[key].update_weight(weight)

        return weights, weight_list
