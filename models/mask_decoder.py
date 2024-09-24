# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch
from torch import Tensor, nn
from torch.nn import functional as F

import math

from typing import List, Tuple, Type



class MLPBlock(nn.Module):
    def __init__(
        self,
        embedding_dim: int,
        mlp_dim: int,
        act: Type[nn.Module] = nn.GELU,
    ) -> None:
        super().__init__()
        self.lin1 = nn.Linear(embedding_dim, mlp_dim)
        self.lin2 = nn.Linear(mlp_dim, embedding_dim)
        self.act = act()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.lin2(self.act(self.lin1(x)))


# From https://github.com/facebookresearch/detectron2/blob/main/detectron2/layers/batch_norm.py # noqa
# Itself from https://github.com/facebookresearch/ConvNeXt/blob/d1fa8f6fef0a165b27399986cc2bdacc92777e40/models/convnext.py#L119  # noqa
class LayerNorm2d(nn.Module):
    def __init__(self, num_channels: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(num_channels))
        self.bias = nn.Parameter(torch.zeros(num_channels))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        u = x.mean(1, keepdim=True)
        s = (x - u).pow(2).mean(1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.eps)
        x = self.weight[:, None, None] * x + self.bias[:, None, None]
        return x


class LayerNorm1d(nn.Module):
    def __init__(self, num_channels: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(num_channels))
        self.bias = nn.Parameter(torch.zeros(num_channels))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        u = x.mean(1, keepdim=True)
        s = (x - u).pow(2).mean(1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.eps)
        x = self.weight[:, None] * x + self.bias[:, None]
        return x

# 这个MaskDecoder类实现了基于transformer的掩码解码器。
class MaskDecoder(nn.Module):
    # __init__方法:
    # 1. 输入参数:
    #     - transformer_dim: transformer 的通道维度
    #     - transformer: 使用的 transformer
    #     - num_multimask_outputs: 在消除掩码歧义时预测的掩码数量。
    #     - activation: 上采样掩码时使用的激活函数类型
    #     - iou_head_depth: 用于预测掩码质量的 MLP 的深度
    #     - iou_head_hidden_dim: 用于预测掩码质量的 MLP 的隐藏维度
    # 2. 记录 transformer_dim 和 transformer。
    # 3. 记录 num_multimask_outputs。
    # 4. 嵌入 iou_token 和 mask_tokens。
    # 5. 定义 output_upscaling 为上采样器,用于上采样 transformer 的输出以得到掩码。
    # 6. 定义 output_hypernetworks_mlps 为 MLP 列表,个数为 num_mask_tokens, 用于从 transformer 的输出生成掩码通道。
    # 7. 定义 iou_prediction_head 为 MLP,用于从 transformer 的输出预测掩码的 IOU。
    def __init__(
        self,
        *,
        transformer_dim: int,
        transformer: nn.Module,
        num_channel_inputs: int = 1,
        num_channel_outputs: int = 1,
        num_multimask_outputs: int = 3,
        activation: Type[nn.Module] = nn.GELU,
        iou_head_depth: int = 3,
        iou_head_hidden_dim: int = 256,
    ) -> None:
        """
        Predicts masks given an image and prompt embeddings, using a
        transformer architecture.

        Arguments:
          transformer_dim (int): the channel dimension of the transformer
          transformer (nn.Module): the transformer used to predict masks
          num_multimask_outputs (int): the number of masks to predict
            when disambiguating masks
          activation (nn.Module): the type of activation to use when
            upscaling masks
          iou_head_depth (int): the depth of the MLP used to predict
            mask quality
          iou_head_hidden_dim (int): the hidden dimension of the MLP
            used to predict mask quality
        """
        super().__init__()
        self.transformer_dim = transformer_dim
        self.transformer = transformer

        self.num_multimask_outputs = num_multimask_outputs
        self.num_channel_inputs = num_channel_inputs
        self.num_channel_outputs = num_channel_outputs

        self.iou_token = nn.Embedding(1, transformer_dim)
        self.num_mask_tokens = num_multimask_outputs
        self.mask_tokens = nn.Embedding(self.num_mask_tokens, transformer_dim)

        self.output_upscaling = nn.Sequential(
            nn.ConvTranspose1d(transformer_dim, transformer_dim // 2, kernel_size=4, stride=2, padding=1),
            LayerNorm1d(transformer_dim // 2),
            activation(),
            nn.ConvTranspose1d(transformer_dim // 2, transformer_dim // 4, kernel_size=4, stride=4, padding=0),
            LayerNorm1d(transformer_dim // 4),
            activation(),
            nn.ConvTranspose1d(transformer_dim // 4, transformer_dim // 8, kernel_size=8, stride=8, padding=0),
            activation()
        )
        self.output_hypernetworks_mlps = nn.ModuleList(
            [
                MLP(transformer_dim, transformer_dim, transformer_dim // 8, 3)
                for i in range(self.num_mask_tokens)
            ]
        )

        self.output_channel_aggretator = nn.Linear(self.num_channel_inputs, self.num_channel_outputs)

        # self.iou_prediction_head = MLP(
        #     transformer_dim, iou_head_hidden_dim, self.num_mask_tokens, iou_head_depth
        # )

    # 这个 forward 方法的作用是根据图像和 prompt 的 embedding 预测掩码。它包含:
    # 1. 输入参数:
    #     - image_embeddings: 图像编码器的输出
    #     - image_pe: 与 image_embeddings 形状相同的位置编码
    #     - sparse_prompt_embeddings: 点和框的 embedding
    #     - dense_prompt_embeddings: 掩码输入的 embedding
    #     - multimask_output: 是否返回多个掩码或单个掩码
    # 2. 调用 predict_masks 根据图像和 prompt 的 embedding 预测掩码 masks 和掩码质量 iou_pred。
    # 3. 如果 multimask_output 为 True,则选择 masks 的第 1 个维度后的全部切片。否则选择第一个切片。
    # 4. 相应地选择 iou_pred 的切片。
    # 5. 准备输出,返回 masks 和 iou_pred。
    # 所以,这个 forward 方法实现了根据图像和 prompt 的 embedding 预测掩码的功能。
    # 它可以根据输入的 prompt 学习掩码生成的高度非线性映射,为 prompt 驱动生成模型提供掩码预测的关键能力。
    # 这个 forward 方法提供了根据 prompt 预测掩码的具体实现。它发挥了 MaskDecoder 类的强大功能,
    # 可以解码出复杂的定制化掩码,为实现高质量的 prompt 驱动生成模型提供强有力的支持。
    def forward(
        self,
        image_embeddings: torch.Tensor,
        image_pe: torch.Tensor,
        sparse_prompt_embeddings: torch.Tensor,
        dense_prompt_embeddings: torch.Tensor,
        multimask_output: bool,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Predict masks given image and prompt embeddings.

        Arguments:
          image_embeddings (torch.Tensor): the embeddings from the image encoder
          image_pe (torch.Tensor): positional encoding with the shape of image_embeddings
          sparse_prompt_embeddings (torch.Tensor): the embeddings of the points and boxes
          dense_prompt_embeddings (torch.Tensor): the embeddings of the mask inputs
          multimask_output (bool): Whether to return multiple masks or a single
            mask.

        Returns:
          torch.Tensor: batched predicted masks
          torch.Tensor: batched predictions of mask quality
        """
        masks = self.predict_masks(
            image_embeddings=image_embeddings,
            image_pe=image_pe
        )

        # Select the correct mask or masks for output
        # if multimask_output:
        #     mask_slice = slice(1, None)
        # else:
        #     mask_slice = slice(0, 1)
        # masks = masks[:, mask_slice, :, :]
        # iou_pred = iou_pred[:, mask_slice]

        # Prepare output
        return masks

    # 这个 predict_masks 方法的作用是预测掩码。它包含:
    # 1. 输入参数:
    #     - image_embeddings: 图像编码器的输出
    #     - image_pe: 与 image_embeddings 形状相同的位置编码
    #     - sparse_prompt_embeddings: 点和框的 embedding
    #     - dense_prompt_embeddings: 掩码输入的 embedding
    # 2. 拼接 iou_token 和 mask_tokens 作为输出 tokens, 扩展至 batch 大小, 与 sparse_prompt_embeddings 拼接作为 tokens。
    # 3. 通过 torch.repeat_interleave 扩展 src 和 pos_src 至与 tokens 相同的 batch 大小。
    # 4. 将 src 和 pos_src 以及 tokens 输入 transformer, 获得 hs 和 src。
    # 5. 获得 iou_token_out 和 mask_tokens_out 作为 transformer 的输出。
    # 6. 上采样 src 得到 upscaled_embedding。
    # 7. 对 mask_tokens_out 中的每个 token, 使用对应 MLP 得到 hyper_in_list 中的 tensor。
    # 8. 使用 torch.stack 将 hyper_in_list 拼接为 hyper_in。
    # 9. 计算 masks=(hyper_in @ upscaled_embedding.view(b, c, h * w)), 形状为 (b, num_mask_tokens, h, w)。
    # 10. 使用 iou_prediction_head 从 iou_token_out 预测 iou_pred。
    # 11. 返回 masks 和 iou_pred。
    # 所以,这个 predict_masks 方法实现了根据prompt预测掩码的功能。
    # 它发挥 transformer 和上采样器的功能,可以从 prompt 学习生成模型的参数
    # 这个 predict_masks 方法提供了根据 prompt 预测掩码的具体实现。
    # 它利用 MaskDecoder 的强大功能,可以解码出复杂的定制化掩码,为实现高质量的 prompt 驱动生成模型提供关键支持。

    def predict_masks(
        self,
        image_embeddings: torch.Tensor, # Bx(embed_dim=256 in vit-h)x(embed_H)x(embed_W)
        image_pe: torch.Tensor, # Bx(embed_dim=256 in vit-h)x(embed_H)x(embed_W)
    ) -> Tensor:
        """Predicts masks. See 'forward' for more details."""
        # Concatenate output tokens
        # output_tokens = torch.cat([self.iou_token.weight, self.mask_tokens.weight], dim=0)
        # output_tokens = output_tokens.unsqueeze(0).expand(sparse_prompt_embeddings.size(0), -1, -1)
        # tokens = torch.cat((output_tokens, sparse_prompt_embeddings), dim=1)

        tokens = torch.cat([self.mask_tokens.weight], dim=0)
        tokens = tokens.unsqueeze(0).expand(image_embeddings.size(0), -1, -1)

        # Expand per-image data in batch direction to be per-mask
        # src = torch.repeat_interleave(image_embeddings, tokens.shape[0], dim=0)
        # src = src + dense_prompt_embeddings
        # pos_src = torch.repeat_interleave(image_pe, tokens.shape[0], dim=0)
        # b, c, h, w = src.shape

        src = image_embeddings
        src = src.permute(0, 3, 1, 2)
        # tokens = src.flatten(2).permute(0, 2, 1)
        b, c, h, w = src.shape
        pos_src = image_pe
        pos_src = pos_src.permute(0, 3, 1, 2)

        # Run the transformer
        hs, src = self.transformer(src, pos_src, tokens)

        # Upscale mask embeddings and predict masks using the mask tokens
        src = src.transpose(1, 2).view(b, c, h, w)
        src = src.permute(0, 2, 1, 3).view(b * h, c, w)
        upscaled_embedding = self.output_upscaling(src)
        upscaled_embedding = upscaled_embedding.view(b, h, *upscaled_embedding.shape[1:]).permute(0, 2, 3, 1)
        upscaled_embedding = self.output_channel_aggretator(upscaled_embedding)
        upscaled_embedding = upscaled_embedding.permute(0, 1, 3, 2)
        hyper_in_list: List[torch.Tensor] = []
        for i in range(self.num_mask_tokens):
            hyper_in_list.append(self.output_hypernetworks_mlps[i](hs[:, i, :]))
        hyper_in = torch.stack(hyper_in_list, dim=1)
        b, c, h, w = upscaled_embedding.shape
        masks = (hyper_in @ upscaled_embedding.reshape(b, c, h * w)).view(b, -1, h, w)

        # Generate mask quality predictions
        # iou_pred = self.iou_prediction_head(iou_token_out)

        return masks

# 这个 MLP 类实现了多层感知机 (Multi-Layer Perceptron)。
# Lightly adapted from
# https://github.com/facebookresearch/MaskFormer/blob/main/mask_former/modeling/transformer/transformer_predictor.py # noqa
class MLP(nn.Module):
    # __init__方法:
    # 1. 输入参数:
    #     - input_dim: 输入维度
    #     - hidden_dim: 隐藏层维度
    #     - output_dim: 输出维度
    #     - num_layers: 隐藏层数
    #     - sigmoid_output: 是否使用 sigmoid 激活函数
    # 2. 记录 num_layers 和 h 为 num_layers-1 个隐藏层维度。
    # 3. 实例化 nn.ModuleList 由 nn.Linear 组成的列表,用于实现 MLP 的线性变换。
    # 4. 记录 sigmoid_output 以决定是否使用 sigmoid 激活函数。
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        num_layers: int,
        sigmoid_output: bool = False,
    ) -> None:
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(
            nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim])
        )
        self.sigmoid_output = sigmoid_output

    # forward 方法:
    # 1. 对输入 x 重复 num_layers 次线性变换和激活。
    # 2. 最后一层只使用线性变换,不使用激活函数。
    # 3. 如果 sigmoid_output 为 True, 使用 sigmoid 激活函数。
    # 4. 返回 MLP 的输出。
    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        if self.sigmoid_output:
            x = F.sigmoid(x)
        return x

# 这个 TwoWayTransformer 类实现了双向transformer解码器。
class TwoWayTransformer(nn.Module):
    # __init__方法:
    # 1. 输入参数:
    #     - depth: transformer 的层数
    #     - embedding_dim: 输入 embedding 的通道维度
    #     - num_heads: 多头注意力的头数
    #     - mlp_dim: MLP 块内部的通道维度
    #     - activation: MLP 块使用的激活函数
    #     - attention_downsample_rate: 注意力下采样率
    # 2. 记录 depth、embedding_dim、num_heads 和 mlp_dim。
    # 3. 定义 layers 为 nn.ModuleList, 包含 depth 个 TwoWayAttentionBlock。
    # 4. 定义 final_attn_token_to_image 为从点到图像的注意力层。
    # 5. 定义 norm_final_attn 为 final_attn_token_to_image 的 LayerNorm。

    def __init__(
            self,
            depth: int,
            embedding_dim: int,
            num_heads: int,
            mlp_dim: int,
            activation: Type[nn.Module] = nn.ReLU,
            attention_downsample_rate: int = 2,
    ) -> None:
        """
        A transformer decoder that attends to an input image using
        queries whose positional embedding is supplied.

        Args:
          depth (int): number of layers in the transformer
          embedding_dim (int): the channel dimension for the input embeddings
          num_heads (int): the number of heads for multihead attention. Must
            divide embedding_dim
          mlp_dim (int): the channel dimension internal to the MLP block
          activation (nn.Module): the activation to use in the MLP block
        """
        super().__init__()
        self.depth = depth
        self.embedding_dim = embedding_dim
        self.num_heads = num_heads
        self.mlp_dim = mlp_dim
        self.layers = nn.ModuleList()

        for i in range(depth):
            self.layers.append(
                TwoWayAttentionBlock(
                    embedding_dim=embedding_dim,
                    num_heads=num_heads,
                    mlp_dim=mlp_dim,
                    activation=activation,
                    attention_downsample_rate=attention_downsample_rate,
                    skip_first_layer_pe=(i == 0),
                )
            )

        self.final_attn_token_to_image = Attention(
            embedding_dim, num_heads, downsample_rate=attention_downsample_rate
        )
        self.norm_final_attn = nn.LayerNorm(embedding_dim)

    # forward 方法:
    # 1. 输入参数:
    #     - image_embedding: 要处理的图像,形状为 B x embedding_dim x h x w
    #     - image_pe: 与 image_embedding 形状相同的位置编码
    #     - point_embedding: 要添加到查询点的 embedding ,形状为 B x N_points x embedding_dim
    # 2. 将 image_embedding 变形为 B x HW x C, image_pe 相应变形。
    # 3. 将 queries 初始化为 point_embedding, keys 初始化为 image_embedding。
    # 4. 对 queries 和 keys 重复使用 layers 中的 TwoWayAttentionBlock。
    # 5. 应用 final_attn_token_to_image 从 points 到 image 的注意力。
    # 6. 使用 norm_final_attn 规范化 queries。
    # 7. 返回 queries 和 keys。

    def forward(
            self,
            image_embedding: Tensor,
            image_pe: Tensor,
            point_embedding: Tensor,
    ) -> Tuple[Tensor, Tensor]:
        """
        Args:
          image_embedding (torch.Tensor): image to attend to. Should be shape
            B x embedding_dim x h x w for any h and w.
          image_pe (torch.Tensor): the positional encoding to add to the image. Must
            have the same shape as image_embedding.
          point_embedding (torch.Tensor): the embedding to add to the query points.
            Must have shape B x N_points x embedding_dim for any N_points.

        Returns:
          torch.Tensor: the processed point_embedding
          torch.Tensor: the processed image_embedding
        """
        # BxCxHxW -> BxHWxC == B x N_image_tokens x C
        bs, c, h, w = image_embedding.shape
        image_embedding = image_embedding.flatten(2).permute(0, 2, 1)
        image_pe = image_pe.flatten(2).permute(0, 2, 1)

        # Prepare queries
        queries = point_embedding
        keys = image_embedding

        # Apply transformer blocks and final layernorm
        for layer in self.layers:
            queries, keys = layer(
                queries=queries,
                keys=keys,
                query_pe=point_embedding,
                key_pe=image_pe,
            )

        # Apply the final attention layer from the points to the image
        q = queries + point_embedding
        k = keys + image_pe
        attn_out = self.final_attn_token_to_image(q=q, k=k, v=keys)
        queries = queries + attn_out
        queries = self.norm_final_attn(queries)

        return queries, keys

# 这个 TwoWayAttentionBlock 类实现了 transformer 块,包含四个层:
# 1. 稀疏输入的自注意力
# 2. 稀疏输入到密集输入的交叉注意力
# 3. 稀疏输入的 MLP 块
# 4. 密集输入到稀疏输入的交叉注意力
class TwoWayAttentionBlock(nn.Module):
    # __init__方法:
    # 1. 输入参数:
    #     - embedding_dim: embedding 的通道维度
    #     - num_heads: 注意力层中的头数
    #     - mlp_dim: MLP 块的隐藏维度
    #     - activation: MLP 块的激活函数
    #     - attention_downsample_rate: 注意力下采样率
    #     - skip_first_layer_pe: 是否跳过第一层的位置编码
    # 2. 定义 self_attn 为自注意力层。
    # 3. 定义 norm1 为 self_attn 的 LayerNorm。
    # 4. 定义 cross_attn_token_to_image 为从 token 到 image 的交叉注意力层。
    # 5. 定义 norm2 为 cross_attn_token_to_image 的 LayerNorm。
    # 6. 定义 mlp 为 MLP 块。
    # 7. 定义 norm3 为 mlp 的 LayerNorm。
    # 8. 定义 norm4 为 LayerNorm。
    # 9. 定义 cross_attn_image_to_token 为从 image 到 token 的交叉注意力层。
    # 10. 记录 skip_first_layer_pe。
    def __init__(
            self,
            embedding_dim: int,
            num_heads: int,
            mlp_dim: int = 2048,
            activation: Type[nn.Module] = nn.ReLU,
            attention_downsample_rate: int = 2,
            skip_first_layer_pe: bool = False,
    ) -> None:
        """
        A transformer block with four layers: (1) self-attention of sparse
        inputs, (2) cross attention of sparse inputs to dense inputs, (3) mlp
        block on sparse inputs, and (4) cross attention of dense inputs to sparse
        inputs.

        Arguments:
          embedding_dim (int): the channel dimension of the embeddings
          num_heads (int): the number of heads in the attention layers
          mlp_dim (int): the hidden dimension of the mlp block
          activation (nn.Module): the activation of the mlp block
          skip_first_layer_pe (bool): skip the PE on the first layer
        """
        super().__init__()
        self.self_attn = Attention(embedding_dim, num_heads)
        self.norm1 = nn.LayerNorm(embedding_dim)

        self.cross_attn_token_to_image = Attention(
            embedding_dim, num_heads, downsample_rate=attention_downsample_rate
        )
        self.norm2 = nn.LayerNorm(embedding_dim)

        self.mlp = MLPBlock(embedding_dim, mlp_dim, activation)
        self.norm3 = nn.LayerNorm(embedding_dim)

        self.norm4 = nn.LayerNorm(embedding_dim)
        self.cross_attn_image_to_token = Attention(
            embedding_dim, num_heads, downsample_rate=attention_downsample_rate
        )

        self.skip_first_layer_pe = skip_first_layer_pe

    # forward 方法:
    # 1. 输入:
    #     - queries: 稀疏输入,即点输入
    #     - query_pe: query 的位置编码
    #     - keys: 密集输入,即图像输入
    #     - key_pe: key 的位置编码
    # 2. 如果 skip_first_layer_pe 为 True，则 qkv 都来自 queries
    # 3. 使用 self_attn 计算 queries 的自注意力。
    # 4. 通过 norm1 规范化 queries。
    # 5. 使用 cross_attn_token_to_image 计算 queries 到 keys 的注意力。
    # 6. 通过 norm2 规范化 queries。
    # 7. 使用 mlp 更新 queries。
    # 8. 通过 norm3 规范化 queries。
    # 9. 使用 cross_attn_image_to_token 计算 keys 到 queries 的注意力。
    # 10. 通过 norm4 规范化 queries。
    # 11. 返回 queries 和 keys。
    def forward(
            self, queries: Tensor, keys: Tensor, query_pe: Tensor, key_pe: Tensor
    ) -> Tuple[Tensor, Tensor]:
        # Self attention block
        if self.skip_first_layer_pe:
            queries = self.self_attn(q=queries, k=queries, v=queries)
        else:
            q = queries + query_pe
            attn_out = self.self_attn(q=q, k=q, v=queries)
            queries = queries + attn_out
        queries = self.norm1(queries)

        # Cross attention block, tokens attending to image embedding
        q = queries + query_pe
        k = keys + key_pe
        attn_out = self.cross_attn_token_to_image(q=q, k=k, v=keys)
        queries = queries + attn_out
        queries = self.norm2(queries)

        # MLP block
        mlp_out = self.mlp(queries)
        queries = queries + mlp_out
        queries = self.norm3(queries)

        # Cross attention block, image embedding attending to tokens
        q = queries + query_pe
        k = keys + key_pe
        attn_out = self.cross_attn_image_to_token(q=k, k=q, v=queries)
        keys = keys + attn_out
        keys = self.norm4(keys)

        return queries, keys

# 这个Attention类实现了带下采样的注意力机制。
class Attention(nn.Module):
    """
    An attention layer that allows for downscaling the size of the embedding
    after projection to queries, keys, and values.
    """

    # __init__ 方法:
    # 1. 输入参数:
    #     - embedding_dim: embedding 的维度
    #     - num_heads: 多头注意力的头数
    #     - downsample_rate: 下采样率
    # 2. 计算 internal_dim 为 embedding_dim 除以 downsample_rate。
    # 3. 确保 internal_dim 可以被 num_heads 整除。
    # 4. 定义 q_proj、k_proj 和 v_proj为 输入的投影层,将 embedding_dim 映射到 internal_dim。
    # 5. 定义 out_proj 为输出的投影层,将 internal_dim 映射回 embedding_dim。
    def __init__(
            self,
            embedding_dim: int,
            num_heads: int,
            downsample_rate: int = 1,
    ) -> None:
        super().__init__()
        self.embedding_dim = embedding_dim
        self.internal_dim = embedding_dim // downsample_rate
        self.num_heads = num_heads
        assert self.internal_dim % num_heads == 0, "num_heads must divide embedding_dim."

        self.q_proj = nn.Linear(embedding_dim, self.internal_dim)
        self.k_proj = nn.Linear(embedding_dim, self.internal_dim)
        self.v_proj = nn.Linear(embedding_dim, self.internal_dim)
        self.out_proj = nn.Linear(self.internal_dim, embedding_dim)

    # _separate_heads方法:
    # 1. 将 x 分离为 num_heads 个头, x 的形状变为 b x n x num_heads x c // num_heads。
    # 2. 交换第二和第三维, x 的形状变为 b x num_heads x n x c // num_heads。
    # 3. 返回x。
    def _separate_heads(self, x: Tensor, num_heads: int) -> Tensor:
        b, n, c = x.shape
        x = x.reshape(b, n, num_heads, c // num_heads)
        return x.transpose(1, 2)  # B x N_heads x N_tokens x C_per_head

    # _recombine_heads 方法:
    # 1. x 的形状为 b x num_heads x n x c // num_heads。
    # 2. 交换第二和第三维,x的形状变为 b x n x num_heads x c // num_heads。
    # 3. 将 x 变形为 b x n x num_heads * c // num_heads。
    # 4. 返回 x。
    def _recombine_heads(self, x: Tensor) -> Tensor:
        b, n_heads, n_tokens, c_per_head = x.shape
        x = x.transpose(1, 2)
        return x.reshape(b, n_tokens, n_heads * c_per_head)  # B x N_tokens x C

    # forward方法:
    # 1. 对 q、k 和 v 使用 q_proj、k_proj 和 v_proj 进行投影,将 embedding_dim 映射到 internal_dim。
    # 2. 使用 _separate_heads 将 q、k 和 v 分离为 num_heads 个头。
    # 3. 计算 attn 为 q 和 k 的点积,除以 c_per_head 开根号,再使用 softmax 归一化。
    # 4. 使用 attn 和 v 计算 out。
    # 5. 使用 _recombine_heads 重新组合出 num_heads 个头。
    # 6. 使用 out_proj 将 out 投影回 embedding_dim。
    # 7. 返回 out。
    def forward(self, q: Tensor, k: Tensor, v: Tensor) -> Tensor:
        # Input projections
        q = self.q_proj(q)
        k = self.k_proj(k)
        v = self.v_proj(v)

        # Separate into heads
        q = self._separate_heads(q, self.num_heads)
        k = self._separate_heads(k, self.num_heads)
        v = self._separate_heads(v, self.num_heads)

        # Attention
        _, _, _, c_per_head = q.shape
        attn = q @ k.permute(0, 1, 3, 2)  # B x N_heads x N_tokens x N_tokens
        attn = attn / math.sqrt(c_per_head)
        attn = torch.softmax(attn, dim=-1)

        # Get output
        out = attn @ v
        out = self._recombine_heads(out)
        out = self.out_proj(out)

        return out

