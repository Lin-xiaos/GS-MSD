# Importing libraries
import numpy as np
import dgl
from dgl.nn.pytorch import conv
import torch
from torch import nn
import torch.nn.functional as F


# 不一致学习模块
def calculate_js_divergence(text_distribution, image_distribution, audio_distribution):
    """
    计算三个模态（文本、图像、音频）之间的 JS 散度。
    :param text_distribution: 文本模态的分布
    :param image_distribution: 图像模态的分布
    :param audio_distribution: 音频模态的分布
    :return: 三个模态之间的 JS 散度
    """
    # 将输入的分布转换为 PyTorch 张量
    # 这里需要确保输入的张量已经处于正确的设备 (device) 上
    # .clone().detach() 是为了防止inplace操作影响计算图，但如果仅用于计算JS散度，且其不参与主反向传播，
    # 可以在必要时简化。这里保留你的原样。
    text_distribution = text_distribution.clone().detach().to(torch.float32)
    image_distribution = image_distribution.clone().detach().to(torch.float32)
    audio_distribution = audio_distribution.clone().detach().to(torch.float32)

    # 对每个分布应用 softmax 函数，将其转换为概率分布
    # 注意：F.softmax 默认在最后一维操作，如果你的分布是(batch_size, num_classes)
    # 则dim=-1是正确的。如果是(num_classes,)，则dim=0是正确的。这里假设是(num_classes,)
    text_prob = F.softmax(text_distribution, dim=-1) # 假设输出是 (batch_size, num_classes)
    image_prob = F.softmax(image_distribution, dim=-1)
    audio_prob = F.softmax(audio_distribution, dim=-1)

    # 计算平均分布
    m = (text_prob + image_prob + audio_prob) / 3

    # 计算每个模态分布与平均分布之间的 KL 散度
    # F.kl_div(input, target) expects input to be log-probabilities.
    # So, use F.log_softmax(distribution) or distribution.log() if already probabilities
    # kl_div的reduction='sum'会求和所有元素的KL散度。如果你希望批次内的平均，可能需要调整
    kl_text_m = F.kl_div(text_prob.log(), m, reduction='batchmean') # 调整为'batchmean'或'mean'
    kl_image_m = F.kl_div(image_prob.log(), m, reduction='batchmean')
    kl_audio_m = F.kl_div(audio_prob.log(), m, reduction='batchmean')

    # 计算 JS 散度
    # JS散度通常是KL散度的一部分，除以log(2)是为了归一化到[0,1]
    # 但如果kl_div的reduction是'batchmean'，这里也应该是求和后除以3。
    # 确保单位一致性。
    js_divergence = (kl_text_m + kl_image_m + kl_audio_m) / 3 # 如果kl_div是'batchmean'，这里就不再除以np.log(2)了
    # 实际应用中，通常不会除以log(2)除非你需要严格的散度值，对于优化来说，保持比例即可。

    return js_divergence


class CrossModalAttention(nn.Module):
    def __init__(self, feature_dim=256, num_modalities=3):
        super().__init__()
        # 保存输入参数
        self.feature_dim = feature_dim
        self.num_modalities = num_modalities
        # 定义查询、键、值的线性变换层
        self.transform_layer = nn.Linear(feature_dim, feature_dim)
        # 定义注意力分数的 softmax 层
        self.softmax = nn.Softmax(dim=-1)
        # 定义特征投影层
        self.projection_layer = nn.Linear(num_modalities * feature_dim, feature_dim)
        # 定义可学习的模态权重
        self.modality_weights = nn.Parameter(torch.ones(num_modalities))
        # 定义 LayerNorm 用于归一化
        self.layer_norm = nn.LayerNorm(feature_dim)

    def calculate_attention(self, query, key, value):
        """计算注意力特征"""
        # 计算注意力分数
        attention_scores = torch.matmul(query, key.transpose(-2, -1)) / (query.size(-1) ** 0.5)
        # 应用 softmax 函数得到注意力权重
        attention_weights = self.softmax(attention_scores)
        # 计算注意力特征
        return torch.matmul(attention_weights, value)

    def generate_mask(self, attended, attended_modalities):
        """根据各模态的注意力特征与共享特征的差异生成掩码"""
        # 计算各模态注意力特征与共享特征的差异
        differences = [torch.abs(attended - attended_modality) for attended_modality in attended_modalities]
        # 对差异进行 LayerNorm 归一化处理
        norm_differences = [self.layer_norm(diff) for diff in differences]
        # 对模态权重进行 softmax 处理，确保权重之和为 1
        weights = nn.functional.softmax(self.modality_weights, dim=0)
        # 结合各模态的权重计算综合差异
        # 确保weights的维度与diffs匹配以便广播
        combined_diff = sum(w.view(-1, 1, 1) * diff for w, diff in zip(weights, norm_differences))
        # 应用 sigmoid 函数生成掩码
        return torch.sigmoid(combined_diff)

    def forward(self, shared_features, text_features, image_features, audio_features):
        # 检查输入特征的维度
        for features in [shared_features, text_features, image_features, audio_features]:
            # 这里的断言可能会有问题，因为GATConv的out_feats是256，但你可能传递的是features.size(-1)
            # 确保传递给CrossModalAttention的特征维度是匹配的
            assert features.size(-1) == self.feature_dim, f"Input feature dimension must be {self.feature_dim}"
            # 如果特征是(batch_size, feature_dim)，则需要增加一个seq_len维度
            if features.dim() == 2:
                features = features.unsqueeze(1)

        features = [shared_features, text_features, image_features, audio_features]
        # 对所有特征进行变换
        transformed_features = [self.transform_layer(f) for f in features]

        query = transformed_features[0]
        keys = transformed_features[1:]
        values = transformed_features[1:]

        # 计算共享特征的注意力
        # 这里假设shared_features (x_fused) 是query, 其他是keys/values
        attended = self.calculate_attention(query, query, query) # Self-attention on shared_features
        # 计算各模态的注意力 (query (shared) attends to key/value (modalities))
        attended_modalities = [self.calculate_attention(query, k, v) for k, v in zip(keys, values)]

        # 根据综合差异生成掩码
        mask = self.generate_mask(attended, attended_modalities)

        # 动态调整特征的贡献
        adjusted_modalities = [attended_modality * mask for attended_modality in attended_modalities]

        # 合并各模态的注意力特征
        combined_features = torch.cat(adjusted_modalities, dim=-1)
        # 投影到最终维度
        # 这里的输出维度应该是 feature_dim
        return self.projection_layer(combined_features)


class MModel(nn.Module):
    def __init__(self):
        super(MModel, self).__init__()

        # Feature dimensions (assuming initial features are 768, final processed are 256)
        self.initial_feature_dim = 768
        self.gat_output_dim = 256

        # Projection layer for shared space
        self.mm_embedding_space = nn.Sequential(
            nn.Linear(self.initial_feature_dim, self.initial_feature_dim),
            nn.ELU(),
            nn.Dropout(0.4)
        )

        # Multi-head GAT Layer for shared space
        # Note: GATConv output is (batch_size, num_nodes, num_heads * out_feats_per_head)
        # So, out_feats = self.gat_output_dim means each head outputs 256/4 = 64
        self.gat_layer = conv.GATConv(
            in_feats=self.initial_feature_dim,
            out_feats=self.gat_output_dim, # Total output features (num_heads * out_feats_per_head)
            num_heads=4,
            feat_drop=0.5,
            attn_drop=0.0,
            residual=True,
            activation=nn.ELU()
        )

        # Projection layers for additional modalities
        self.additional_modalities_projection =nn.Sequential(
                nn.Linear(self.initial_feature_dim, self.initial_feature_dim),
                nn.ELU(),
                nn.Dropout(0.4)
            )

        # Cross-modal attention layer
        # feature_dim for CrossModalAttention should be self.gat_output_dim (256) after GAT layer
        self.cross_modal_attention = CrossModalAttention(feature_dim=self.gat_output_dim)

        # --- Multi-task decoder module ---

        # Sentiment Analysis
        self.sentimentGRU = nn.GRU(
            input_size=self.gat_output_dim, # Input to GRU is x (fused_features)
            hidden_size=self.gat_output_dim,
            num_layers=1,
            batch_first=True
        )
        self.directSent = nn.Linear(
            in_features=self.gat_output_dim, out_features=3) # 3 for positive/negative/neutral

        # Sarcasm Detection GRU
        # Input to sarcasmGRU is torch.cat([x, x_cross.unsqueeze(1)], dim=2)
        # x is gat_output_dim, x_cross is gat_output_dim, so input_size = 2 * gat_output_dim
        self.sarcasmGRU = nn.GRU(
            input_size=2 * self.gat_output_dim, # Input to GRU is concatenated x and x_cross
            hidden_size=self.gat_output_dim, # Can be same as gat_output_dim or different
            num_layers=1,
            batch_first=True
        )

        # Gate mechanisms for sentiment
        self.sent_gate = nn.Linear(self.gat_output_dim, 1)

        # Gate mechanisms for sarcasm
        # sar_gate_x takes sarInput (2 * gat_output_dim)
        self.sar_gate_x = nn.Linear(2 * self.gat_output_dim, 1)
        # sar_gate_sent takes sentHidden[-1] (gat_output_dim)
        self.sar_gate_sent = nn.Linear(self.gat_output_dim, 1)
        # sar_gate_sar takes sarHidden[-1] (gat_output_dim)
        self.sar_gate_sar = nn.Linear(self.gat_output_dim, 1)

        # --- CLASSIFICATION HEADS ---
        # Stage 1: Binary Sarcasm Classification Head
        # Input features for directSar come from sarHidden[-1] (gat_output_dim) and sentHidden[-1] (gat_output_dim)
        # Concatenated: 2 * gat_output_dim
        self.binary_sarcasm_head = nn.Linear(2 * self.gat_output_dim, 2) # 2 classes: Non-Sarcasm, Sarcasm

        # Stage 2: Fine-grained Sarcasm Classification Head
        # This head also takes the same combined features as binary_sarcasm_head
        self.fine_grained_sarcasm_head = nn.Linear(2 * self.gat_output_dim, 4) # 4 classes: Sarcasm Type A, B, C, D

    def forward(self, g, text_graph, image_graph, audio_graph, stage='binary_sarcasm'):
        # Initial feature extraction for all nodes in the main graph
        x = g.ndata['features'] # (num_nodes, initial_feature_dim)
        text_features = text_graph.ndata['features'] # (num_text_nodes, initial_feature_dim)
        image_features = image_graph.ndata['features'] # (num_image_nodes, initial_feature_dim)
        audio_features = audio_graph.ndata['features'] # (num_audio_nodes, initial_feature_dim)

        # Project to common multimodal space for shared features
        x_shared = self.mm_embedding_space(x) # (num_nodes, initial_feature_dim)

        # Apply GAT layer to shared space (main graph)
        # gat_layer returns (num_nodes, num_heads, out_feats_per_head)
        x_fused_per_head = self.gat_layer(g, x_shared) # (num_nodes, 4, gat_output_dim / 4)
        x_fused = x_fused_per_head.mean(1) # Average over heads to get (num_nodes, gat_output_dim)

        # Process additional modalities separately
        # Note: Your current code takes .mean(0) for text/image/audio features *after* GAT.
        # This assumes each graph (text_graph, etc.) has only one node or you want a single mean vector.
        # If text_graph has multiple nodes, dgl.mean_nodes(text_graph, 'features') is more appropriate.
        # Let's assume for now that these are single-node graphs or you intend to average all nodes.
        text_feature_proj = self.additional_modalities_projection(text_features)
        text_feature_gat = self.gat_layer(text_graph, text_feature_proj)
        # Ensure text_feature is (batch_size, feature_dim) or (batch_size, seq_len, feature_dim) for CMA
        # If your text_graph contains only one node per batch item, this is fine
        # If text_graph contains all text nodes for the batch, then you need to manage batching.
        # Assuming dgl.mean_nodes gives a (batch_size, feature_dim) like structure for the whole graph.
        text_feature = text_feature_gat.mean(1) # Average over heads (num_nodes, gat_output_dim)
        if text_feature.dim() == 2:
            # If after mean(1) it's (num_nodes, feature_dim), and you need (batch_size, feature_dim)
            # You might need to restructure your batching or how dgl.mean_nodes is applied later.
            # For CrossModalAttention, it expects (batch_size, seq_len, feature_dim) or (batch_size, feature_dim)
            # If your graphs are batched, g.batch_size will give you the number of elements in the batch.
            # For simplicity, let's assume text_feature.mean(0) below acts as per-batch feature now.
            text_feature = text_feature.mean(0).unsqueeze(0) # Assuming this results in (1, gat_output_dim) -> (1, 1, gat_output_dim) for CMA input if batch=1
        elif text_feature.dim() == 1:
            text_feature = text_feature.unsqueeze(0).unsqueeze(0) # (1, 1, gat_output_dim)


        image_feature_proj = self.additional_modalities_projection(image_features)
        image_feature_gat = self.gat_layer(image_graph, image_feature_proj)
        image_feature = image_feature_gat.mean(1)
        if image_feature.dim() == 2:
            image_feature = image_feature.mean(0).unsqueeze(0)
        elif image_feature.dim() == 1:
            image_feature = image_feature.unsqueeze(0).unsqueeze(0)

        audio_feature_proj = self.additional_modalities_projection(audio_features)
        audio_feature_gat = self.gat_layer(audio_graph, audio_feature_proj)
        audio_feature = audio_feature_gat.mean(1)
        if audio_feature.dim() == 2:
            audio_feature = audio_feature.mean(0).unsqueeze(0)
        elif audio_feature.dim() == 1:
            audio_feature = audio_feature.unsqueeze(0).unsqueeze(0)


        # Average nodes in the main graph for `x` (this seems to be your global feature for fusion)
        # After GAT, x_fused is (num_nodes, gat_output_dim)
        g.ndata['features'] = x_fused
        x = dgl.mean_nodes(g, 'features') # (batch_size, gat_output_dim) - assuming dgl.mean_nodes handles batching correctly

        # Apply cross-modal attention
        # x_fused here is (num_nodes, gat_output_dim) or (batch_size, num_nodes_per_graph, gat_output_dim)
        # If x_fused is a batch of graphs, you might need to average nodes per graph first,
        # or feed it to cross_modal_attention with its (batch_size, seq_len, feature_dim)
        # Let's assume x_fused from GAT (after mean(1)) is indeed the shared feature for the CMA.
        # And x is the aggregated batch feature.
        # Ensure x_fused and x_cross have the same shape for CMA if you intend x_fused to be the query.
        # If x is the query, then x.unsqueeze(1) needs to be passed.
        # Let's adjust x_fused to be the shared_features, and it needs to be (batch_size, 1, feature_dim) for CMA.
        x_fused_for_cma = x.unsqueeze(1) # Now (batch_size, 1, gat_output_dim)

        # CrossModalAttention expects (batch_size, seq_len, feature_dim) for all inputs
        # The text/image/audio_feature should also be (batch_size, seq_len, feature_dim)
        # Assuming the .mean(0).unsqueeze(0) makes them (1, 1, feature_dim) for a single sample.
        # If batch_size > 1, these also need to be batched.
        # For simplicity, let's assume they are already batched with a seq_len of 1.
        text_feature_cma = text_feature # Already (batch_size, 1, feature_dim) from processing above
        image_feature_cma = image_feature
        audio_feature_cma = audio_feature

        # x_cross output of CMA is (batch_size, 1, feature_dim)
        x_cross = self.cross_modal_attention(x_fused_for_cma, text_feature_cma, image_feature_cma, audio_feature_cma)
        x_cross = x_cross.squeeze(1) # (batch_size, gat_output_dim) for subsequent layers

        # JS Divergence is calculated on distributions (logits or probabilities)
        # These features (text_feature, image_feature, audio_feature) are typically raw feature vectors
        # If you want JS divergence, they should probably be outputs of a classification head per modality,
        # or have a linear layer to project them to class distribution space (e.g., 5 classes).
        # Assuming they are outputs of some "class distribution" prediction for JS loss.
        js_divergence_loss = calculate_js_divergence(
            text_feature.squeeze(1), # (batch_size, feature_dim)
            image_feature.squeeze(1),
            audio_feature.squeeze(1)
        )

        # Prepare input for GRU
        x_gru_input = x.unsqueeze(1) # Add sequence dimension: (batch_size, 1, gat_output_dim)

        # --- Multi-task processing ---

        # 1. Sentiment Analysis
        sentGRU_out, sentHidden = self.sentimentGRU(x_gru_input) # sentHidden[-1] is (batch_size, gat_output_dim)
        sent_gate_value = torch.sigmoid(self.sent_gate(sentHidden[-1]))
        sentOutput = self.directSent(sent_gate_value * sentHidden[-1]) # (batch_size, 3)

        # 2. Sarcasm Detection
        # sarInput is (batch_size, 1, 2 * gat_output_dim)
        sarInput = torch.cat([x_gru_input, x_cross.unsqueeze(1)], dim=2)

        sar_gate_x_value = torch.sigmoid(self.sar_gate_x(sarInput))
        sarGRU_out, sarHidden = self.sarcasmGRU(sar_gate_x_value * sarInput) # sarHidden[-1] is (batch_size, gat_output_dim)

        sar_gate_sent_value = torch.sigmoid(self.sar_gate_sent(sentHidden[-1]))
        sar_gate_sar_value = torch.sigmoid(self.sar_gate_sar(sarHidden[-1]))

        # Final combined features for sarcasm classification heads
        final_sarcasm_features = torch.cat([sar_gate_sar_value * sarHidden[-1], sar_gate_sent_value * sentHidden[-1]], dim=1) # (batch_size, 2 * gat_output_dim)

        # --- Conditional Output based on Stage ---
        if stage == 'binary_sarcasm':
            # Stage 1: Binary Sarcasm Classification
            sarOutput = self.binary_sarcasm_head(final_sarcasm_features) # (batch_size, 2)
            return sarOutput, sentOutput, js_divergence_loss
        elif stage == 'fine_grained_sarcasm':
            # Stage 2: Fine-grained Sarcasm Classification
            sarOutput = self.fine_grained_sarcasm_head(final_sarcasm_features) # (batch_size, 4)
            return sarOutput, sentOutput, js_divergence_loss # sentOutput and JS loss still returned for consistency, but might not be used in stage 2 loss
        else: # For inference or combined output
            # This part will be used during inference
            binary_sar_logits = self.binary_sarcasm_head(final_sarcasm_features)
            # You might want to return the actual prediction here or just the logits for external handling
            return binary_sar_logits, sentOutput, js_divergence_loss # Returning binary logits for inference