import torch
import torch.nn as nn
import torch.nn.functional as F


# Patch Embedding (ViT)
class PatchEmbedding(nn.Module):
    def __init__(self, img_size, patch_size, in_channels, embed_dim):
        super(PatchEmbedding, self).__init__()
        self.proj = nn.Conv2d(in_channels, embed_dim, kernel_size=patch_size, stride=patch_size)
        self.flatten = nn.Flatten(2)

    def forward(self, x):
        x = self.proj(x)  # (batch_size, embed_dim, H', W')
        x = self.flatten(x)  # (batch_size, embed_dim, num_patches)
        x = x.permute(0, 2, 1)  # (batch_size, num_patches, embed_dim)
        return x



# TCN
class TCNEmbedding(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, num_layers):
        super(TCNEmbedding, self).__init__()
        layers = []
        for i in range(num_layers):
            layers.append(nn.Conv1d(in_channels if i == 0 else out_channels, out_channels, kernel_size, padding="same"))
            layers.append(nn.ReLU())
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        x = x.permute(0, 2, 1)  #  (batch_size, in_channels, seq_len)
        x = self.net(x)
        return x.permute(0, 2, 1)  #  (batch_size, seq_len, out_channels)


# multihead
class MultiHeadAttention(nn.Module):
    def __init__(self, embed_dim, num_heads):
        super(MultiHeadAttention, self).__init__()
        self.attention = nn.MultiheadAttention(embed_dim, num_heads)

    def forward(self, x):
        x = x.permute(1, 0, 2)  # (seq_len, batch_size, embed_dim)
        attn_output, _ = self.attention(x, x, x)
        return attn_output.permute(1, 0, 2)  # (batch_size, seq_len, embed_dim)


# feedforward
class FeedForward(nn.Module):
    def __init__(self, embed_dim, hidden_dim):
        super(FeedForward, self).__init__()
        self.fc1 = nn.Linear(embed_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, embed_dim)
        self.activation = nn.ReLU()
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x):
        residual = x
        x = self.fc1(x)
        x = self.activation(x)
        x = self.fc2(x)
        x = self.norm(x + residual)
        return x



class ImageEncoder(nn.Module):
    def __init__(self, embed_dim, num_heads, hidden_dim, num_layers):
        super(ImageEncoder, self).__init__()
        self.layers = nn.ModuleList([
            nn.Sequential(
                MultiHeadAttention(embed_dim, num_heads),
                FeedForward(embed_dim, hidden_dim)
            ) for _ in range(num_layers)
        ])

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x



class CrossAttention(nn.Module):
    def __init__(self, embed_dim, num_heads):
        super(CrossAttention, self).__init__()
        self.attention = nn.MultiheadAttention(embed_dim, num_heads)

    def forward(self, query, key, value):
        query = query.permute(1, 0, 2)
        key = key.permute(1, 0, 2)
        value = value.permute(1, 0, 2)
        attn_output, _ = self.attention(query, key, value)
        return attn_output.permute(1, 0, 2)


class GatedResidualNetwork(nn.Module):
    def __init__(self, embed_dim):
        super(GatedResidualNetwork, self).__init__()
        self.fc1 = nn.Linear(embed_dim, embed_dim)
        self.fc2 = nn.Linear(embed_dim, embed_dim)
        self.elu = nn.ELU()
        self.dropout = nn.Dropout(0.1)
        self.gate = nn.Sigmoid()

    def forward(self, x):
        residual = x
        x = self.fc1(x)
        x = self.elu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        gate = self.gate(x)
        return gate * x + (1 - gate) * residual


class AdaptiveModule(nn.Module):
    def __init__(self, input_dim, output_dim, num_layers=2):
        """
        Adaptive Module with N layers of MLP for feature fusion.
        :param input_dim: Total dimension of all concatenated features.
        :param output_dim: Final output dimension after fusion.
        :param num_layers: Number of MLP layers for fusion.
        """
        super(AdaptiveModule, self).__init__()
        layers = []
        for i in range(num_layers):
            in_dim = input_dim if i == 0 else output_dim
            layers.append(nn.Linear(in_dim, output_dim))
            layers.append(nn.ReLU())
        self.net = nn.Sequential(*layers)

    def forward(self, *inputs):
        """
        Perform adaptive fusion for multiple inputs.
        :param inputs: Variable number of tensor inputs to concatenate and fuse.
        :return: Fused feature of specified output_dim.
        """
        x = torch.cat(inputs, dim=-2)  # Concatenate all inputs along the feature dimension
        batch_size, seq_len, feature_dim = x.shape
        x = x.reshape(batch_size * seq_len, feature_dim)  # Flatten to (batch_size * seq_len, feature_dim)
        x = self.net(x)  # Pass through MLP
        x = x.reshape(batch_size, seq_len, -1)  # Reshape back to (batch_size, seq_len, output_dim)
        return x


class MultiModalModel(nn.Module):
    def __init__(self, img_size, patch_size, in_channels, embed_dim, num_heads, hidden_dim, num_layers, seq_channels, kernel_size, adaptive_layers=2):
        super(MultiModalModel, self).__init__()
        # Patch Embedding for Images
        self.patch_embedding = PatchEmbedding(img_size, patch_size, in_channels, embed_dim)
        # TCN for Sequence
        self.tcn_embedding = TCNEmbedding(seq_channels, embed_dim, kernel_size, num_layers=2)
        # Sequence Encoder
        self.seq_encoder = ImageEncoder(embed_dim, num_heads, hidden_dim, num_layers)
        # Image Encoders
        self.image_encoder1 = ImageEncoder(embed_dim, num_heads, hidden_dim, num_layers)
        self.image_encoder2 = ImageEncoder(embed_dim, num_heads, hidden_dim, num_layers)
        # Cross Attention
        self.cross_attention1 = CrossAttention(embed_dim, num_heads)
        self.cross_attention2 = CrossAttention(embed_dim, num_heads)
        # FeedForward for Image Features (only for img_feat1)
        self.feedforward = FeedForward(embed_dim, hidden_dim)
        # Gated Residual Network
        self.grn = GatedResidualNetwork(embed_dim)
        # Adaptive Module
        adaptive_input_dim = embed_dim * 5  # Concatenate 5 inputs (img_feat1_transformed, fusion2, ...)
        self.adaptive_module = AdaptiveModule(adaptive_input_dim, embed_dim, num_layers=adaptive_layers)
        # Classification Head
        self.cls_head = nn.Linear(embed_dim, 2)

    def forward(self, img1, img2, sequence):
        # Patch Embedding for Images
        img1_embedded = self.patch_embedding(img1)
        img2_embedded = self.patch_embedding(img2)
        # TCN Embedding for Sequence
        seq_embedded = self.tcn_embedding(sequence)
        # Sequence Encoding
        seq_feat = self.seq_encoder(seq_embedded)
        # Image Encoding
        img_feat1 = self.image_encoder1(img1_embedded)
        img_feat2 = self.image_encoder2(img2_embedded)
        # Cross Attention Fusion1: img1(QK), img2(V)
        fusion1 = self.cross_attention1(img_feat1, img_feat1, img_feat2)
        # FeedForward for img_feat1
        img_feat1_transformed = self.feedforward(img_feat1)
        # Cross Attention Fusion2: img_feat1_transformed, fusion1, img_feat2
        fusion2 = self.cross_attention2(img_feat1_transformed, fusion1, img_feat2)
        # Gated Residual Network (Fusion2 + Sequence Features)
        grn_feat = self.grn(torch.cat([fusion2, seq_feat], dim=1))
        # Adaptive Module (Combining multiple features)
        adaptive_feat = self.adaptive_module(
            img_feat1_transformed,
            fusion2,
            torch.cat([grn_feat, img_feat2], dim=-2),
            img_feat2,
            seq_feat
        )
        # Classification
        output = self.cls_head(adaptive_feat.mean(dim=1))  # Global average pooling before classification
        return output


