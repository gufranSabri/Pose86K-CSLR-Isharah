import torch
import torch.nn as nn
import numpy as np

class PositionalEncoding1D(nn.Module):
    def __init__(self, channels):
        super(PositionalEncoding1D, self).__init__()
        self.channels = int(np.ceil(channels / 2) * 2)  # Ensure even channels
        inv_freq = 1.0 / (10000 ** (torch.arange(0, self.channels, 2).float() / self.channels))
        self.register_buffer("inv_freq", inv_freq)

    def forward(self, tensor):
        batch_size, seq_len, _ = tensor.shape
        pos_x = torch.arange(seq_len, device=tensor.device).type(self.inv_freq.type())
        sin_inp_x = torch.einsum("i,j->ij", pos_x, self.inv_freq)
        emb_x = torch.stack((sin_inp_x.sin(), sin_inp_x.cos()), dim=-1).flatten(-2, -1)

        emb = torch.zeros((seq_len, self.channels), device=tensor.device).type(tensor.type())
        emb[:, : self.channels] = emb_x

        return emb[None, :, :tensor.shape[-1]].repeat(batch_size, 1, 1)

class TransformerEncoder(nn.TransformerEncoderLayer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.attention_weights = None

    def forward(self, src, src_mask=None, src_key_padding_mask=None):
        src2, attn_weights = self.self_attn(src, src, src, attn_mask=src_mask,
                                            key_padding_mask=src_key_padding_mask,
                                            need_weights=True, average_attn_weights=False)
        
        self.attention_weights = attn_weights.detach()  # save attention weights for visualization
        src = src + self.dropout1(src2)
        src = self.norm1(src)

        src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src = src + self.dropout2(src2)
        src = self.norm2(src)

        return src

class CSLRTransformer(nn.Module):
    def __init__(self, num_classes, input_dim=42, d_model=512, nhead=8, num_layers=3, dropout=0.1):
        super(CSLRTransformer, self).__init__()
        self.d_model = d_model
        self.pose_embed = nn.Linear(input_dim*2, d_model)
        self.positional_encoder = PositionalEncoding1D(d_model)

        self.transformer_encoder = nn.TransformerEncoder(
            TransformerEncoder(d_model=d_model, nhead=nhead, dropout=dropout, batch_first=True),
            num_layers=num_layers
        )
        
        self.temporal_pool_1 = nn.AvgPool1d(kernel_size=2, stride=2)  
        self.temporal_pool_2 = nn.AvgPool1d(kernel_size=2, stride=2)
        self.tcn_1 = nn.Conv1d(in_channels=d_model, out_channels=512, kernel_size=5, padding=1)
        self.tcn_2 = nn.Conv1d(in_channels=512, out_channels=d_model, kernel_size=5, padding=1)

        self.fc = nn.Sequential(
            nn.Linear(d_model, 128),
            nn.Dropout(0.2),
            nn.Linear(128, num_classes)
        )

    def forward(self, poses):
        poses = poses.view(poses.shape[0], poses.shape[1], -1)  # Flatten last two dims new shape (B, T, D)
        poses = self.pose_embed(poses)

        pos_embd = self.positional_encoder(poses)
        src = poses + pos_embd.to(poses.device)

        encoder_output = self.transformer_encoder(src)

        encoder_output = self.temporal_pool_1(encoder_output.transpose(1, 2)).transpose(1, 2)
        encoder_output = self.tcn_1(encoder_output.transpose(1, 2)).transpose(1, 2)  # (B, T/2, TCN channels)
        encoder_output = self.temporal_pool_2(encoder_output.transpose(1, 2)).transpose(1, 2)
        encoder_output = self.tcn_2(encoder_output.transpose(1, 2)).transpose(1, 2)  # (B, T/4, d_model)

        logits = self.fc(encoder_output)
        return logits

