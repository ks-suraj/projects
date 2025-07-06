# Generated on 2025-07-06 07:58:37
# Model implementation below

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
import math

class VisionEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        resnet = models.resnet18(pretrained=False)
        self.features = nn.Sequential(*list(resnet.children())[:-1])
        self.feature_projector = nn.Linear(512, 256)
        
    def forward(self, x):
        encoded = self.features(x)
        encoded = encoded.squeeze(-1).squeeze(-1)
        encoded = self.feature_projector(encoded)
        return encoded

class TactileVQ(nn.Module):
    def __init__(self, num_embeddings=128, embedding_dim=128):
        super().__init__()
        self.embedding = nn.Embedding(num_embeddings, embedding_dim)
        self.embedding.weight.data.uniform_(-1.0 / num_embeddings, 1.0 / num_embeddings)
        
    def forward(self, z_e, return_indices=False):
        z_e_flat = z_e.reshape(-1, z_e.size(-1))
        distances = torch.cdist(z_e_flat, self.embedding.weight, p=2)
        encoding_indices = torch.argmin(distances, dim=-1)
        encodings = torch.zeros(encoding_indices.shape[0], self.num_embeddings, device=z_e.device)
        encodings.scatter_(1, encoding_indices.unsqueeze(1), 1)
        z_q = self.embedding(encoding_indices).reshape(z_e.shape)
        commitment_loss = F.mse_loss(z_q, z_e.detach())
        
        if return_indices:
            return z_q, commitment_loss, encoding_indices
        return z_q, commitment_loss

class TactileNetwork(nn.Module):
    def __init__(self, embed_dim=256, num_embeddings=128):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 32, 4, 2, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, 4, 2, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, 4, 2, 1),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1),
        )
        self.hid2lat = nn.Linear(128, embed_dim)
        self.vq = TactileVQ(num_embeddings=128, embedding_dim=embed_dim)
        self.proj2hid = nn.Linear(embed_dim, 128)
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(128, 64, 4, 2, 1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(64, 32, 4, 2, 1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(32, 1, 4, 2, 1)
        )
        
    def forward(self, x):
        z_e = self.encoder(x)
        z_e = self.hid2lat(z_e.squeeze(-1).squeeze(-1))
        z_q, commitment_loss = self.vq(z_e)
        z_d = self.proj2hid(z_q)
        z_d = z_d.view(-1, 128, 1, 1)
        x_recon = self.decoder(z_d)
        return x_recon, z_e, z_q, commitment_loss

class CrossAttention(nn.Module):
    def __init__(self, embed_dim):
        super().__init__()
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.attn = nn.MultiheadAttention(embed_dim, 4, dropout=0.1)
        
    def forward(self, x1, x2):
        queries = self.q_proj(x1).unsqueeze(0)
        keys = self.k_proj(x2).unsqueeze(0)
        values = self.v_proj(x2).unsqueeze(0)
        context, _ = self.attn(queries, keys, values)
        return context.squeeze(0)

class MultiGenGraspingModel(nn.Module):
    def __init__(self, tactile_dim=256, visual_dim=256, action_dim=4):
        super().__init__()
        self.vision_enc = VisionEncoder()
        self.tactile_enc = TactileNetwork(embed_dim=tactile_dim)
        self.vision2tactile_attn = CrossAttention(256)
        self.tactile2vision_attn = CrossAttention(256)
        self.fused_proj = nn.Linear(256*2, 256)
        self.policy = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, action_dim)
        )
        
    def forward(self, visual_input, tactile_input=None):
        v_features = self.vision_enc(visual_input)
        
        if tactile_input is not None:
            t_recon, z_e, z_q, vq_loss = self.tactile_enc(tactile_input)
        else:
            z_e = torch.rand(visual_input.size(0), 256, device=visual_input.device)
            z_q, vq_loss = self.vq(z_e)
            t_recon = torch.zeros((visual_input.size(0), 1, 64, 64), device=visual_input.device)
        
        vision_att = self.vision2tactile_attn(v_features, z_q)
        tactile_att = self.tactile2vision_attn(z_e, v_features)
        fused = torch.cat([v_features, vision_att], dim=1)
        fused = self.fused_proj(F.leaky_relu(fused))
        actions = self.policy(fused)
        return actions, t_recon, vq_loss

class PolicyLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.mse = nn.MSELoss()
        
    def forward(self, pred_actions, target_actions, success_labels):
        action_loss = self.mse(pred_actions, target_actions)
        policy_targets = success_labels.float()
        return action_loss, policy_targets
        
def initialize_model():
    model = MultiGenGraspingModel()
    tactile_vq_weights = model.tactile_enc.vq.embedding.weight.data
    return model, F.softmax(tactile_vq_weights.norm(p=2, dim=1), dim=0)