import torch
from torch import nn
from torch.nn.functional import mse_loss
from torchvision import models
from torch.nn import MultiheadAttention
import logging
from typing import Tuple, Optional, List
import math

# Initialize logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class VisionEncoder(nn.Module):
    """
    Vision encoder using ResNet-18 as a feature extractor.
    Encodes visual inputs into a 256-dimensional latent space.
    
    Attributes:
        base_layers: Modified ResNet-18 features layers
        feature_projector: Linear projection layer
    """
    def __init__(self) -> None:
        """
        Initialize VisionEncoder with ResNet-18 architecture.
        """
        super().__init__()
        try:
            resnet = models.resnet18(pretrained=False)
            self.base_layers = nn.Sequential(*list(resnet.children())[:-1])
            self.feature_projector = nn.Linear(512, 256)
        except Exception as e:
            logger.error(f"VisionEncoder initialization failed: {e}", exc_info=True)
            raise

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for the vision encoder.
        
        Args:
            x: Input tensor of shape (batch, channels, height, width)
            
        Returns:
            Encoded features tensor (batch, 256)
        """
        try:
            if x.dim() != 4:
                raise ValueError(f"VisionEncoder input must be 4D tensor, got {x.dim()}")
            encoded = self.base_layers(x)
            encoded = encoded.squeeze().float()
            return self.feature_projector(encoded)
        except Exception as e:
            logger.error(f"VisionEncoder forward error: {e}", exc_info=True)
            raise

class TactileVQ(nn.Module):
    """
    Tactile Vector Quantization module that maps tactile features to 
    learned vector quantization codes. Uses embedding layer with uniform 
    initialization.
    
    Attributes:
        num_embeddings: Number of VQ codes
        embedding_dim: Dimension of each code vector
    """
    def __init__(self, num_embeddings: int = 128, embedding_dim: int = 128) -> None:
        """
        Args:
            num_embeddings: Number of code vectors in the dictionary
            embedding_dim: Dimension of each code vector
            
        Raises:
            ValueError: If either dimension is non-positive
        """
        super().__init__()
        if num_embeddings <= 0 or embedding_dim <= 0:
            raise ValueError("Embedding parameters must be positive integers")
        try:
            self.num_embeddings = num_embeddings
            self.embedding_dim = embedding_dim
            self.embedding = nn.Embedding(num_embeddings, embedding_dim)
            with torch.no_grad():
                self.embedding.weight.data.uniform_(-1.0 / num_embeddings, 1.0 / num_embeddings)
        except Exception as e:
            logger.error(f"TactileVQ initialization failed: {e}", exc_info=True)
            raise

    def forward(self, z_e: torch.Tensor, return_indices: bool = False) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        """
        Perform vector quantization on input features.
        
        Args:
            z_e: Input features tensor (batch, emb_dim)
            return_indices: Whether to return encoding indices
            
        Returns:
            Tuple of (quantized features, commitment loss, encoding indices)
        """
        try:
            if z_e.dim() != 2:
                raise ValueError(f"VQ expects 2D tensor, got {z_e.dim()}")
            
            batch_size, emb_dim = z_e.size()
            if emb_dim != self.embedding_dim:
                raise ValueError(f"Expected embedding dim {self.embedding_dim}, got {emb_dim}")
                
            z_e_flat = z_e.reshape(batch_size, self.embedding_dim)
            distances = torch.cdist(z_e_flat, self.embedding.weight, p=2)
            encoding_indices = torch.argmin(distances, dim=-1)
            encoding_matrix = torch.nn.functional.one_hot(encoding_indices, self.num_embeddings).to(torch.float32)
            
            z_q = self.embedding(encoding_indices)
            z_q = z_q.view(z_e.shape)
            
            commitment_loss = mse_loss(z_q, z_e.detach())
            
            if return_indices:
                return z_q, commitment_loss, encoding_indices
            return z_q, commitment_loss, None
        except Exception as e:
            logger.error(f"TactileVQ forward error: {e}", exc_info=True)
            raise

class TactileNetwork(nn.Module):
    """
    Tactile processing network with convolutional encoder, VQ layer, and 
    transposed convolutional decoder. Preserves input resolution in 
    reconstruction.
    """
    def __init__(self, embed_dim: int = 256, num_embeddings: int = 128) -> None:
        """
        Initialize tactile processing network components.
        
        Args:
            embed_dim: Dimension of hidden layers
            num_embeddings: Size of VQ codebook
            
        Raises:
            ValueError: If any dimension parameter is non-positive
        """
        super().__init__()
        if embed_dim <= 0 or num_embeddings <= 0:
            raise ValueError("Embedding parameters must be positive integers")
        try:
            self.encoder = nn.Sequential(
                nn.Conv2d(1, 32, 4, 2, 1),
                nn.ReLU(inplace=True),
                nn.Conv2d(32, 64, 4, 2, 1),
                nn.ReLU(inplace=True),
                nn.Conv2d(64, 128, 4, 2, 1),
                nn.ReLU(inplace=True),
                nn.AdaptiveAvgPool2d(1)
            )
            self.hid2lat = nn.Linear(128, embed_dim)
            self.vq = TactileVQ(num_embeddings=num_embeddings, embedding_dim=embed_dim)
            self.proj2hid = nn.Linear(embed_dim, 128)
            self.decoder = nn.Sequential(
                nn.ConvTranspose2d(128, 64, 4, 2, 1),
                nn.ReLU(inplace=True),
                nn.ConvTranspose2d(64, 32, 4, 2, 1),
                nn.ReLU(inplace=True),
                nn.ConvTranspose2d(32, 1, 4, 2, 1)
            )
            logger.info(f"TactileNetwork initialized - EmbedDim: {embed_dim}, NumEmbed: {num_embeddings}")
        except Exception as e:
            logger.error(f"TactileNetwork initialization failed: {e}", exc_info=True)
            raise

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Process tactile input through encoder and VQ module, then decode.
        
        Args:
            x: Tactile input tensor (batch, 1, height, width)
            
        Returns:
            Tuple of (reconstruction, encoded features, quantized features, VQ loss)
        """
        try:
            if x.dim() != 4:
                raise ValueError(f"TactileNetwork input must be 4D tensor, got {x.dim()}")
            
            features = self.encoder(x)
            if features.size(-1) != 1 or features.size(-2) != 1:
                raise ValueError(f"Expected spatial output size [1x1], got {features.shape}")
                
            z_e_encoded = self.hid2lat(features.squeeze())
            z_q, vq_loss, _ = self.vq(z_e_encoded, return_indices=True)
            latent = self.proj2hid(z_q)
            latent = latent.view(-1, 128, 1, 1)
            reconstruction = self.decoder(latent)
            
            return reconstruction, z_e_encoded, z_q, vq_loss
            
        except Exception as e:
            logger.error(f"TactileNetwork forward error: {e}", exc_info=True)
            raise

class CrossAttention(nn.Module):
    """
    Cross-attention module that computes attention between vision and 
    tactile features bidirectionally. Uses 4-head attention with dropout.
    
    Attributes:
        query_projector: Linear layer for queries
        key_projector: Linear layer for keys
        value_projector: Linear layer for values
    """
    def __init__(self, embed_dim: int) -> None:
        """
        Initialize cross-attention components.
        
        Args:
            embed_dim: Dimension of input features
            
        Raises:
            ValueError: If embed_dim is non-positive
        """
        super().__init__()
        if embed_dim <= 0:
            raise ValueError("Embedding dimension must be positive")
        try:
            self.query_projector = nn.Linear(embed_dim, embed_dim)
            self.key_projector = nn.Linear(embed_dim, embed_dim)
            self.value_projector = nn.Linear(embed_dim, embed_dim)
            self.attention = MultiheadAttention(embed_dim, 4, dropout=0.1)
            logger.info(f"CrossAttention initialized with dimension: {embed_dim}")
        except Exception as e:
            logger.error(f"CrossAttention initialization failed: {e}", exc_info=True)
            raise

    def forward(self, queries: torch.Tensor, keys: torch.Tensor) -> torch.Tensor:
        """
        Compute cross-attention between two feature sets.
        
        Args:
            queries: Query features (batch, emb_dim)
            keys: Key/value features (batch, emb_dim)
            
        Returns:
            Attention context features (batch, emb_dim)
        """
        try:
            if queries.dim() != 2 or keys.dim() != 2:
                raise ValueError(f"CrossAttention expects 2D tensors, got Q {queries.shape} and K {keys.shape}")
                
            batch_size, emb_dim = queries.shape
            if emb_dim != keys.shape[1]:
                raise ValueError(f"Query and key dimensions mismatch: {queries.shape[1]} vs {keys.shape[1]}")
                
            q = self.query_projector(queries).unsqueeze(0)  # [1, batch, emb_dim]
            k = self.key_projector(keys).unsqueeze(0)
            v = self.value_projector(keys).unsqueeze(0)
            
            context, _ = self.attention(q, k, v)
            return context.squeeze(0)  # [batch, emb_dim]
            
        except Exception as e:
            logger.error(f"CrossAttention forward error: {e}", exc_info=True)
            raise

class MultiGenGraspingModel(nn.Module):
    """
    Multimodal grasping policy network that fuses vision and tactile 
    inputs through cross-attention modules. Handles missing tactile input 
    by using learned representations.
    """
    def __init__(self, tactile_dim: int = 256, visual_dim: int = 256, action_dim: int = 4) -> None:
        """
        Initialize multimodal grasping policy.
        
        Args:
            tactile_dim: Tactile encoder output dimension
            visual_dim: Vision encoder output dimension
            action_dim: Dimension of output action space
            
        Raises:
            ValueError: If any dimension parameter is non-positive
        """
        super().__init__()
        if tactile_dim <= 0 or visual_dim <= 0 or action_dim < 3:
            raise ValueError("All dimensions must be positive (action_dim ≥ 3 required)")
        try:
            self.vision_enc = VisionEncoder()
            self.tactile_enc = TactileNetwork(embed_dim=tactile_dim, num_embeddings=128)
            
            self.vision2tactile = CrossAttention(visual_dim)
            self.tactile2vision = CrossAttention(tactile_dim)
            
            self.fusion_projector = nn.Linear(visual_dim*2, visual_dim)
            self.final_policy = nn.Sequential(
                nn.Linear(visual_dim, 128),
                nn.ReLU(inplace=True),
                nn.Linear(128, 64),
                nn.ReLU(inplace=True),
                nn.Linear(64, action_dim)
            )
            
            logger.info(f"MultiGenGraspingModel initialized - Tactile:{tactile_dim}, Visual:{visual_dim} Action:{action_dim}")
        except Exception as e:
            logger.error(f"Multimodal model initialization failed: {e}", exc_info=True)
            raise

    def forward(self, visual_input: torch.Tensor, tactile_input: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass for multimodal grasping policy.
        
        Args:
            visual_input: RGB image tensor (batch, 3, height, width)
            tactile_input: Tactile input tensor (batch, 1, height, width)
            
        Returns:
            Tuple of (predicted actions, tactile reconstruction, VQ loss)
        """
        try:
            if visual_input.dim() != 4:
                raise ValueError(f"Visual input must be 4D tensor, got {visual_input.dim()}")
            if tactile_input is not None and tactile_input.dim() != 4:
                raise ValueError(f"Tactile input must be 4D tensor, got {tactile_input.dim()}")
                
            visual_features = self.vision_enc(visual_input)
            
            if tactile_input is not None:
                tactile_recon, z_e, z_q, vq_loss = self.tactile_enc(tactile_input)
                vision_att = self.vision2tactile(visual_features, z_q)
                tactile_att = self.tactile2vision(z_e, visual_features)
            else:
                batch_size = visual_input.size(0)
                z_e = torch.rand(batch_size, self.tactile_enc.hid2lat.out_features, device=visual_features.device)
                z_q, vq_loss, _ = self.tactile_enc.vq(z_e, return_indices=True)
                vision_att = self.vision2tactile(visual_features, z_q)
                tactile_att = torch.zeros_like(visual_features)
                tactile_recon = torch.zeros(batch_size, 1, 64, 64, device=visual_input.device)
            
            context = torch.cat([visual_features, vision_att], dim=1)
            context = self.fusion_projector(context)
            actions = self.final_policy(context)
            
            return actions, tactile_recon, vq_loss
            
        except Exception as e:
            logger.error(f"Multimodal model forward error: {e}", exc_info=True)
            raise

class PolicyLoss(nn.Module):
    """
    Combined loss function for the grasping policy network. Calculates 
    action reconstruction loss and policy objective.
    """
    def __init__(self, action_weight: float = 1.0, commitment_weight: float = 0.25) -> None:
        """
        Initialize loss components and weighting parameters.
        
        Args:
            action_weight: Weight coefficient for action loss
            commitment_weight: Weight coefficient for VQ commitment loss
        """
        super().__init__()
        self.mse_criterion = nn.MSELoss(reduction='none')
        self.action_weight = action_weight
        self.commitment_weight = commitment_weight
        logger.info(f"PolicyLoss initialized with weights: action={action_weight}, commitment={commitment_weight}")

    def forward(self, pred_actions: torch.Tensor, target_actions: torch.Tensor, 
               tactile_recon: torch.Tensor, tactile_gt: torch.Tensor, 
               success_labels: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Calculate combined loss components.
        
        Args:
            pred_actions: Predicted action tensor (batch, action_dim)
            target_actions: Ground truth actions (batch, action_dim)
            tactile_recon: Tactile reconstruction (batch, 1, H, W)
            tactile_gt: Ground truth tactile input (batch, 1, H, W)
            success_labels: Binary success labels (batch, 1)
            
        Returns:
            Tuple of (total loss, action loss, commitment loss)
        """
        try:
            if pred_actions.dim() != 2 or target_actions.dim() != 2:
                raise ValueError("Action preds and targets must be 2D tensors")
            if tactile_recon.dim() != 4 or tactile_gt.dim() !=4:
                raise ValueError("Tactile tensors must be 4D")
            if success_labels.dim() != 2:
                raise ValueError("Success labels must be 2D tensor")
                
            action_loss = self.mse_criterion(pred_actions, target_actions).mean()
            tactile_loss = self.mse_criterion(tactile_recon, tactile_gt).mean() * 0.1
            policy_weights = torch.sigmoid(pred_actions).mean() * success_labels.float()
            policy_loss = (1. - policy_weights).mean()
            
            total_loss = self.action_weight * action_loss + policy_loss + self.commitment_weight * tactile_loss
            
            return total_loss, action_loss, tactile_loss
            
        except Exception as e:
            logger.error(f"PolicyLoss calculation failed: {e}", exc_info=True)
            raise

def initialize_model() -> Tuple[MultiGenGraspingModel, torch.Tensor]:
    """
    Initialize a MultiGenGraspingModel instance with proper 
    component configuration for deployment.
    
    Returns:
        Tuple of (model, normalized embedding weights)
        
    Raises:
        RuntimeError: If model components cannot be validated
    """
    try:
        model = MultiGenGraspingModel()
        
        # Validate model architecture
        if not isinstance(model.vision_enc, VisionEncoder):
            raise RuntimeError("Vision encoder validation failed")
        if not all(hasattr(model, comp) for comp in ['tactile_enc', 'vision2tactile', 'tactile2vision']):
            raise RuntimeError("Missing model components")
            
        # Calculate normalized embedding weights
        vq_weights = model.tactile_enc.vq.embedding.weight.data
        emb_norms = torch.nn.functional.normalize(vq_weights, p=2, dim=1)
        normalized_emb = torch.softmax(emb_norms.norm(p=2, dim=1), dim=0)
        
        logger.info("Model initialized successfully")
        return model, normalized_emb
        
    except Exception as e:
        logger.critical(f"Model initialization failed: {e}", exc_info=True)
        raise
