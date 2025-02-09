
# Brain-to-Text Decoding:
A Non-invasive Approach via Typing

[![Join our Discord](https://img.shields.io/badge/Discord-Join%20our%20server-5865F2?style=for-the-badge&logo=discord&logoColor=white)](https://discord.gg/swarms) [![Subscribe on YouTube](https://img.shields.io/badge/YouTube-Subscribe-red?style=for-the-badge&logo=youtube&logoColor=white)](https://www.youtube.com/@kyegomez3242) [![Connect on LinkedIn](https://img.shields.io/badge/LinkedIn-Connect-blue?style=for-the-badge&logo=linkedin&logoColor=white)](https://www.linkedin.com/in/kye-g-38759a207/) [![Follow on X.com](https://img.shields.io/badge/X.com-Follow-1DA1F2?style=for-the-badge&logo=x&logoColor=white)](https://x.com/kyegomezb)


[![GitHub stars](https://img.shields.io/github/stars/The-Swarm-Corporation/Legal-Swarm-Template?style=social)](https://github.com/The-Swarm-Corporation/Legal-Swarm-Template)
[![Swarms Framework](https://img.shields.io/badge/Built%20with-Swarms-blue)](https://github.com/kyegomez/swarms)


[PAPER LINK](https://ai.meta.com/research/publications/brain-to-text-decoding-a-non-invasive-approach-via-typing/)

Abstract:
Modern neuroprostheses can now restore communication in patients who have lost the ability to speak or move. However, these invasive devices entail risks inherent to neurosurgery. Here, we introduce a non-invasive method to decode the production of sentences from brain activity and demonstrate its efficacy in a cohort of 35 healthy volunteers. For this, we present Brain2Qwerty, a new deep learning architecture trained to decode sentences from either electro- (EEG) or magneto-encephalography (MEG), while participants typed briefly memorized sentences on a QWERTY keyboard. With MEG, Brain2Qwerty reaches, on average, a character-error-rate (CER) of 32% and substantially outperforms EEG (CER: 67%). For the best participants, the model achieves a CER of 19%, and can perfectly decode a variety of sentences outside of the training set. While error analyses suggest that decoding depends on motor processes, the analysis of typographical errors suggests that it also involves higherlevel cognitive factors. Overall, these results narrow the gap between invasive and non-invasive methods and thus open the path for developing safe brain-computer interfaces for non-communicating patients.

## Code

```python

"""
Brain2Qwerty: A non-invasive brain-to-text decoding model.
"""

import os
from typing import Optional, List, Dict

import torch
import torch.nn as nn
import torch.nn.functional as F
from loguru import logger


class ModelConfigError(Exception):
    """Raised when model configuration parameters are invalid."""

    pass


class DataError(Exception):
    """Raised when input data doesn't meet requirements."""

    pass


class ConvBlock(nn.Module):
    """Convolutional block with skip connections, dropout and GELU activation."""

    def __init__(
        self,
        channels: int,
        sequence_length: int,
        kernel_size: int = 3,
        dilation: int = 3,
        dropout: float = 0.3,
    ):
        super().__init__()
        try:
            self.conv1 = nn.Conv1d(
                channels,
                channels,
                kernel_size,
                padding="same",
                dilation=dilation,
            )
            self.conv2 = nn.Conv1d(
                channels,
                channels,
                kernel_size,
                padding="same",
                dilation=dilation,
            )
            self.dropout = nn.Dropout(dropout)
            # Change normalization to operate on last dimension
            self.norm1 = nn.LayerNorm(sequence_length)
            self.norm2 = nn.LayerNorm(sequence_length)
        except Exception as e:
            logger.error(f"Failed to initialize ConvBlock: {str(e)}")
            raise ModelConfigError(
                f"ConvBlock initialization failed: {str(e)}"
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through convolutional block.

        Args:
            x: Input tensor of shape (batch_size, channels, sequence_length)
        Returns:
            Output tensor of same shape as input
        """
        try:
            residual = x
            # Normalize over the sequence dimension
            x = x.transpose(1, 2)  # [batch, seq_len, channels]
            x = self.norm1(x)
            x = x.transpose(1, 2)  # [batch, channels, seq_len]

            x = F.gelu(self.conv1(x))
            x = self.dropout(x)

            x = x.transpose(1, 2)  # [batch, seq_len, channels]
            x = self.norm2(x)
            x = x.transpose(1, 2)  # [batch, channels, seq_len]

            x = F.gelu(self.conv2(x))
            x = self.dropout(x)
            return x + residual
        except Exception as e:
            logger.error(f"Error in ConvBlock forward pass: {str(e)}")
            raise


class SpatialAttention(nn.Module):
    """Spatial attention mechanism for sensor positions."""

    def __init__(self, num_sensors: int, embedding_dim: int):
        super().__init__()
        try:
            self.position_embedding = nn.Parameter(
                torch.randn(num_sensors, embedding_dim)
            )
            self.input_projection = nn.Linear(1, embedding_dim)
            self.attention = nn.MultiheadAttention(
                embedding_dim, num_heads=1, batch_first=True
            )
            logger.info(
                f"Initialized SpatialAttention with {num_sensors} sensors and {embedding_dim} dimensions"
            )
        except Exception as e:
            logger.error(
                f"Failed to initialize SpatialAttention: {str(e)}"
            )
            raise ModelConfigError(
                f"SpatialAttention initialization failed: {str(e)}"
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply spatial attention to input tensor.

        Args:
            x: Input tensor of shape (batch_size, num_sensors, num_timepoints)
        """
        try:
            batch_size, num_sensors, num_timepoints = x.shape

            # Project each sensor reading to embedding dimension
            x = x.unsqueeze(-1)  # [batch, sensors, time, 1]
            x = self.input_projection(
                x
            )  # [batch, sensors, time, embed_dim]

            # Add positional embeddings
            x = x + self.position_embedding.unsqueeze(0).unsqueeze(2)

            # Reshape for attention
            x = x.view(batch_size * num_timepoints, num_sensors, -1)

            # Apply self-attention
            x, _ = self.attention(x, x, x)

            # Reshape back
            x = x.view(batch_size, num_timepoints, num_sensors, -1)
            x = x.mean(dim=-1)  # [batch, time, sensors]
            x = x.transpose(1, 2)  # [batch, sensors, time]

            return x

        except Exception as e:
            logger.error(
                f"Error in SpatialAttention forward pass: {str(e)}"
            )
            raise


class ConvolutionalModule(nn.Module):
    """Convolutional module for processing MEG/EEG signals."""

    def __init__(
        self,
        num_sensors: int,
        num_timepoints: int,
        num_subjects: int,
        hidden_dim: int = 256,
    ):
        super().__init__()
        try:
            self.spatial_attention = SpatialAttention(
                num_sensors, hidden_dim
            )

            # Projection to hidden dimension
            self.input_projection = nn.Linear(
                num_timepoints, hidden_dim
            )

            # Subject-specific layers
            self.subject_layers = nn.ModuleList(
                [
                    nn.Linear(hidden_dim, hidden_dim)
                    for _ in range(num_subjects)
                ]
            )

            # Store sequence length for conv blocks
            self.sequence_length = hidden_dim

            # Convolutional blocks
            self.conv_blocks = nn.ModuleList(
                [
                    ConvBlock(hidden_dim, sequence_length=hidden_dim)
                    for _ in range(8)
                ]
            )

            # Temporal attention
            self.temporal_attention = nn.MultiheadAttention(
                hidden_dim, num_heads=1, batch_first=True
            )

            logger.info(
                f"Initialized ConvolutionalModule with {hidden_dim} hidden dimensions"
            )
        except Exception as e:
            logger.error(
                f"Failed to initialize ConvolutionalModule: {str(e)}"
            )
            raise ModelConfigError(
                f"ConvolutionalModule initialization failed: {str(e)}"
            )

    def forward(
        self, x: torch.Tensor, subject_ids: torch.Tensor
    ) -> torch.Tensor:
        """
        Forward pass through convolutional module.

        Args:
            x: Input tensor of shape (batch_size, num_sensors, num_timepoints)
            subject_ids: Tensor of subject IDs
        """
        try:
            batch_size = x.size(0)

            # Apply spatial attention
            x = self.spatial_attention(x)  # [batch, sensors, time]

            # Project to hidden dimension
            x = self.input_projection(x)  # [batch, sensors, hidden]

            # Apply subject-specific layers
            outputs = []
            for i, subject_id in enumerate(subject_ids):
                subject_output = self.subject_layers[subject_id](x[i])
                outputs.append(subject_output)
            x = torch.stack(outputs)  # [batch, sensors, hidden]

            # Transpose for conv blocks
            x = x.transpose(1, 2)  # [batch, hidden, sensors]

            # Apply convolutional blocks
            for conv_block in self.conv_blocks:
                x = conv_block(x)

            # Apply temporal attention
            x = x.transpose(1, 2)  # [batch, sensors, hidden]
            x, _ = self.temporal_attention(x, x, x)

            return x.mean(
                dim=1
            )  # Average pooling over sensors -> [batch, hidden]

        except Exception as e:
            logger.error(
                f"Error in ConvolutionalModule forward pass: {str(e)}"
            )
            raise


class TransformerModule(nn.Module):
    """Transformer module for sentence-level context."""

    def __init__(
        self,
        hidden_dim: int = 256,
        num_layers: int = 4,
        num_heads: int = 2,
        num_classes: int = 29,
    ):
        super().__init__()
        try:
            encoder_layer = nn.TransformerEncoderLayer(
                d_model=hidden_dim, nhead=num_heads, batch_first=True
            )
            self.transformer = nn.TransformerEncoder(
                encoder_layer, num_layers
            )
            self.output_projection = nn.Linear(
                hidden_dim, num_classes
            )
            logger.info(
                f"Initialized TransformerModule with {num_layers} layers"
            )
        except Exception as e:
            logger.error(
                f"Failed to initialize TransformerModule: {str(e)}"
            )
            raise ModelConfigError(
                f"TransformerModule initialization failed: {str(e)}"
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        try:
            x = self.transformer(x)
            return self.output_projection(x)
        except Exception as e:
            logger.error(
                f"Error in TransformerModule forward pass: {str(e)}"
            )
            raise


class Brain2Qwerty(nn.Module):
    """Complete Brain2Qwerty model."""

    def __init__(
        self,
        num_sensors: int,
        num_timepoints: int,
        num_subjects: int,
        hidden_dim: int = 256,
        num_classes: int = 29,
        lm_weight: float = 5.0,
        lm_path: Optional[str] = None,
    ):
        super().__init__()
        try:
            self.conv_module = ConvolutionalModule(
                num_sensors=num_sensors,
                num_timepoints=num_timepoints,
                num_subjects=num_subjects,
                hidden_dim=hidden_dim,
            )

            self.transformer_module = TransformerModule(
                hidden_dim=hidden_dim, num_classes=num_classes
            )

            self.lm_weight = lm_weight
            if lm_path and os.path.exists(lm_path):
                import kenlm

                self.language_model = kenlm.Model(lm_path)
            else:
                logger.warning(
                    "No language model provided or file not found"
                )
                self.language_model = None

            logger.info("Successfully initialized Brain2Qwerty model")
        except Exception as e:
            logger.error(
                f"Failed to initialize Brain2Qwerty: {str(e)}"
            )
            raise ModelConfigError(
                f"Brain2Qwerty initialization failed: {str(e)}"
            )

    def forward(
        self,
        x: torch.Tensor,
        subject_ids: torch.Tensor,
        prev_chars: Optional[List[str]] = None,
    ) -> torch.Tensor:
        """
        Forward pass through complete model.

        Args:
            x: Input tensor of shape (batch_size, num_sensors, num_timepoints)
            subject_ids: Tensor of subject IDs
            prev_chars: Optional list of previous characters for language model
        """
        try:
            # Input validation
            if x.dim() != 3:
                raise DataError(
                    f"Expected 3D input tensor, got shape {x.shape}"
                )
            if len(subject_ids) != x.size(0):
                raise DataError(
                    "Number of subject IDs must match batch size"
                )

            logger.debug(
                f"Input shape: {x.shape}, Subject IDs: {subject_ids.shape}"
            )

            # Process through convolutional module
            conv_out = self.conv_module(
                x, subject_ids
            )  # [batch, hidden]

            # Process through transformer module
            transformer_out = self.transformer_module(
                conv_out.unsqueeze(1)
            )  # Add sequence dim

            return transformer_out

        except Exception as e:
            logger.error(
                f"Error in Brain2Qwerty forward pass: {str(e)}"
            )
            raise


def create_model(config: Dict) -> Brain2Qwerty:
    """Create Brain2Qwerty model from configuration dictionary."""
    try:
        required_params = [
            "num_sensors",
            "num_timepoints",
            "num_subjects",
        ]
        for param in required_params:
            if param not in config:
                raise ModelConfigError(
                    f"Missing required parameter: {param}"
                )

        model = Brain2Qwerty(
            num_sensors=config["num_sensors"],
            num_timepoints=config["num_timepoints"],
            num_subjects=config["num_subjects"],
            hidden_dim=config.get("hidden_dim", 256),
            num_classes=config.get("num_classes", 29),
            lm_weight=config.get("lm_weight", 5.0),
            lm_path=config.get("lm_path", None),
        )

        logger.info("Successfully created Brain2Qwerty model")
        return model

    except Exception as e:
        logger.error(f"Failed to create model: {str(e)}")
        raise


if __name__ == "__main__":
    # Configure logger
    logger.add("brain2qwerty.log", rotation="500 MB")

    try:
        # Create test input
        batch_size = 1
        num_sensors = 100
        num_timepoints = 100
        num_subjects = 100

        x = torch.randn(batch_size, num_sensors, num_timepoints)
        subject_ids = torch.randint(0, num_subjects, (batch_size,))

        # Create model
        config = {
            "num_sensors": num_sensors,
            "num_timepoints": num_timepoints,
            "num_subjects": num_subjects,
            "hidden_dim": 256,
        }

        model = create_model(config)

        # Forward pass
        output = model(x, subject_ids)
        logger.info(f"Output shape: {output.shape}")

    except Exception as e:
        logger.error(f"Error in example usage: {str(e)}")
        raise

```
