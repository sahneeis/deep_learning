from torch import nn
import torch
from ..network import MultiHeadAttention
from ..network import FeedForwardNeuralNetwork

class EncoderBlock(nn.Module):

    def __init__(self,
                 d_model: int,
                 d_k: int,
                 d_v: int,
                 n_heads: int,
                 d_ff: int,
                 dropout: float = 0.0):
        """

        Args:
            d_model: Dimension of Embedding
            d_k: Dimension of Keys and Queries
            d_v: Dimension of Values
            n_heads: Number of Attention Heads
            d_ff: Dimension of hidden layer
            dropout: Dropout probability
        """
        super().__init__()

        self.multi_head = None
        self.layer_norm1 = None
        self.ffn = None
        self.layer_norm2 = None

        ########################################################################
        # TODO:                                                                #
        #   Task 4: Initialize the Encoder Block                               #
        #           You will need:                                             #
        #                           - Multi-Head Self-Attention layer          #
        #                           - Layer Normalization                      #
        #                           - Feed forward neural network layer        #
        #                           - Layer Normalization                      #
        #                                                                      #
        # Hint 4: Check out the pytorch layer norm module                      #
        ########################################################################

        self.multi_head = MultiHeadAttention(d_model, d_k, d_v, n_heads, dropout)
        self.layer_norm1 = nn.LayerNorm(d_model)
        self.ffn = FeedForwardNeuralNetwork(d_model, d_ff, dropout)
        self.layer_norm2 = nn.LayerNorm(d_model)

        pass

        ########################################################################
        #                           END OF YOUR CODE                           #
        ########################################################################

    def forward(self,
                inputs: torch.Tensor,
                pad_mask: torch.Tensor = None) -> torch.Tensor:
        """

        Args:
            inputs: Inputs to the Encoder Block
            pad_mask: Optional Padding Mask

        Shape:
            - inputs: (batch_size, sequence_length, d_model)
            - pad_mask: (batch_size, sequence_length, sequence_length)
            - outputs: (batch_size, sequence_length, d_model)
        """
        outputs = None

        ########################################################################
        # TODO:                                                                #
        #   Task 4: Implement the forward pass of the encoder block            #
        #   Task 10: Pass on the padding mask                                  #
        #                                                                      #
        # Hint 4: Don't forget the residual connection! You can forget about   #
        #         the pad_mask for now!                                        #
        ########################################################################
        # Multi-head attention with residual connection and layer norm
        attention_output = self.multi_head(inputs, inputs, inputs, pad_mask)
        outputs = self.layer_norm1(inputs + attention_output)

        # Feed-forward network with residual connection and layer norm
        ffn_output = self.ffn(outputs)
        outputs = self.layer_norm2(outputs + ffn_output)

        pass

        ########################################################################
        #                           END OF YOUR CODE                           #
        ########################################################################

        return outputs