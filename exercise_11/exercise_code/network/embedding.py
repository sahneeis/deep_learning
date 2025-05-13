from torch import nn
import torch

def positional_encoding(d_model: int,
                        max_length: int) -> torch.Tensor:
    """
    Computes the positional encoding matrix
    Args:
        d_model: Dimension of Embedding
        max_length: Maximums sequence length

    Shape:
        - output: (max_length, d_model)
    """

    output = None
    
#     position = torch.arange(max_length).unsqueeze(1)
#     div_term = torch.exp(torch.arange(0, d_model, 2) * 
#                          (-torch.log(torch.Tensor([10000])) / d_model))
#     pos_encoding = torch.zeros(max_length, d_model)
#     pos_encoding[:, 0::2] = torch.sin(position * div_term)
#     pos_encoding[:, 1::2] = torch.cos(position * div_term)
#     output = pos_encoding


    exponent = torch.arange(0, d_model, 2, dtype=torch.float32) / d_model 
    pos = torch.arange(0, max_length, dtype=torch.float32).unsqueeze(1) 
    angle_freq = torch.exp(exponent * -torch.log(torch.tensor(10000.0, dtype=torch.float32)))
    pos_encoding = torch.zeros((max_length, d_model), dtype=torch.float32)
    pos_encoding[:, 0::2] = torch.sin(pos * angle_freq)  # 偶数索引
    pos_encoding[:, 1::2] = torch.cos(pos * angle_freq)  # 奇数索引
    output = pos_encoding
    ########################################################################
    # TODO:                                                                #
    #   Task 4: Initialize the positional encoding layer.                  #
    #                                                                      #
    # Hints 4:                                                             #
    #       - You can copy the implementation from the notebook, just      #
    #         make sure to use torch instead of numpy!                     #
    #       - Use torch.log(torch.Tensor([10000])), to make use of the     #
    #         torch implementation of the natural logarithm.               #
    #       - Implement the alternating sin and cos functions the way we   #
    #         did in the notebook.                                         #
    ########################################################################


    pass

    ########################################################################
    #                           END OF YOUR CODE                           #
    ########################################################################

    return output

class Embedding(nn.Module):

    def __init__(self,
                 vocab_size: int,
                 d_model: int,
                 max_length: int):
        """

        Args:
            vocab_size: Number of elements in the vocabulary
            d_model: Dimension of Embedding
            max_length: Maximum sequence length
        """
        super().__init__()

        self.embedding = None
        self.pos_encoding = None
        
        # 1. Embedding layer
        self.embedding = nn.Embedding(num_embeddings=vocab_size, 
                                      embedding_dim=d_model)
        
        # 4. Positional Encoding as a parameter (non-trainable)
        pe = positional_encoding(d_model, max_length)  
        self.pos_encoding = nn.Parameter(pe, requires_grad=False)

        ########################################################################
        # TODO:                                                                #
        #   Task 1: Initialize the embedding layer (torch.nn implementation)   #
        #   Task 4: Initialize the positional encoding layer.                  #
        #                                                                      #
        # Hints 1:                                                             #
        #       - Have a look at pytorch embedding module                      #
        # Hints 4:                                                             #
        #       - Initialize it using d_model and max_length                   #
        ########################################################################


    def forward(self,
                inputs: torch.Tensor) -> torch.Tensor:
        """
        The forward function takes in tensors of token ids and transforms them into vector embeddings. 
        It then adds the positional encoding to the embeddings, and if configured, performs dropout on the layer!

        Args:
            inputs: Batched Sequence of Token Ids

        Shape:
            - inputs: (batch_size, sequence_length)
            - outputs: (batch_size, sequence_length, d_model)
        """

        outputs = None
        
        # 1. Convert token IDs --> embeddings
        embeddings = self.embedding(inputs)

        # 2. Slice correct number of positions from pos_encoding
        sequence_length = inputs.size(1)
        pos_encoding_slice = self.pos_encoding[:sequence_length, :]
        outputs = embeddings + pos_encoding_slice.unsqueeze(0)

        ########################################################################
        # TODO:                                                                #
        #   Task 1: Compute the outputs of the embedding layer                 #
        #   Task 4: Add the positional encoding to the output                  #
        #                                                                      #
        # Hint 4: We have already extracted them for you, all you have to do   #
        #         is add them to the embeddings!                               #
        ########################################################################


        pass

        ########################################################################
        #                           END OF YOUR CODE                           #
        ########################################################################

        return outputs