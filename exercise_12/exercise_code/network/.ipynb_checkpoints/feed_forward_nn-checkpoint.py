from torch import nn
import torch

class FeedForwardNeuralNetwork(nn.Module):

    def __init__(self,
                 d_model: int,
                 d_ff: int,
                 dropout: float = 0.0):
        """

        Args:
            d_model: Dimension of Embedding
            d_ff: Dimension of hidden layer
            dropout: Dropout probability
        """
        super().__init__()

        self.linear_1 = None
        self.relu = None
        self.linear_2 = None
        self.dropout = None

        ########################################################################
        # TODO:                                                                #
        #   Task 3: Initialize the feed forward network                        #
        #   Task 11: Initialize the dropout layer (torch.nn implementation)    #
        #                                                                      #
        ########################################################################
        self.linear_1 = nn.Linear(d_model, d_ff)
        self.relu = nn.ReLU()
        self.linear_2 = nn.Linear(d_ff, d_model)

        self.dropout = nn.Dropout(p=dropout)
        ########################################################################
        #                           END OF YOUR CODE                           #
        ########################################################################

    def forward(self,
                inputs: torch.Tensor) -> torch.Tensor:
        """

        Args:
            inputs: Inputs to the Feed Forward Network

        Shape:
            - inputs: (batch_size, sequence_length_queries, d_model)
            - outputs: (batch_size, sequence_length_queries, d_model)
        """
        outputs = None

        ########################################################################
        # TODO:                                                                #
        #   Task 3: Implement forward pass of feed forward layer               #
        #   Task 11: Pass the output through a dropout layer as a final step   #
        #                                                                      #
        ########################################################################
        outputs = self.linear_1(inputs)
        outputs = self.relu(outputs)
        outputs = self.linear_2(outputs)

        outputs = self.dropout(outputs)

        ########################################################################
        #                           END OF YOUR CODE                           #
        ########################################################################

        return outputs