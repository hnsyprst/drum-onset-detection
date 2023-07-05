import torch
import torch.nn as nn
from torchshape import tensorshape

class Stream(nn.Module):
    """
    An implementation of streaming-aware neural networks, as proposed by
    https://arxiv.org/pdf/2005.06720.pdf

    :param layer: (nn.Module) Placeholder
    :param frame_shape: (tuple) Placeholder
    :param state_shape: (tuple) Placeholder
    """
    def __init__(self, layer: nn.Module, frame_shape: tuple, state_shape: tuple):
        super().__init__()

        self.layer = layer

        # We'll hold a state of shape state_shape, and roll samples of shape frame_shape onto the end each iteration
        self.frame_len = frame_shape
        self.state_len = state_shape

        # To start with, the input_state will contain zeroes
        self.input_state = torch.zeros(tensorshape(layer, self.state_len))

    def forward(self, input):
        # Create the updated_state from the input_state
        # Slide the window backwards
        updated_state = torch.roll(self.input_state, self.frame_len)
        # Add the new information
        updated_state[self.frame_len:] = input

        # Pass self.layer the updated_state, creating the output_state
        output_state = self.layer(updated_state)

        # Update the input_state with the new output_state,
        # ready for the next iteration
        self.input_state = output_state

        # Return the output_state
        return output_state