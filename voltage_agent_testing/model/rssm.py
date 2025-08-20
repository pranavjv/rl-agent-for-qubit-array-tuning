import torch
import torch.nn as nn
from typing import Tuple

class RecurrentLSTMModel(nn.Module):
    """
    Recurrent model LSTM class for storing recurrent states in an RNNState compatible syntax
    shape = (batch, layers, hidden) for both hidden vectors
    """
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        num_layers: int,
    ) -> None:
        super().__init__()
        assert num_layers == 1, "Only 1 layer in LSTM supported for now (due to reshaping tensors)"
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=False) # we pass in batch as the second dim

    def forward(self, recurrent_states: Tuple[torch.Tensor, torch.Tensor], latent: torch.Tensor) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        if isinstance(recurrent_states, torch.Tensor):
             raise ValueError("Expected recurrent_states to be a tuple (h, c), got a tensor instead.")

        if latent.ndim < 3:
            latent = latent.unsqueeze(1)

        out = self.lstm(latent, recurrent_states)
        # _, hidden = out
        # assert isinstance(hidden, tuple), f"Expected output to be (h, c) tuple, got {type(hidden)}"
        return out


class RecurrentModel(nn.Module):
    """
    Recurrent model class for generating a persistent recurrent state embedding
    note the recurrent embedding takes in the new state latent vector
    recurrent size = (batch, layers, hidden)
    """
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        num_layers: int,
    ) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        # For first layer, input is input_dim + hidden_dim; for others, hidden_dim + hidden_dim
        self.linears = nn.ModuleList()
        self.lns = nn.ModuleList()
        for i in range(num_layers):
            in_dim = input_dim + hidden_dim if i == 0 else hidden_dim + hidden_dim
            self.linears.append(nn.Linear(in_dim, 3 * hidden_dim))
            self.lns.append(nn.LayerNorm(3 * hidden_dim))

    def forward(self, recurrent_state: torch.Tensor, latent: torch.Tensor) -> torch.Tensor:
        """
        Takes in the previous recurrent state and the latent vector and outputs the new recurrent state.
        recurrent_state: shape (..., num_layers, hidden_dim) or (..., hidden_dim) if num_layers==1
        latent: shape (..., input_dim)
        Returns: new_recurrent_state of same shape as recurrent_state
        """

        if recurrent_state.dim() == latent.dim():
            # Add layer dimension if missing (num_layers==1)
            recurrent_state = recurrent_state.unsqueeze(-2)
        x = latent
        new_states = []
        for i in range(self.num_layers):
            h = recurrent_state[..., i, :]
            # For first layer, concat latent; for others, concat previous layer's output
            inp = torch.cat((h, x), dim=-1)
            out = self.linears[i](inp)
            out = self.lns[i](out)
            reset, candidate, update = torch.chunk(out, 3, dim=-1)
            reset = torch.sigmoid(reset)
            candidate = torch.tanh(candidate * reset)
            update = torch.sigmoid(update - 1)
            new_h = update * candidate + (1 - update) * h
            new_states.append(new_h)
            x = new_h  # output of this layer is input to next
        new_recurrent = torch.stack(new_states, dim=-2)
        if self.num_layers == 1:
            new_recurrent = new_recurrent.squeeze(-2)
        return new_recurrent
