import torch
import torch.nn as nn


class RecurrentModel(nn.Module):
    """
    Recurrent model class for generating a persistent recurrent state embedding
    note the recurrent embedding takes in the new state latent vector
    """
    def __init__(
        self,
        input_size: int,
        hidden_size: int,
    ) -> None:
        super().__init__()
        
        self.linear = nn.Linear(input_size + hidden_size, 3 * hidden_size)
        self.ln = nn.LayerNorm(3 * hidden_size)

    def forward(self, recurrent_state: torch.Tensor, latent: torch.Tensor) -> torch.Tensor:
        """
        Takes in the previous recurrent state and the latent vector and outputs the new recurrent state
        """
        x = torch.cat((recurrent_state, latent), dim=-1)
        x = self.linear(x)
        x = self.ln(x)
        reset, candidate, update = torch.chunk(x, 3, dim=-1)
        reset = torch.sigmoid(reset) # how much of the previous state to forget
        candidate = torch.tanh(candidate * reset) # new candidate state
        update = torch.sigmoid(update - 1) # -1 for bias towards keeping memory
        new_recurrent = update * candidate + (1 - update) * recurrent_state
        return new_recurrent
