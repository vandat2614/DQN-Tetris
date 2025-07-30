import torch
import torch.nn as nn
from typing import List, Type, Optional

ACTIVATION_MAP = {
    'ReLU': nn.ReLU,
    'Tanh': nn.Tanh,
    'LeakyReLU': nn.LeakyReLU,
    'Sigmoid': nn.Sigmoid,
    'Softmax': lambda: nn.Softmax(dim=-1),
    'Identity': nn.Identity,
}

class NeuralNetwork(nn.Module):
    def __init__(
        self,
        input_size: int,
        hidden_sizes: List[int],
        hidden_activations: List[Type[nn.Module]],
        output_size: int,
        output_activation: Optional[Type[nn.Module]] = None,
        init_method: str = "xavier_uniform"  
    ):
        super(NeuralNetwork, self).__init__()

        layers = []
        self.linear_layers = []  

        for i in range(len(hidden_sizes)):
            in_features = input_size if i == 0 else hidden_sizes[i - 1]
            out_features = hidden_sizes[i]
            linear = nn.Linear(in_features, out_features)
            layers.extend([linear, hidden_activations[i]()])
            self.linear_layers.append(linear)

        output_layer = nn.Linear(hidden_sizes[-1], output_size)
        layers.append(output_layer)
        self.linear_layers.append(output_layer)
        
        if output_activation is not None:
            layers.append(output_activation())

        self.network = nn.Sequential(*layers)
        self._initialize_weights(init_method)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)
    
    def _initialize_weights(self, method: str):
        for layer in self.linear_layers:
            if method == "xavier_uniform":
                nn.init.xavier_uniform_(layer.weight)
            elif method == "xavier_normal":
                nn.init.xavier_normal_(layer.weight)
            elif method == "kaiming_uniform":
                nn.init.kaiming_uniform_(layer.weight, nonlinearity='relu')
            elif method == "kaiming_normal":
                nn.init.kaiming_normal_(layer.weight, nonlinearity='relu')
            elif method == "orthogonal":
                nn.init.orthogonal_(layer.weight)
            elif method == "normal":
                nn.init.normal_(layer.weight, mean=0.0, std=0.02)
            elif method == "uniform":
                nn.init.uniform_(layer.weight, a=-0.1, b=0.1)
            else:
                raise ValueError(f"Unsupported init method: {method}")
            
            if layer.bias is not None:
                nn.init.constant_(layer.bias, 0)

    @classmethod
    def from_config(cls, config: dict, input_size: int, output_size: int):
        hidden_activations = [ACTIVATION_MAP[act] for act in config['hidden_activations']]
        output_activation = ACTIVATION_MAP.get(config['output_activation'], None)

        return cls(
            input_size=input_size,
            hidden_sizes=config['hidden_sizes'],
            output_size=output_size,
            hidden_activations=hidden_activations,
            output_activation=output_activation,
            init_method=config.get("init_method", "xavier_uniform")
        )
    
    def load(self, path: str):
        device = next(self.parameters()).device
        state_dict = torch.load(path, map_location=device)
        self.load_state_dict(state_dict)