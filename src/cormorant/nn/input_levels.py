import torch
import torch.nn as nn

from cormorant.nn.generic_levels import BasicMLP

############# Input to network #############

class InputLinear(nn.Module):
    def __init__(self, num_in, num_out, bias=True, device=torch.device('cpu'), dtype=torch.float):
        super(InputLinear, self).__init__()

        self.num_in = num_in
        self.num_out = num_out

        self.lin = nn.Linear(num_in, 2*num_out, bias=bias)
        self.lin.to(device=device, dtype=dtype)

        self.zero = torch.tensor(0, dtype=dtype, device=device)

    def forward(self, input_scalars, ignore, atom_mask):
        atom_mask = atom_mask.unsqueeze(-1)

        out = torch.where(atom_mask, self.lin(input_scalars), self.zero)
        out = out.view(input_scalars.shape[0:2] + (self.num_out, 1, 2))

        return out

class InputMPNN(nn.Module):
    def __init__(self, num_in, num_out, num_layers=1, soft_cut_rad=None, soft_cut_width=None, hard_cut_rad=None, num_mlp=-1, num_hidden=1, layer_width=256, activation='leakyrelu', device=torch.device('cpu'), dtype=torch.float):
        super(InputMPNN, self).__init__()

        self.soft_cut_rad = soft_cut_rad
        self.soft_cut_width = soft_cut_width
        self.hard_cut_rad = hard_cut_rad

        self.dtype = dtype
        self.device = device

        if num_mlp < 0:
            num_mlp = max(num_in, num_out)

        num_inputs = [num_in] + [2*num_mlp]*(num_layers-1) + [2*num_out]

        self.num_in = num_in
        self.num_mlp = num_mlp
        self.num_out = num_out

        # Set up MLPs
        self.mlps = nn.ModuleList()
        for n1, n2 in zip(num_inputs[:-1], num_inputs[1:]):
            mlp = BasicMLP(2*n1, n2, num_hidden=num_hidden, layer_width=layer_width, activation=activation, device=device, dtype=dtype)
            self.mlps.append(mlp)

        self.zero = torch.tensor(0, dtype=dtype, device=device)

    def forward(self, input_scalars, pos, atom_mask):
        atom_mask = atom_mask.unsqueeze(-1)

        norms = (pos.unsqueeze(-2) - pos.unsqueeze(-3)).norm(dim=-1)

        Adj = (mask.unsqueeze(-1) * mask.unsqueeze(-2)) * (norms > 0)
        mask = mask.unsqueeze(-1)

        if self.hard_cut_rad is not None:
            Adj = (Adj * (norms < self.hard_cut_rad))

        Adj = Adj.to(self.dtype)

        if self.soft_cut_rad is not None and self.soft_cut_width is not None:
            Adj *= torch.sigmoid(-(norms - self.soft_cut_rad)/self.soft_cut_width)

        features = input_scalars
        for mlp in self.mlps:
            message_pass = torch.matmul(Adj, features)
            message_pass = torch.cat([message_pass, features], dim=-1)
            features = torch.where(mask, mlp(message_pass), self.zero)

        out = features.view(input_scalars.shape[0:2] + (self.num_out, 1, 2))

        return out
