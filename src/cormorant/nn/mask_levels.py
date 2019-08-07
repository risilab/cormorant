import torch
import torch.nn as nn

class MaskLevel(nn.Module):
    """
    Mask level for implementing hard and soft cutoffs. With the current
    architecutre, we have all-to-all communication.

    This mask takes relative position vectors :math:`r_{ij} = r_i - r_j`
    and implements either a hard cutoff, a soft cutoff, or both. The soft
    cutoffs can also be made learnable.
    """
    def __init__(self, num_rad_channels, hard_cut_rad, soft_cut_rad, soft_cut_width, cutoff_type,
                 gaussian_mask=False, eps=1e-3, device=torch.device('cpu'), dtype=torch.float):
        super(MaskLevel, self).__init__()

        self.gaussian_mask = gaussian_mask

        self.num_rad_channels = num_rad_channels

        # Initialize hard/soft cutoff to None as default.
        self.hard_cut_rad = None
        self.soft_cut_rad = None
        self.soft_cut_width = None

        if 'hard' in cutoff_type:
            self.hard_cut_rad = hard_cut_rad

        if ('soft' in cutoff_type) or ('learn' in cutoff_type) or ('learn_rad' in cutoff_type) or ('learn_width' in cutoff_type):

            self.soft_cut_rad = soft_cut_rad*torch.ones(num_rad_channels, device=device, dtype=dtype).view((1, 1, 1, -1))
            self.soft_cut_width = soft_cut_width*torch.ones(num_rad_channels, device=device, dtype=dtype).view((1, 1, 1, -1))

            if ('learn' in cutoff_type) or ('learn_rad' in cutoff_type):
                self.soft_cut_rad = nn.Parameter(self.soft_cut_rad)

            if ('learn' in cutoff_type) or ('learn_width' in cutoff_type):
                self.soft_cut_width = nn.Parameter(self.soft_cut_width)

        # Standard bookkeeping
        self.dtype = dtype
        self.device = device

        self.zero = torch.tensor(0, device=device, dtype=dtype)
        self.eps = torch.tensor(eps, device=device, dtype=dtype)

    def forward(self, edge_net, hard_mask, norms):
        if self.hard_cut_rad is not None:
            hard_mask = (hard_mask * (norms < self.hard_cut_rad))

        edge_mask = hard_mask.to(self.dtype).unsqueeze(-1).to(self.dtype)

        if self.soft_cut_rad is not None:
            cut_width = torch.max(self.eps, self.soft_cut_width.abs())
            cut_rad = torch.max(self.eps, self.soft_cut_rad.abs())

            if self.gaussian_mask:
                edge_mask = edge_mask * torch.exp(-(norms.unsqueeze(-1)/cut_rad).pow(2))
            else:
                edge_mask = edge_mask * torch.sigmoid((cut_rad - norms.unsqueeze(-1))/cut_width)

        edge_mask = edge_mask.unsqueeze(-1)

        edge_net = [edge*edge_mask for edge in edge_net]

        return edge_net
