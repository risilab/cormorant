import torch
import torch.nn as nn

from math import pi

class RadialFilters(nn.Module):
    """
    Generate a set of learnable scalar functions for the aggregation/point-wise
    convolution step.

    One set of radial filters is created for each irrep (l = 0, ..., max_sh).

    :max_sh: Maximum l to use for the spherical harmonics.
    :basis_set: Parameters of basis set to use. See RadPolyTrig for mroe details.
    :num_channels_out: Number of output channels to mix the resulting function
    into if mix is set to True in RadPolyTrig
    :num_levels: Number of CG levels in the Cormorant.
    """
    def __init__(self, max_sh, basis_set, num_channels_out,
                 num_levels, device=torch.device('cpu'), dtype=torch.float):
        super(RadialFilters, self).__init__()

        self.num_levels = num_levels
        self.max_sh = max_sh

        rad_funcs = [RadPolyTrig(max_sh[level], basis_set, num_channels_out[level], device=device, dtype=dtype) for level in range(self.num_levels)]
        self.rad_funcs = nn.ModuleList(rad_funcs)
        self.tau = [rad_func.radial_types for rad_func in self.rad_funcs]

        self.num_rad_channels = self.tau[0][0]

        # Other things
        self.device = device
        self.dtype = dtype

        self.zero = torch.tensor(0, device=device, dtype=dtype)

    def forward(self, norms, base_mask):

        return [rad_func(norms, base_mask) for rad_func in self.rad_funcs]


class RadPolyTrig(nn.Module):
    """
    A variation/generalization of spherical bessel functions.
    Rather than than introducing the bessel functions explicitly we just write out a basis
    that can produce them. Then, when apply a weight mixing matrix to reduce the number of channels
    at the end.
    :max_sh:
    """
    def __init__(self, max_sh, basis_set, num_channels, mix=False, device=torch.device('cpu'), dtype=torch.float):
        super(RadPolyTrig, self).__init__()

        trig_basis, rpow = basis_set
        self.rpow = rpow
        self.max_sh = max_sh

        assert(trig_basis >= 0 and rpow >= 0)

        self.num_rad = (trig_basis+1)*(rpow+1)
        self.num_channels = num_channels

        # This instantiates a set of functions sin(2*pi*n*x/a), cos(2*pi*n*x/a) with a=1.
        self.scales = torch.cat([torch.arange(trig_basis+1), torch.arange(trig_basis+1)]).view(1, 1, 1, -1).to(device=device, dtype=dtype)
        self.phases = torch.cat([torch.zeros(trig_basis+1), pi/2*torch.ones(trig_basis+1)]).view(1, 1, 1, -1).to(device=device, dtype=dtype)

        # This avoids the sin(0*r + 0) = 0 part from wasting computations.
        self.phases[0, 0, 0, 0] = pi/2

        # Now, make the above learnable
        self.scales = nn.Parameter(self.scales)
        self.phases = nn.Parameter(self.phases)

        # If desired, mix the radial components to a desired shape
        self.mix = mix
        if (mix == 'cplx') or (mix is True):
            self.linear = nn.ModuleList([nn.Linear(2*self.num_rad, 2*self.num_channels).to(device=device, dtype=dtype) for _ in range(max_sh+1)])
            self.radial_types = (num_channels,) * (max_sh + 1)
        elif mix == 'real':
            self.linear = nn.ModuleList([nn.Linear(2*self.num_rad, self.num_channels).to(device=device, dtype=dtype) for _ in range(max_sh+1)])
            self.radial_types = (num_channels,) * (max_sh + 1)
        elif (mix == 'none') or (mix is False):
            self.linear = None
            self.radial_types = (self.num_rad,) * (max_sh + 1)
        else:
            raise ValueError('Can only specify mix = real, cplx, or none! {}'.format(mix))

        self.zero = torch.tensor(0, device=device, dtype=dtype)

    def forward(self, norms, edge_mask):
        # Shape to resize at end
        s = norms.shape

        # Mask and reshape
        edge_mask = (edge_mask * (norms > 0).byte()).unsqueeze(-1)
        norms = norms.unsqueeze(-1)

        # Get inverse powers
        rad_powers = torch.stack([torch.where(edge_mask, norms.pow(-pow), self.zero) for pow in range(self.rpow+1)], dim=-1)

        # Calculate trig functions
        rad_trig = torch.where(edge_mask, torch.sin((2*pi*self.scales)*norms+self.phases), self.zero).unsqueeze(-1)

        # Take the product of the radial powers and the trig components and reshape
        rad_prod = (rad_powers*rad_trig).view(s + (1, 2*self.num_rad,))

        # Apply linear mixing function, if desired
        if self.mix == 'cplx':
            radial_functions = [linear(rad_prod).view(s + (self.num_channels, 2)) for linear in self.linear]
        elif self.mix == 'real':
            radial_functions = [linear(rad_prod).view(s + (self.num_channels,)) for linear in self.linear]
        else:
            radial_functions = [rad_prod.view(s + (self.num_rad, 2))] * (self.max_sh + 1)

        return radial_functions
