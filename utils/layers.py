import base64
import io
import pickle
import numpy as np
import torch

print(f'TORCH VERSION: {torch.__version__}')

import packaging.version
current_version = packaging.version.parse(torch.__version__)
min_version = packaging.version.parse('1.5.0')
if current_version < min_version:
    raise RuntimeError('Torch versions lower than 1.5.0 not supported')

if torch.cuda.is_available():
    torch_device = 'cuda'
    float_dtype = np.float32  # single
    torch.set_default_tensor_type(torch.cuda.FloatTensor)

else:
    torch_device = 'cpu'
    float_dtype = np.float64
    torch.set_default_tensor_type(torch.DoubleTensor)

print(f'TORCH DEVICE: {torch_device}')


def torch_mod(x):
    return torch.remainder(x, 2 * np.pi)


def torch_wrap(x):
    return torch_mod(x + np.pi) - np.pi


def grab(x):
    return x.detach().cpu().numpy()



def make_checker_mask(shape, parity):
    checker = torch.ones(shape, dtype=torch.uint8) - parity
    checker[::2, ::2] = parity
    checker[1::2, 1::2] = parity
    return checker.to(torch_device)


class SimpleCoupling(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.s = torch.nn.Sequential(
            torch.nn.Linear(1, 8),
            torch.nn.ReLU(),
            torch.nn.Linear(8, 8),
            torch.nn.ReLU(),
            # TODO: Finish
        )


class AffineCoupling(torch.nn.Module):
    def __init__(self, net, *, mask_shape, mask_parity):
        super().__init__()
        self.mask = make_checker_mask(mask_shape, mask_parity)
        self.net = net

    def forward(self, x):
        x_frozen = self.mask * x
        x_active = (1 - self.mask) * x
        net_out = self.net(x_frozen.unsqueeze(1))
        s, t = net_out[:, 0], net_out[:, 1]
        fx = (1 - self.mask) * t + x_active * torch.exp(s) + x_frozen
        axes = range(1, len(s.size()))
        logJ = torch.sum((1 - self.mask) * s, dim=tuple(axes))
        return fx, logJ

    def reverse(self, fx):
        fx_frozen = self.mask * fx
        fx_active = (1 - self.mask) * fx
        net_out = self.net(fx_frozen.unsqueeze(1))
        s, t = net_out[:, 0], net_out[:, 1]
        x = (fx_active - (1 - self.mask) * t) * torch.exp(-s) + fx_frozen
        axes = range(1, len(s.size()))
        logJ = torch.sum((1 - self.mask) * (-s), dim=tuple(axes))
        return x, logJ


def make_conv_net(
        *,
        hidden_sizes,
        kernel_size,
        in_channels,
        out_channels,
        use_final_tanh
):
    sizes = [in_channels]  + hidden_sizes + [out_channels]
    assert
