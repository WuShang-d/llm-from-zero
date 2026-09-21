import torch

def RMSNorm(x: torch.Tensor, g: torch.Tensor):
    """
    x shape: [B, T, D]
    g shape: [D]
    """
    epsilon = 1e-6

    # [B, T, D] -> [B, T, 1]
    rms = torch.mean(x**2, dim=-1, keepdim=True)
    rms = torch.sqrt(rms + epsilon)

    # [B, T, D] / [B, T, 1] -> [B, T, D]
    # [B, T, D] * [D]       -> [B, T, D]
    return (x / rms) * g

# x = torch.tensor(
#     [
#         [[1, 2, 1, 2], [3, 4, 1, 2], [5, 7, 1, 2]],
#         [[5, 6, 1, 2], [7, 8, 1, 2], [9, 0, 1, 2]],
#     ],
#     dtype=torch.float32,
# )

# g = torch.tensor([1, 2, 3, 4], dtype=torch.float32)

# x = RMSNorm(x, g)
# print(x)