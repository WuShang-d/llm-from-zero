import torch

def get_rotate_frequency(
    dim: int,
    base: float = 10000.0,
    device=None,
):
    frequency = torch.zeros(dim // 2, device=device)

    for i in range(dim // 2):
        frequency[i] = pow(base, -2 * i / dim)

    return frequency


def apply_rope(q: torch.Tensor, k: torch.Tensor):
    """
    q shape: [B, H, T, D]
    k shape: [B, H, T, D]

    return:
        q_rotated shape: [B, H, T, D]
        k_rotated shape: [B, H, T, D]
    """
    assert q.shape == k.shape
    assert q.ndim == 4

    _, _, sequence_length, head_dim = q.shape
    assert head_dim % 2 == 0

    position = torch.arange(sequence_length, device=q.device)
    frequency = get_rotate_frequency(head_dim, device=q.device)

    angle = position[:, None] * frequency[None, :]
    
    #print(angle)

    cos_angle = torch.cos(angle).to(dtype=q.dtype)
    sin_angle = torch.sin(angle).to(dtype=q.dtype)

    def rotate(x: torch.Tensor):
        x_even = x[..., 0::2]
        x_odd = x[..., 1::2]

        rotated_even = (
            x_even * cos_angle
            - x_odd * sin_angle
        )
        rotated_odd = (
            x_odd * cos_angle
            + x_even * sin_angle
        )

        return torch.stack(
            [rotated_even, rotated_odd],
            dim=-1,
        ).flatten(-2)

    return rotate(q), rotate(k)

# q = torch.tensor(
#     [[[
#         [1.0, 2.0, 3.0, 4.0],  # 位置 m=0
#         [1.0, 0.0, 1.0, 0.0],  # 位置 m=1
#         [0.0, 1.0, 0.0, 1.0],  # 位置 m=2
#     ]]]
# )

# k = torch.tensor(
#     [[[
#         [2.0, 1.0, 4.0, 3.0],  # 位置 m=0
#         [0.0, 1.0, 0.0, 1.0],  # 位置 m=1
#         [1.0, 0.0, 1.0, 0.0],  # 位置 m=2
#     ]]]
# )

# q_rotated, k_rotated = apply_rope(q, k)

# print("q shape:", q.shape)
# print("q rotated:")
# print(q_rotated)

# print("\nk rotated:")
# print(k_rotated)
