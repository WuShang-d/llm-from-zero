"""
运行：
    python3 -m unittest discover -s model -p "test_*.py" -v

这些测试分别检查：
    1. shape 是否符合模块约定
    2. 前向计算的数值是否正确
    3. 参数量是否正确
    4. 梯度能否传回可学习参数
"""

import unittest

import torch

from embedding import TokenEmbedding
from layers import Linear
from rmsnorm import RMSNorm


class TestTokenEmbedding(unittest.TestCase):
    def setUp(self):
        # 固定随机数种子，让测试每次运行都可复现。
        torch.manual_seed(0)
        self.vocab_size = 5
        self.d_model = 3
        self.embedding = TokenEmbedding(self.vocab_size, self.d_model)

    def test_output_shape(self):
        token_ids = torch.tensor(
            [[0, 1, 2], [3, 4, 1]],
            dtype=torch.long,
        )  # [B=2, T=3]

        output = self.embedding(token_ids)

        self.assertEqual(output.shape, (2, 3, self.d_model))

    def test_lookup_returns_the_selected_rows(self):
        token_ids = torch.tensor([[1, 4], [2, 1]], dtype=torch.long)

        output = self.embedding(token_ids)

        # token_ids[0, 0] == 1，因此这里应该等于权重表的第 1 行。
        torch.testing.assert_close(output[0, 0], self.embedding.weight[1])
        torch.testing.assert_close(output[0, 1], self.embedding.weight[4])

        # 相同 token ID 必须查到相同的向量。
        torch.testing.assert_close(output[0, 0], output[1, 1])

    def test_parameter_count(self):
        parameter_count = sum(p.numel() for p in self.embedding.parameters())
        self.assertEqual(parameter_count, self.vocab_size * self.d_model)

    def test_gradient_reaches_used_rows(self):
        token_ids = torch.tensor([[1, 3, 1]], dtype=torch.long)

        output = self.embedding(token_ids)
        output.sum().backward()

        self.assertIsNotNone(self.embedding.weight.grad)

        # token 1 和 token 3 在本批次出现过，因此对应行有梯度。
        self.assertGreater(self.embedding.weight.grad[1].abs().sum().item(), 0)
        self.assertGreater(self.embedding.weight.grad[3].abs().sum().item(), 0)

        # token 0 没有出现，因此对应行的梯度为 0。
        self.assertEqual(self.embedding.weight.grad[0].abs().sum().item(), 0)


class TestLinear(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(0)
        self.d_in = 3
        self.d_out = 2
        self.linear = Linear(self.d_in, self.d_out)

    def test_output_shape(self):
        x = torch.randn(2, 4, self.d_in)  # [B=2, T=4, Din=3]

        output = self.linear(x)

        self.assertEqual(output.shape, (2, 4, self.d_out))

    def test_forward_matches_hand_calculation(self):
        # 明确指定参数，避免测试依赖随机初始化值。
        with torch.no_grad():
            self.linear.weight.copy_(
                torch.tensor(
                    [
                        [1.0, 2.0],
                        [3.0, 4.0],
                        [5.0, 6.0],
                    ]
                )
            )
            self.linear.bias.copy_(torch.tensor([0.5, -0.5]))

        x = torch.tensor([[[1.0, 2.0, 3.0]]])  # [1, 1, 3]
        output = self.linear(x)

        # 第一个输出：1*1 + 2*3 + 3*5 + 0.5 = 22.5
        # 第二个输出：1*2 + 2*4 + 3*6 - 0.5 = 27.5
        expected = torch.tensor([[[22.5, 27.5]]])
        torch.testing.assert_close(output, expected)

    def test_parameter_count(self):
        parameter_count = sum(p.numel() for p in self.linear.parameters())
        expected = self.d_in * self.d_out + self.d_out
        self.assertEqual(parameter_count, expected)

    def test_gradients_reach_input_weight_and_bias(self):
        x = torch.randn(2, 4, self.d_in, requires_grad=True)

        self.linear(x).sum().backward()

        self.assertIsNotNone(x.grad)
        self.assertIsNotNone(self.linear.weight.grad)
        self.assertIsNotNone(self.linear.bias.grad)


class TestRMSNorm(unittest.TestCase):
    def setUp(self):
        self.d_model = 4
        self.g = torch.ones(self.d_model, requires_grad=True)

    def test_output_shape(self):
        x = torch.randn(2, 3, self.d_model)

        output = RMSNorm(x, self.g)

        self.assertEqual(output.shape, x.shape)

    def test_forward_matches_reference_formula(self):
        x = torch.tensor([[[1.0, 2.0, 3.0, 4.0]]])
        g = torch.tensor([1.0, 2.0, 3.0, 4.0])

        output = RMSNorm(x, g)

        rms = torch.sqrt(torch.tensor((1.0 + 4.0 + 9.0 + 16.0) / 4.0 + 1e-6))
        expected = (x / rms) * g
        torch.testing.assert_close(output, expected)

    def test_rms_is_nearly_one_when_g_is_one(self):
        x = torch.tensor(
            [
                [[1.0, 2.0, 3.0, 4.0], [2.0, -2.0, 2.0, -2.0]],
                [[4.0, 3.0, 2.0, 1.0], [-1.0, -2.0, -3.0, -4.0]],
            ]
        )

        output = RMSNorm(x, self.g)
        output_rms = torch.sqrt(torch.mean(output**2, dim=-1))

        torch.testing.assert_close(
            output_rms,
            torch.ones_like(output_rms),
            rtol=1e-5,
            atol=1e-5,
        )

    def test_zero_input_is_finite(self):
        x = torch.zeros(2, 3, self.d_model)

        output = RMSNorm(x, self.g)

        self.assertTrue(torch.isfinite(output).all().item())
        torch.testing.assert_close(output, torch.zeros_like(output))

    def test_does_not_modify_input(self):
        x = torch.randn(2, 3, self.d_model)
        original = x.clone()

        RMSNorm(x, self.g)

        torch.testing.assert_close(x, original)

    def test_gradients_reach_input_and_scale(self):
        x = torch.randn(2, 3, self.d_model, requires_grad=True)

        RMSNorm(x, self.g).sum().backward()

        self.assertIsNotNone(x.grad)
        self.assertIsNotNone(self.g.grad)


if __name__ == "__main__":
    unittest.main()
