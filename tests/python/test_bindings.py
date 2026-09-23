import unittest

from odelab import Method, solve, solve_linear, solve_linear_batch


class BindingsTest(unittest.TestCase):
    def test_decay_all_methods(self):
        for method, factor in [
            (Method.forward_euler, 0.9),
            (Method.backward_euler, 1 / 1.1),
            (Method.trapezoidal, 0.95 / 1.05),
        ]:
            with self.subTest(method=method):
                linear = solve_linear(-1, 0, 1, 0, 1, 10, method)
                callback = solve(lambda t, y: -y, 1, 0, 1, 10, method)
                self.assertEqual(len(linear.t), 11)
                self.assertEqual(linear.t[-1], 1)
                self.assertAlmostEqual(linear.y[-1], factor**10)
                self.assertAlmostEqual(callback.y[-1], factor**10)
                batch = solve_linear_batch([-1, -1], [0, 0], [1, 2], 0, 1, 10, method)
                self.assertEqual(batch.shape, (2, 11))
                self.assertAlmostEqual(batch[0, -1], factor**10)
                self.assertAlmostEqual(batch[1, -1], 2 * factor**10)

    def test_invalid_steps(self):
        with self.assertRaises(ValueError):
            solve_linear(-1, 0, 1, 0, 1, 0, Method.forward_euler)

    def test_callback_exception(self):
        def rhs(t, y):
            raise ValueError("callback failed")

        with self.assertRaisesRegex(ValueError, "callback failed"):
            solve(rhs, 1, 0, 1, 10, Method.forward_euler)


if __name__ == "__main__":
    unittest.main()
