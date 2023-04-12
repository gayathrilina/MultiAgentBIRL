import unittest
import sys
import logging
import numpy

logging.basicConfig(level=logging.DEBUG)

from mbirl.env.lb_foraging import ForagingEnv


class TestEnv(unittest.TestCase):

    LOGGER = logging.getLogger(__name__)

    def setUp(self):
        numpy.random.seed(10)

    def test_env_init(self):
        self.LOGGER.info("Testing env")

        env = ForagingEnv(3, 2, 2, 2)

        _obs = env.reset()
        self.assertIsNotNone(_obs)

        self.LOGGER.info(f"Initial state: {_obs}")

    def test_env_next_states(self):

        env = ForagingEnv(3, 2, 2, 2)
        _obs = env.reset()

        for i in range(len(env.A)):
            # self.LOGGER.info(f"Testing {env.A[i]} on {_obs}")
            next_ = env.get_next_state(env.S_map[_obs], i)
            # self.LOGGER.info(f"Got: {next_}")

            self.assertTrue(next_ in env.S_map.keys())

    def test_env_T(self):

        env = ForagingEnv(3, 2, 2, 2)
        _obs = env.reset()

        self.assertTrue(numpy.min(env.T) == 0.0)
        self.assertTrue(numpy.max(env.T) == 1.0)

        self.assertTrue((numpy.sum(env.T, axis=2) == 1.0).all())

    def test_env_R(self):

        env = ForagingEnv(3, 2, 2, 2)
        _obs = env.reset()

        for s in range(len(env.S)):
            for a in range(len(env.A)):
                #print(f"R({env.S[s]}, {env.A[a]}) = {env.R[s, a]}")
                pass


if __name__ == "__main__":
    unittest.main()


