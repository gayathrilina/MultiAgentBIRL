import unittest
import sys
import logging
import numpy
import time


logging.basicConfig(level=logging.DEBUG)

from mbirl.env.lb_foraging import ForagingEnv
from mbirl.birl.dp import DP


class TestDP(unittest.TestCase):

    LOGGER = logging.getLogger(__name__)

    def setUp(self):
        numpy.random.seed(12)

    def test_dp_init(self):
        self.LOGGER.info("Testing DP")

        env = ForagingEnv(3, 2, 2, 2)
        _obs = env.reset()

        dp = DP(env)

    def test_dp_policy_iteration(self):
        self.LOGGER.info("Testing DP")

        env = ForagingEnv(3, 2, 2, 2)
        _obs = env.reset()

        dp = DP(env)
        dp.policy_iteration()

        print(dp.policy)

    def test_dp_policy(self):
        self.LOGGER.info("Testing Policy")

        env = ForagingEnv(3, 2, 2, 2)
        _obs = env.reset()

        dp = DP(env)
        dp.policy_iteration()

        env.render()
        time.sleep(3)

        while 1:

            a = numpy.argmax(dp.policy[env.S_map[_obs]])

            _obs, _, done, _ = env.step(a)

            env.render()
            self.LOGGER.info(f"Agents took action {env.A[a]}")
            time.sleep(1)

            if done:
                break



if __name__ == "__main__":
    unittest.main()


