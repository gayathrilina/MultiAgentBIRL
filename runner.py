import numpy
import logging

from mbirl.env.lb_foraging import ForagingEnv


LOGGER = logging.getLogger(__name__)
logging.basicConfig(level=logging.DEBUG)

# seed the randomness
numpy.random.seed(12)

env = ForagingEnv(3, 2, 2, 2)
_s = env.reset()
env.render()

print(f"Initial state sampled to {_s}")
LOGGER.debug(f"Env has {len(env.S)} states in total")
