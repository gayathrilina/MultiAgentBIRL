import logging
from typing import List
from typing import Tuple
import numpy
from itertools import product
from functools import reduce
import os


class ForagingEnv:
    """
        Defines a basic foraging env
    """

    LOGGER = logging.getLogger(__name__)

    def __init__(self, x_size, y_size, num_agents, num_foods, rewards = None):

        self.x_size: int = x_size
        self.y_size: int = y_size

        self.n: int = num_agents
        self.f: int = num_foods

        if rewards is not None:
            self.rewards = rewards
            #assert len(self.rewards) == self.nrow*self.ncol
        else:
            self.rewards = None

        # create action space
        self.actions = ["X", "N", "S", "E", "W", "L"]
        self.A = list(product(self.actions,
            repeat=self.n))

        # create product space
        agent_range = list(product(range(self.x_size), range(self.y_size)))
        food_range = self.init_foods(agent_range)

        _S = [agent_range[:] for i in range(self.n)] + \
                [_food for _food in food_range]
        _S = product(*_S)

        # remove collisions
        self.S = list(filter(lambda x: not self.__is_collision(x), _S))
        self.S = self.__remove_done_states(self.S)

        # for DP to work
        self.num_states = len(self.S)
        self.num_actions = len(self.A)

        self.S_map = dict({(self.S[i], i) for i in range(len(self.S))})

        self.LOGGER.info(f"State space initialized with {len(self.S)}" +
                " states")
        self.LOGGER.info(f"Action space initialized with " +
                f"{len(self.A)}")

        self.init_T()
        self.init_R()

        self.P = self.T

    def __remove_done_states(self, S):

        done_state_check = tuple([(-1, -1) for f in range(self.f)])
        done_states = set(filter(lambda x: tuple(x[self.n:]) \
                == done_state_check, S))
        
        S = list(filter(lambda x: x not in done_states, S))
        S.append(tuple([(-1, -1) for i in range(self.n + self.f)]))

        return S

    def __is_collision(self, s: List[Tuple[int, int]]) -> bool:

        _locs = set()
        for _s in s:

            if _s in _locs:
                return True

            else:
                if _s != (-1, -1):
                    _locs.add(_s)

        return False

    def init_foods(self, grid) -> None:

        food_locs = []

        for f in range(self.f):
            while 1:
                _loc = grid[numpy.random.randint(len(grid))]
                if _loc not in food_locs:
                    food_locs.append(_loc)
                    break

        # Commented this to remove randomness
        #self.food_weights = [numpy.random.randint(1, self.n + 1) for i in
        #        range(self.f)]
        #self.food_locs = food_locs

        self.food_weights = [1, 2]
        food_locs = [(1, 1), (2, 1)]
        self.food_locs = food_locs

        end_locs = [(-1, -1) for f in food_locs]

        return [[food_locs[i], end_locs[i]] for i in \
                range(len(self.food_weights))]

    def next_state_agent(self, 
            prev_pos: Tuple[int, int], 
            action: str) -> Tuple[int, int]:

        if action == "X":
            return prev_pos

        elif action == "S":

            if prev_pos[1] < self.y_size - 1:
                return (prev_pos[0], prev_pos[1] + 1)

            else:
                return prev_pos

        elif action == "N":

            if prev_pos[1] > 0:
                return (prev_pos[0], prev_pos[1] - 1)

            else:
                return prev_pos

        elif action == "E":

            if prev_pos[0] < self.x_size - 1:
                return (prev_pos[0] + 1, prev_pos[1])

            else:
                return prev_pos

        elif action == "W":

            if prev_pos[0] > 0:
                return (prev_pos[0] - 1, prev_pos[1])

            else:
                return prev_pos

        elif action == "L":
            return prev_pos

        else:
            self.LOGGER.error(f"Incorrect action {action} at pos {prev_pos}")
            return None

    def get_neighbor_set(self, pos):

        n = set()
        if pos[0] < self.x_size - 1:
            n.add((pos[0] + 1, pos[1]))

        if pos[0] > 0:
            n.add((pos[0] - 1, pos[1]))

        if pos[1] < self.y_size - 1:
            n.add((pos[0], pos[1] + 1))

        if pos[1] > 0:
            n.add((pos[0], pos[1] - 1))

        return n

    def get_next_state(self, i_s: int, i_a: int):

        s = self.S[i_s]
        a = self.A[i_a]

        foods = [s[f] for f in range(self.n, self.n + self.f)]
        _foods = len(list(filter(lambda x: x == (-1, -1), foods)))

        if _foods == self.f:
            return s

        next_state = []
        loaders = set()
        for i in range(len(a)):

            occupied: List[Tuple[int, int]] = list(s)[:len(a)]
            occupied.pop(i)

            if a[i] == "L":
                loaders.add(s[i])
                next_state.append(s[i])
                continue

            next_pos = self.next_state_agent(s[i], a[i])

            if next_pos not in next_state and \
                    next_pos not in foods and next_pos not in occupied:
                        next_state.append(next_pos)

            else:
                next_state.append(s[i])

        for i, f in enumerate(foods):

            if f == (-1, -1):
                next_state.append(f)
                continue

            if len(loaders.intersection(self.get_neighbor_set(f))) >= \
                    self.food_weights[i]:
                next_state.append((-1, -1))

                for _n in loaders.intersection(self.get_neighbor_set(f)):
                    loaders.remove(_n)

            else:
                next_state.append(f)

        _food_state = next_state[self.n:]
        _done_food_state = tuple([(-1, -1) for f in range(self.f)])

        if tuple(_food_state) == _done_food_state:
            return tuple([(-1, -1) for i in range(self.f + self.n)])

        return tuple(next_state)

    def step(self, a: int):

        done = False

        s = self.S_map[self.current_state]
        self.current_state = self.get_next_state(s, a)

        if self.current_state not in self.S_map.keys():
            print(f"[!!!] Fatal error: {self.current_state} not in T")

        done_test = tuple([(-1, -1) for i in range(self.n + self.f)])

        if self.current_state == done_test:
            done = True

        return self.current_state, self.R[s, a], done, {}

    def init_T(self):

        self.T = numpy.zeros(shape=[len(self.S), len(self.A), len(self.S)])

        for s in range(len(self.S)):
            for a in range(len(self.A)):
                next_state = self.get_next_state(s, a)
                i_s_p = self.S_map[next_state]

                self.T[s, a, i_s_p] = 1.0

    def init_R(self):

        self.R = numpy.zeros(shape=[len(self.S), len(self.A)])
        
        for s in range(len(self.S)):
            for a in range(len(self.A)):
                next_state = self.get_next_state(s, a)
                count_done = len(list(filter(
                        lambda x: x == (-1, -1),
                        next_state[self.n:])))

                if count_done == 0:
                    count_done = -0.1

                self.R[s, a] = float(count_done)



    def reset(self) -> List[Tuple[int, int]]:

        while 1:
            _sample = self.S[numpy.random.randint(len(self.S))]
            if (-1, -1) not in _sample:
                break
        
        self.current_state = _sample

        return _sample

    def T(self, 
            s: int, 
            a: int, 
            s_p: int) -> float:

        return 0.0

    def render(self):
        os.system("clear")

        display = []

        for j in range(self.y_size):

            row = ""
            for i in range(self.x_size):

                if (i, j) in self.current_state:

                    for n in range(self.n):

                        if (i, j) == self.current_state[n]:
                            row += f" {n} "

                    for f in range(self.f):

                        if (i, j) == self.current_state[f + self.n]:
                            row += " @ "

                else:
                    row += " . "

            display.append(row)

        for r in display:
            print(r)

        print(f"Food weights: {self.food_weights}")




