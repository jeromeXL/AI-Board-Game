# Student agent: Add your own agent here
from agents.agent import Agent
from store import register_agent
import sys
import numpy as np
from copy import deepcopy
import time
import pdb


@register_agent("student_agent")
class StudentAgent(Agent):
    """
    A dummy class for your implementation. Feel free to use this class to
    add any helper functionalities needed for your agent.
    """

    def __init__(self):
        super(StudentAgent, self).__init__()
        self.name = "StudentAgent"
        self.dir_map = {
            "u": 0,
            "r": 1,
            "d": 2,
            "l": 3,
        }
        self.moves_taken = 0

    def get_f_value(self, position, adv_pos):
        heuristic_value = compute_heuristic(position, adv_pos)
        return heuristic_value

    def step(self, chess_board, my_pos, adv_pos, max_step):
        """
        Implement the step function of your agent here.
        You can use the following variables to access the chess board:
        - chess_board: a numpy array of shape (x_max, y_max, 4)
        - my_pos: a tuple of (x, y)
        - adv_pos: a tuple of (x, y)
        - max_step: an integer

        You should return a tuple of ((x, y), dir),
        where (x, y) is the next position of your agent and dir is the direction of the wall
        you want to put on.

        So you return new_pos, self.dir_map[dir]

        Please check the sample implementation in agents/random_agent.py or agents/human_agent.py for more details.
        """
        # pdb.set_trace()
        # Some simple code to help you with timing. Consider checking
        # time_taken during your search and breaking with the best answer
        # so far when it nears 2 seconds.
        start_time = time.time()
        self.moves_taken = 0
        # Moves (Up, Right, Down, Left)
        moves = ((-1, 0), (0, 1), (1, 0), (0, -1))
        visited = []
        state_queue = [(my_pos, self.get_f_value(my_pos, adv_pos))]

        while self.moves_taken <= max_step:
            # pdb.set_trace()
            state_queue = sort_state_queue(state_queue)
            # print("State Queue Start Of Loop: ", state_queue)
            if (not state_queue):
                return my_pos, 0
            cur_pos, f_value = state_queue.pop(0)
            visited.append(cur_pos)
            self.moves_taken += 1
            x, y = cur_pos
            for dir, move in enumerate(moves):
                if chess_board[x, y, dir]:
                    continue
                next_pos = tuple(np.array(cur_pos) + np.array(move))
                # print("Cur Position: ", cur_pos)
                # print("Next Position: ", next_pos)
                # print("Adv Position: ", adv_pos)
                if next_pos == adv_pos:
                    continue
                new_element_in_queue = (
                    next_pos, self.get_f_value(next_pos, adv_pos))
                # pdb.set_trace()
                # print("Next element in queue: ", new_element_in_queue)
                state_queue.append(new_element_in_queue)
                # print("State Queue At End Of Loop: ", state_queue)

        # Final portion, pick where to put our new barrier, at random
        my_pos = visited[1]
        x, y = my_pos
        # Possibilities, any direction such that chess_board is False
        allowed_barriers = [i for i in range(0, 4) if not chess_board[x, y, i]]
        # Sanity check, no way to be fully enclosed in a square, else game already ended
        assert len(allowed_barriers) >= 1
        dir = allowed_barriers[np.random.randint(0, len(allowed_barriers))]
        time_taken = time.time() - start_time
        print("My AI's turn took ", time_taken, "seconds.")

        return my_pos, dir


def compute_heuristic(position, adv_pos):
    distance = get_manhattan_distance(position, adv_pos)
    return distance


def get_manhattan_distance(my_pos, adv_pos):
    return abs(my_pos[0] - adv_pos[0]) + abs(my_pos[1]-adv_pos[1])


def sort_state_queue(state_queue):
    return sorted(state_queue, key=lambda x: x[1])
