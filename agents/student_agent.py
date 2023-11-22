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
        self.cost_of_path=0
        self.dir_map = {
            "u": 0,
            "r": 1,
            "d": 2,
            "l": 3,
        }

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
        pdb.set_trace()
        # Some simple code to help you with timing. Consider checking 
        # time_taken during your search and breaking with the best answer
        # so far when it nears 2 seconds.
        start_time = time.time()
        # Moves (Up, Right, Down, Left)
        moves = ((-1, 0), (0, 1), (1, 0), (0, -1))
        visited = {tuple(my_pos)}
        state_queue = [(my_pos, get_f_value(my_pos, adv_pos, self.cost_of_path))]
        is_reached = False
        
        while state_queue and not is_reached:
            state_queue = sort_state_queue(state_queue)
            cur_pos, cur_step = state_queue.pop(0)
            self.cost_of_path += get_manhattan_distance(my_pos, cur_pos)
            x,y = cur_pos

        time_taken = time.time() - start_time
        
        print("My AI's turn took ", time_taken, "seconds.")

        return my_pos, dir

def get_f_value(position, adv_pos, cost_of_path):
    heuristic_value = compute_heuristic(position, adv_pos)
    return heuristic_value + cost_of_path

def compute_heuristic(position, adv_pos):
    distance = get_manhattan_distance(position, adv_pos)
    return distance

def get_manhattan_distance(my_pos, adv_pos):
    return abs(my_pos[0] - adv_pos[0]) + abs(my_pos[1]-adv_pos[1])

def sort_state_queue(state_queue):
    return sorted(state_queue, key = lambda x: x[1])


