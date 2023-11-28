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

    def get_f_value(self, position, adv_pos, chess_board, max_step):
        heuristic_value = compute_heuristic(
            position, adv_pos, chess_board, max_step)
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
        start_time = time.time()
        # Counter for moves taken in a given turn
        self.moves_taken = 0
        # Moves (Up, Right, Down, Left)
        moves = ((-1, 0), (0, 1), (1, 0), (0, -1))
        # List of all positions visited
        visited = []
        # Queue of states of the form (pos, f_value(pos))
        state_queue = [
            (my_pos, self.get_f_value(my_pos, adv_pos, chess_board, max_step))]

        # While loop, break if num moves taken is more than max # moves
        while self.moves_taken <= max_step:
            # pdb.set_trace()
            # Sort Queue of states in increasing order of f-value to convert queue to priority queue
            state_queue = sort_position_queue(state_queue)
            # print("State Queue Start Of Loop: ", state_queue)
            # If no states in queue, no possible moves
            # 3 walls are around my_pos and opponent is blocking exit, return my_pos and place wall facing opponent - accepting loss
            if not state_queue:
                # pdb.set_trace()
                x, y = my_pos
                # Get allowed barries, will only be one facing the opponent
                allowed_barriers = [i for i in range(
                    0, 4) if not chess_board[x, y, i]]
                # Return current position and barrier
                return my_pos, allowed_barriers[0]
            # Get current position and f-value from front of queue
            cur_pos, f_value = state_queue.pop(0)
            # Add current position and f-value onto list of visited positions
            visited_list_element = (cur_pos, f_value)
            visited.append(visited_list_element)
            # Increment number of moves taken
            self.moves_taken += 1
            x, y = cur_pos
            print("Cur Pos: ", cur_pos)
            # Iterate through list of moves
            for dir, move in enumerate(moves):
                # Check if there is a wall in the direction dir given position x,y
                if chess_board[x, y, dir]:
                    continue
                # Compute possible next pos based on move (move current position up down right or left)
                next_pos = tuple(np.array(cur_pos) + np.array(move))
                # print("Cur Position: ", cur_pos)
                # print("Next Position: ", next_pos)
                # print("Adv Position: ", adv_pos)
                # Do not allow my agent position to collide with adversary agent position
                if next_pos == adv_pos:
                    continue
                print("Move Taken: ", move)
                print("Next Position: ", next_pos)
                # Add next pos as element of queue along with its f-value
                new_element_in_queue = (
                    next_pos,
                    self.get_f_value(next_pos, adv_pos, chess_board, max_step),
                )
                # pdb.set_trace()
                # print("Next element in queue: ", new_element_in_queue)
                state_queue.append(new_element_in_queue)
                # print("State Queue At End Of Loop: ", state_queue)

        # Sort list of visited positions in search in increasing order of f-value
        visited = sort_position_queue(visited)
        # Pick to move to position with lowest f-value
        my_pos = visited[1][0]
        x, y = my_pos
        # Get allowed directions for barrier placement
        allowed_barriers = [i for i in range(0, 4) if not chess_board[x, y, i]]
        # Sanity check, no way to be fully enclosed in a square, else game already ended
        assert len(allowed_barriers) >= 1
        # Get a random direction in which to place barrier
        dir = pick_wall_direction(my_pos, adv_pos, chess_board, max_step)
        # dir = allowed_barriers[np.random.randint(0, len(allowed_barriers))]
        time_taken = time.time() - start_time
        print("My AI's turn took ", time_taken, "seconds.")
        # Return position to move to and direction to place barrier
        return my_pos, dir


# Compute heuristic of position to move to


def compute_heuristic(position, adv_pos, chess_board, max_step):
    # pdb.set_trace()
    if (get_manhattan_distance(position, adv_pos) > max_step * 4):
        heuristic = get_manhattan_distance(position, adv_pos)
    else:
        heuristic = get_num_walls(position, adv_pos, chess_board)
    return heuristic


# Get num walls surrounding a position to move to


def get_num_walls(position, adv_pos, chess_board):
    # Counter for num walls around position to move to
    num_walls = 0
    moves = ((-1, 0), (0, 1), (1, 0), (0, -1))
    x = position[0]
    y = position[1]
    # Iterate through different moves
    for dir, move in enumerate(moves):
        # Check if there is a wall in all directions
        if chess_board[x, y, dir]:
            # Increment num walls
            num_walls += 1
    # If number of walls is 3, assign arbitrary high number to ensure that this position is not picked as a next move
    if num_walls == 3:
        return 100
    return num_walls


# Sort a queue of the form [position, f-value(position)] in increasing order


def sort_position_queue(queue):
    return sorted(queue, key=lambda x: x[1])

# Number of moves opponent can do when selecting a potential position to move to


def get_num_possible_op_moves(position, adv_pos, chess_board, max_step):
    opponent_steps = 0
    state_queue = [adv_pos]
    visited = []
    moves = ((-1, 0), (0, 1), (1, 0), (0, -1))
    while state_queue and opponent_steps <= max_step:
        cur_pos = state_queue.pop(0)
        opponent_steps += 1
        r, c = cur_pos
        for dir, move in enumerate(moves):
            if chess_board[r, c, dir]:
                continue
            next_pos = tuple(np.array(cur_pos) + np.array(move))
            if next_pos == position or next_pos in visited:
                continue
            visited.append(next_pos)
            state_queue.append(next_pos)
    return len(visited)

# Get manhattan distance


def get_manhattan_distance(my_pos, adv_pos):
    return abs(my_pos[0] - adv_pos[0]) + abs(my_pos[1]-adv_pos[1])


# Pick wall direction after movement


def pick_wall_direction(my_pos, adv_pos, chess_board, max_step):
    adv_posx, adv_posy = adv_pos
    my_posx, my_posy = my_pos
    # print("My Position: ", my_pos)
    # print("Adv Position: ", adv_pos)
    # print("Max Step: ", max_step)
    allowed_barriers = [i for i in range(
        0, 4) if not chess_board[my_posx, my_posy, i]]
    # UP AND DOWN
    # print("Allowed Barriers: ", allowed_barriers)
    barrier_opp_move_number_list = []
    for barrier in allowed_barriers:
        copy_chess_board = deepcopy(chess_board)
        copy_chess_board[my_posx, my_posy, barrier] = True
        next_barrier_and_move_num = (barrier, get_num_possible_op_moves(
            my_pos, adv_pos, copy_chess_board, max_step))
        barrier_opp_move_number_list.append(next_barrier_and_move_num)
    # print("Barrier Op Move Num List: ", barrier_opp_move_number_list)

    if (barrier_opp_move_number_list):
        sorted_barrier_opp_move_number_list = sorted(
            barrier_opp_move_number_list, key=lambda x: x[1])
        dir = sorted_barrier_opp_move_number_list[0][0]
    else:
        dir = allowed_barriers[np.random.randint(0, len(allowed_barriers))]
        # for direction in allowed_barriers:
        #     if (not chess_board[adv_posx, adv_posy, direction]):
        #         dir = direction
    return dir
