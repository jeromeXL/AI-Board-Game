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
        if (get_distance(my_pos, adv_pos, chess_board) > max_step):
            heuristic_choice = "distance"
            # print("Current position: ", my_pos)
            # Queue of states of the form (pos, f_value(pos))
            state_queue = get_all_pos_reachable_by_priority(
                my_pos, max_step, chess_board, adv_pos, heuristic_choice)
            # If nothing in queue, no moves to make, game is lost
            if not state_queue:
                # pdb.set_trace()
                x, y = my_pos
                # Get allowed barriers, will only be one facing the opponent
                allowed_barriers = [i for i in range(
                    0, 4) if not chess_board[x, y, i]]
                # Return current position and barrier
                return my_pos, allowed_barriers[0]

            # Sort list of states in search in increasing order of f-value
            sorted_state_queue = sort_position_queue(state_queue)
            # Pick to move to position with lowest f-value
            my_pos = sorted_state_queue[0][0]
            x, y = my_pos
            # Get allowed directions for barrier placement
            allowed_barriers = [i for i in range(
                0, 4) if not chess_board[x, y, i]]
            # Sanity check, no way to be fully enclosed in a square, else game already ended
            assert len(allowed_barriers) >= 1
            # Get a random direction in which to place barrier
            dir = pick_wall_direction(my_pos, adv_pos, chess_board, max_step)
            # dir = allowed_barriers[np.random.randint(0, len(allowed_barriers))]
            time_taken = time.time() - start_time
            # print("My AI's turn took ", time_taken, "seconds.")
            # Return position to move to and direction to place barrier
            return my_pos, dir
        else:
            heuristic_choice = "num_escapes"
            state_queue = get_all_pos_reachable_by_priority(
                my_pos, max_step, chess_board, adv_pos, heuristic_choice)
            # If nothing in queue, no moves to make, game is lost
            if not state_queue:
                # pdb.set_trace()
                x, y = my_pos
                # Get allowed barriers, will only be one facing the opponent
                allowed_barriers = [i for i in range(
                    0, 4) if not chess_board[x, y, i]]
                # Return current position and barrier
                return my_pos, allowed_barriers[0]

            # Sort list of states in search in increasing order of f-value
            sorted_state_queue = sort_position_queue(state_queue)
            # Pick to move to position with lowest f-value
            my_pos = sorted_state_queue[0][0]
            x, y = my_pos
            # Get allowed directions for barrier placement
            allowed_barriers = [i for i in range(
                0, 4) if not chess_board[x, y, i]]
            # Sanity check, no way to be fully enclosed in a square, else game already ended
            assert len(allowed_barriers) >= 1
            # Get a random direction in which to place barrier
            dir = pick_wall_direction(my_pos, adv_pos, chess_board, max_step)
            # dir = allowed_barriers[np.random.randint(0, len(allowed_barriers))]
            time_taken = time.time() - start_time
            # print("My AI's turn took ", time_taken, "seconds.")
            # Return position to move to and direction to place barrier
            return my_pos, dir


# Compute heuristic of position to move to


def compute_heuristic(position, adv_pos, chess_board, max_step, heuristic_choice):
    # pdb.set_trace()
    if (heuristic_choice == "distance"):
        heuristic = get_distance(position, adv_pos, chess_board)
    elif (heuristic_choice == "num_escapes"):
        heuristic = get_num_escapes(position, adv_pos, chess_board, max_step)
    return heuristic


# Get num walls surrounding a position to move to


def get_num_escapes(position, adv_pos, chess_board, max_step):
    # Counter for num walls around position to move to
    x, y = position
    max_num_pos_visitable = 0
    allowed_barriers = [i for i in range(
        0, 4) if not chess_board[x, y, i]]
    for barrier in allowed_barriers:
        copy_chess_board = deepcopy(chess_board)
        copy_chess_board[x, y, barrier] = True
        num_pos_visitable = get_my_num_moves_from_pos(
            my_pos=position, chess_board=copy_chess_board, max_step=max_step, adv_pos=adv_pos)
        if (num_pos_visitable > max_num_pos_visitable):
            max_num_pos_visitable = num_pos_visitable
    return max_num_pos_visitable * -1


# Sort a queue of the form [position, f-value(position)] in increasing order


def sort_position_queue(queue):
    return sorted(queue, key=lambda x: x[1])

# Get actual number of steps between my_pos and adv_pos


def get_distance(my_pos, adv_pos, chess_board):
    state_queue = [(my_pos, 0)]
    visited = []
    moves = ((-1, 0), (0, 1), (1, 0), (0, -1))
    while state_queue:
        cur_pos, cur_step = state_queue.pop(0)
        r, c = cur_pos
        for dir, move in enumerate(moves):
            if chess_board[r, c, dir]:
                continue
            next_pos = tuple(np.array(cur_pos) + np.array(move))
            if next_pos == adv_pos:
                return cur_step
            if next_pos in visited:
                continue
            visited.append(next_pos)
            state_queue.append((next_pos, cur_step + 1))
    return 0

# Number of moves I can do when selecting a potential position to move to


def get_my_num_moves_from_pos(my_pos, chess_board, max_step, adv_pos):
    state_queue = [(my_pos, 0)]
    visited = []
    moves = ((-1, 0), (0, 1), (1, 0), (0, -1))
    while state_queue:
        cur_pos, cur_step = state_queue.pop(0)
        r, c = cur_pos
        if cur_step == max_step:
            break
        for dir, move in enumerate(moves):
            if chess_board[r, c, dir]:
                continue
            next_pos = tuple(np.array(cur_pos) + np.array(move))
            if next_pos == adv_pos or next_pos in visited:
                continue
            visited.append(next_pos)
            state_queue.append((next_pos, cur_step + 1))
    return len(visited)

# Number of moves opponent can do when selecting a potential position to move to


def get_opp_num_moves_from_pos(adv_pos, chess_board, max_step, my_pos):
    state_queue = [(adv_pos, 0)]
    visited = []
    moves = ((-1, 0), (0, 1), (1, 0), (0, -1))
    while state_queue:
        cur_pos, cur_step = state_queue.pop(0)
        r, c = cur_pos
        if cur_step == max_step:
            break
        for dir, move in enumerate(moves):
            if chess_board[r, c, dir]:
                continue
            next_pos = tuple(np.array(cur_pos) + np.array(move))
            if next_pos == my_pos or next_pos in visited:
                continue
            visited.append(next_pos)
            state_queue.append((next_pos, cur_step + 1))
    return len(visited)

# Pick wall direction after movement


def pick_wall_direction(my_pos, adv_pos, chess_board, max_step):
    my_posx, my_posy = my_pos
    # print("My Position: ", my_pos)
    # print("Adv Position: ", adv_pos)
    # print("Max Step: ", max_step)
    num_opp_moves = get_opp_num_moves_from_pos(
        adv_pos=adv_pos, chess_board=chess_board, max_step=max_step, my_pos=my_pos)
    num_my_moves = get_my_num_moves_from_pos(
        my_pos=my_pos, chess_board=chess_board, max_step=max_step, adv_pos=adv_pos)
    allowed_barriers = [i for i in range(
        0, 4) if not chess_board[my_posx, my_posy, i]]
    # print("Allowed Barriers: ", allowed_barriers)
    barrier_opp_move_number_list = []
    for barrier in allowed_barriers:
        copy_chess_board = deepcopy(chess_board)
        copy_chess_board[my_posx, my_posy, barrier] = True
        next_barrier_and_move_num = (barrier, get_opp_num_moves_from_pos(
            adv_pos=adv_pos, chess_board=copy_chess_board, max_step=max_step, my_pos=my_pos))
        barrier_opp_move_number_list.append(next_barrier_and_move_num)
    # print("Barrier Op Move Num List: ", barrier_opp_move_number_list)
    barrier_my_move_number_list = []
    for barrier in allowed_barriers:
        copy_chess_board = deepcopy(chess_board)
        copy_chess_board[my_posx, my_posy, barrier] = True
        next_barrier_and_move_num = (barrier, get_my_num_moves_from_pos(
            my_pos=my_pos, chess_board=copy_chess_board, max_step=max_step, adv_pos=adv_pos))
        barrier_my_move_number_list.append(next_barrier_and_move_num)
    sorted_barrier_opp_move_number_list = sorted(
        barrier_opp_move_number_list, key=lambda x: x[1])
    sorted_barrier_my_move_number_list = sorted(
        barrier_my_move_number_list, key=lambda x: x[1], reverse=True)
    num_opp_moves_new = sorted_barrier_opp_move_number_list[0][1]
    num_my_moves_new = sorted_barrier_my_move_number_list[0][1]
    diff_my_num_moves = abs(num_my_moves - num_my_moves_new)
    diff_opp_num_moves = abs(num_opp_moves - num_opp_moves_new)
    if (diff_my_num_moves > diff_opp_num_moves):
        dir = sorted_barrier_my_move_number_list[0][0]
    elif (diff_opp_num_moves > diff_my_num_moves):
        dir = sorted_barrier_opp_move_number_list[0][0]
    else:
        direction_list = [sorted_barrier_my_move_number_list[0]
                          [0], sorted_barrier_opp_move_number_list[0][0]]
        dir = np.random.choice(direction_list)
    return dir

# Obtain list of all positions that can be reached and compute their f value and store both the positions and f value into a list


def get_all_pos_reachable_by_priority(current_pos, max_step, chess_board, adv_pos, heuristic_choice):
    x, y = current_pos
    all_pos_reachable_by_priority = []
    advx, advy = adv_pos
    # Iterate through all positions reachable from current position within max steps
    if (heuristic_choice == "distance"):
        for i in range(x - max_step, x + max_step + 1):
            for j in range(y - max_step, y + max_step + 1):
                pos_to_move_to = (i, j)
                # Only add position to list of reachable positions along with its f value if it is reachable via BFS from current position within max steps
                if (position_is_reachable(current_pos=current_pos, position_to_reach=pos_to_move_to, chess_board=chess_board, adv_pos=adv_pos, max_step=max_step)):
                    # Create new list element of (position_to_move_to, f-value(position_to_move_to))
                    new_pos_and_f_value = (pos_to_move_to, get_f_value(
                        pos_to_move_to, adv_pos, chess_board, max_step, heuristic_choice))
                    all_pos_reachable_by_priority.append(new_pos_and_f_value)
        return all_pos_reachable_by_priority
    elif (heuristic_choice == "num_escapes"):
        for i in range(advx - max_step, advx + max_step + 1):
            for j in range(advy - max_step, advy + max_step + 1):
                pos_to_move_to = (i, j)
                # Only add position to list of reachable positions along with its f value if it is reachable via BFS from current position within max steps
                if (position_is_reachable(current_pos=current_pos, position_to_reach=pos_to_move_to, chess_board=chess_board, adv_pos=adv_pos, max_step=max_step)):
                    # Create new list element of (position_to_move_to, f-value(position_to_move_to))
                    new_pos_and_f_value = (pos_to_move_to, get_f_value(
                        pos_to_move_to, adv_pos, chess_board, max_step, heuristic_choice))
                    all_pos_reachable_by_priority.append(new_pos_and_f_value)
        return all_pos_reachable_by_priority

# Check if position is reachable from current position within max steps


def position_is_reachable(current_pos, position_to_reach, chess_board, adv_pos, max_step):
    # BFS
    moves = ((-1, 0), (0, 1), (1, 0), (0, -1))
    state_queue = [(current_pos, 0)]
    visited = [current_pos]
    is_reached = False
    while state_queue and not is_reached:
        cur_pos, cur_step = state_queue.pop(0)
        r, c = cur_pos
        if cur_step == max_step:
            break
        for dir, move in enumerate(moves):
            if chess_board[r, c, dir]:
                continue
            next_pos = tuple(np.array(cur_pos) + np.array(move))

            if next_pos == adv_pos or next_pos in visited:
                continue
            if next_pos == position_to_reach:
                is_reached = True
                break

            visited.append(next_pos)
            state_queue.append((next_pos, cur_step + 1))

    return is_reached

#  Get f value of a position based on heuristic


def get_f_value(position, adv_pos, chess_board, max_step, heuristic_choice):
    # Get heuristic value and return as f value
    heuristic_value = compute_heuristic(
        position, adv_pos, chess_board, max_step, heuristic_choice)
    return heuristic_value
