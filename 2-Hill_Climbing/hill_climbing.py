# based on code from: https://github.com/aimacode/aima-python

import random


class NQueensProblem:
    """
    The problem of placing N queens on an NxN board with none attacking
    each other. A state is represented as an N-element array, where
    a value of r in the c-th entry means there is a queen at column c,
    row r.

    This class operates on complete state descriptions where all queens are
    on the board in each state (one in each column c). This is in contrast to
    the NQueensProblem implementation in aima-python, whose initial state has
    no queens on the board, and whose actions method generates all the valid
    positions to place a queen in the first free column.
    """

    def __init__(self, N=None, state=None):
        if N is None:
            N = len(state)
        if state is None:
            state = tuple(0 for _ in range(N))  # 初始状态：所有皇后放最低行
        assert N == len(state)
        self.N = N
        self.initial = state

    def actions(self, state: tuple) -> list:
        """Return a list containing all the valid actions for `state`.

        For each column c, one action is generated for each free row r in c,
        describing moving the queen in c from her current row to row r.

        This method does not take conflicts into account. It returns all
        actions which transform the current state into a neighbouring state.
        The neighbours of the current state are all states in which the
        position of exactly one queen is different. For example:
        (0, 0, 0, 0) and (0, 0, 0, 2) are neighbours, but
        (0, 0, 0, 0) and (0, 0, 1, 1) are not.

        Node.expand calls `result` with each action returned by `actions`.
        """
        ######################
        ### Your code here ###
        act_res = []
        for col in range(self.N):
            new_state = ()
            pri = tuple(state[i] for i in range(col))  # 0~col-1列的皇后位置不变
            sub = new_state + tuple(state[i] for i in range(col+1, self.N))  # col+1 ~ N列的皇后位置不变
            for row in range(self.N):
                if row != state[col]:  # 要移动的第col列皇后，不能在原位
                    new_state = pri + tuple([row]) + sub
                    act_res.append(new_state)
        ######################
        return act_res  # type(act_res) = list

    def result(self, state: tuple, action) -> tuple:
        """Return the result of applying `action` to `state`.

        Move the queen in the column specified by `action` to the row specified by `action`.
        Node.expand calls `result` on each action returned by `actions`.
        """
        ######################
        ### Your code here ###
        ######################
        return action

    def goal_test(self, state):
        """Check if all columns filled, no conflicts."""
        return self.value(state) == 0

    def value(self, state):
        """Return 0 minus the number of conflicts in `state`."""
        return -self.num_conflicts(state)

    def num_conflicts(self, state):
        """Return the number of conflicts in `state`."""
        num_conflicts = 0
        for (col1, row1) in enumerate(state):
            for (col2, row2) in enumerate(state):
                if (col1, row1) != (col2, row2):
                    num_conflicts += self.conflict(row1, col1, row2, col2)
        return num_conflicts

    def conflict(self, row1, col1, row2, col2):
        """Would putting two queens in (row1, col1) and (row2, col2) conflict?"""
        return (row1 == row2 or  # same row
                col1 == col2 or  # same column
                row1 - col1 == row2 - col2 or  # same \ diagonal
                row1 + col1 == row2 + col2)  # same / diagonal

    def random_state(self):
        """Return a new random n-queens state.

        Use this to implement hill_climbing_random_restart.
        """
        return tuple(random.choice(range(self.N)) for _ in range(self.N))  # "_"是临时变量名，用来让for-in执行N次


class Node:
    """
    A node in a search tree. Contains a pointer to the parent (the node
    that this is a successor of) and to the actual state for this node.
    Delegates problem specific functionality to self.problem.
    """

    def __init__(self, problem, state, parent=None, action=None):
        """Create a search tree Node, derived from a parent by an action."""
        self.problem = problem
        self.state = state
        self.parent = parent
        self.action = action
        self.depth = 0
        if parent:
            self.depth = parent.depth + 1

    def __repr__(self):
        return "<Node {}>".format(self.state)

    def __lt__(self, node):
        return self.state < node.state

    def __eq__(self, node):
        return self.state == node.state

    def value(self):
        return self.problem.value(self.state)

    def goal_test(self):
        return self.problem.goal_test(self.state)

    def expand(self):
        """List the nodes reachable from this node."""
        state = self.state
        problem = self.problem
        return [
            Node(
                state=problem.result(state, action),
                problem=problem,
                parent=self,
                action=action,
            )
            for action in problem.actions(state)
        ]

    def best_of(self, nodes):  # nodes是列表，每个元素是class Node
        """Return the best Node from a list of Nodes, based on problem.value.

        Sorting the nodes is not the best for runtime or search performance,
        but it ensures that the result is deterministic for the purpose of
        this assignment.
        """
        return max(
            sorted(nodes),
            key=lambda node: node.value(),
        )


def hill_climbing(problem):
    """
    [Figure 4.2] in the textbook.
    From the initial node, keep choosing the neighbor with highest value,
    stopping when no neighbor is better.
    """
    current = Node(problem=problem, state=problem.initial)
    while True:
        if current.goal_test():  # 如果当前节点对应的棋盘没有(皇后)冲突
            break  # 说明找到了N皇后的 1种可行解，退出
        # 若当前棋盘还有冲突，则探寻它的 所有的 1步移动
        neighbours = current.expand()  # type(neighbours) == 'list'
        if not neighbours:
            break
        neighbour = current.best_of(neighbours)  # 选择最优的``下一步''，使得棋局的冲突最小
        if neighbour.value() <= current.value():  # 如果新棋局的冲突比原棋局还大，则原棋局已是最优( value()内有*-1的取反操作 )
            break  # 退出
        current = neighbour  # ``当前棋局''从当前态迁移至``一步后''的新状态
    return current.state


def hill_climbing_instrumented(problem):
    """
    Find the same solution as `hill_climbing`, and return a dictionary
    recording the number of nodes expanded, and whether the problem was
    solved.
    """
    num_expanded = 0
    finally_solved = False
    current = Node(problem=problem, state=problem.initial)
    while True:
        if current.goal_test():  # 如果当前节点对应的棋盘没有(皇后)冲突
            finally_solved = True
            break  # 说明找到了N皇后的 1种可行解，退出
        # 若当前棋盘还有冲突，则探寻它的 所有的 1步移动
        neighbours = current.expand()  # type(neighbours) == 'list'
        if not neighbours:
            break
        neighbour = current.best_of(neighbours)  # 选择最优的``下一步''，使得棋局的冲突最小
        num_expanded += 1
        if neighbour.value() <= current.value():  # 如果新棋局的冲突比原棋局还大，则原棋局已是最优( value()内有*-1的取反操作 )
            break  # 退出
        current = neighbour  # ``当前棋局''从当前态迁移至``一步后''的新状态

    return {
        "expanded": num_expanded,
        "solved": finally_solved,
        "best_state": current.state,
    }


def hill_climbing_sideways(problem, max_sideways_moves=40):
    """
    When the search would terminate because the best neighbour doesn't
    have a higher value than the current state, continue the search if 
    the the best neighbour's value is equal to that of the current state.

    But don't do this more than `max_sideways_moves` times. Watch out for
    off by one errors, and don't forget to return early if the search finds
    a goal state.
    """
    ######################
    ### Your code here ###
    num_expanded = 0
    finally_solved = False
    num_side_moves = 0
    current = Node(problem=problem, state=problem.initial)
    while True:
        if current.goal_test():  # 如果当前节点对应的棋盘没有(皇后)冲突
            finally_solved = True
            break  # 说明找到了N皇后的 1种可行解，退出
        # 若当前棋盘还有冲突，则探寻它的 所有的 1步移动
        neighbours = current.expand()  # type(neighbours) == 'list'
        if not neighbours:
            break
        neighbour = current.best_of(neighbours)  # 选择最优的``下一步''，使得棋局的冲突最小
        num_expanded += 1
        if neighbour.value() < current.value():  # 如果新棋局的冲突比原棋局还大，则原棋局已是最优( value()内有*-1的取反操作 )
            break  # 退出
        elif neighbour.value() == current.value():  # 没有冲突数更小的棋局了，只有一样的
            if num_side_moves < max_sideways_moves:  # 若有侧移余额，则侧移
                num_side_moves += 1
            else:  # 否则，直接退出
                break
        current = neighbour  # ``当前棋局''从当前态迁移至``一步后''的新状态

    ######################
    return {
        "expanded": num_expanded,
        "solved": finally_solved,
        "best_state": current.state,
        "sideways_moves": num_side_moves,
    }


def generage_random_board(n):  # input: int n;  output: class Node
    new_board = []
    for col in range(n):
        row = random.randint(0, n-1)
        new_board[col] = row
    return Node(problem=NQueensProblem(N=n, state=tuple(new_board)), state=tuple(new_board))

def hill_climbing_random_restart(problem, max_restarts=40):
    """
    When the search would terminate because the best neighbour doesn't
    have a higher value than the current state, generate a new state to
    continue the search from (using problem.random_state).

    But don't do this more than `max_restarts` times. Watch out for
    off by one errors, and don't forget to return early if the search finds
    a goal state.

    To get consistent results each time, call random.seed(YOUR_FAVOURITE_SEED)
    before calling this function.
    """
    ######################
    ### Your code here ###
    num_expanded = 0
    finally_solved = False
    num_restarts = 0

    current = Node(problem=problem, state=problem.initial)
    while True:
        if current.goal_test():  # 如果当前节点对应的棋盘没有(皇后)冲突
            finally_solved = True
            break  # 说明找到了N皇后的 1种可行解，退出

        # 若当前棋盘还有冲突，则探寻它的 所有的 1步移动
        neighbours = current.expand()  # type(neighbours) == 'list'
        neighbour = current.best_of(neighbours)  # 选择最优的``下一步''，使得棋局的冲突最小
        num_expanded += 1
        if neighbour.value() < current.value():  # 如果新棋局的冲突比原棋局还大，则原棋局已是最优( value()内有*-1的取反操作 )
            break  # 退出
        elif neighbour.value() == current.value():  # 若冲突数相等，尝试重启
            if num_restarts < max_restarts:
                neighbour.state = problem.random_state()
                neighbour.problem.initial = neighbour.state
                num_restarts += 1
            else:
                break
        current = neighbour  # ``当前棋局''从当前态迁移至``一步后''的新状态

    ######################
    return {
        "expanded": num_expanded,
        "solved": finally_solved,
        "best_state": current.state,
        "restarts": num_restarts,
    }

