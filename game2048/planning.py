import numpy as np
import copy


def nextMove(board):
    m,s= nextMoveRecur(board,3,3)
    return m

def nextMoveRecur(board, depth, max_depth):
    bestScore = -1
    bestMove = 0
    for i in range(0,4):
        newBoard = copy.deepcopy(board)
        newBoard = move(newBoard,i)

        score = Score(newBoard)
        if depth != 0:
            my_m,my_s = nextMoveRecur(newBoard,depth - 1,max_depth)
            score += my_s*pow(0.9,max_depth - depth + 1)

        if(score > bestScore):
            bestMove = i
            bestScore = score

    return (bestMove,bestScore);

def Score(board_before):
    commonRatio = 0.2
    weight = 10.0
    linearWeightedVal = 0
    flag = 0

    for direction in range(4):

        board = np.rot90(board_before,direction)
        tmp_value = 0
        for x in range(4):
            if flag == 0:
                for y in range(4):
                    tmp_value = tmp_value + board[x,y]*weight
                    weight = weight*commonRatio
                    flag = 1
            else:
                for y in range(4):
                    tmp_value = tmp_value + board[x,3-y]*weight
                    weight = weight*commonRatio
                    flag = 0

        if(tmp_value >= linearWeightedVal):
            linearWeightedVal = tmp_value
        return linearWeightedVal

def move(board, direction):
    board_to_left = np.rot90(board, -direction)
    for row in range(4):
            core = merge(board_to_left[row])
            board_to_left[row, :len(core)] = core
            board_to_left[row, len(core):] = 0
    board = np.rot90(board_to_left, direction)

    where_empty = list(zip(*np.where(board == 0)))
    if where_empty:
            selected = where_empty[np.random.randint(0, len(where_empty))]
            
    return board

def merge(row):
    '''merge the row, there may be some improvement'''
    non_zero = row[row != 0]  # remove zeros
    core = [None]
    for elem in non_zero:
        if core[-1] is None:
            core[-1] = elem
        elif core[-1] == elem:
            core[-1] = 2 * elem
            core.append(None)
        else:
            core.append(elem)
    if core[-1] is None:
        core.pop()
    return core

