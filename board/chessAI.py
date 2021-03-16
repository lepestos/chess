from random import choice
from time import sleep
from copy import deepcopy
from abc import ABC, abstractmethod

from . import pieces


PIECE_EVAL = {
    pieces.KNIGHT: 3,
    pieces.BISHOP: 3,
    pieces.ROOK: 5,
    pieces.QUEEN: 9,
    pieces.PAWN: 1,
    pieces.KING: 0
}

BOARD_MAP_EVAL_WHITE = {
    pieces.KNIGHT: [0.0, 0.1, 0.1, 0.2, 0.2, 0.1, 0.1, 0.0,
                    0.1, 0.1, 0.2, 0.3, 0.3, 0.2, 0.1, 0.1,
                    0.2, 0.3, 0.4, 0.5, 0.5, 0.4, 0.3, 0.2,
                    0.3, 0.5, 0.5, 0.6, 0.6, 0.5, 0.4, 0.3,
                    0.3, 0.4, 0.5, 0.6, 0.6, 0.5, 0.4, 0.3,
                    0.2, 0.3, 0.4, 0.5, 0.5, 0.4, 0.3, 0.2,
                    0.1, 0.1, 0.2, 0.3, 0.3, 0.2, 0.1, 0.1,
                    0.0, 0.1, 0.1, 0.2, 0.2, 0.1, 0.1, 0.0],
    pieces.BISHOP: [0.0, 0.1, 0.1, 0.2, 0.2, 0.1, 0.1, 0.0,
                    0.1, 0.1, 0.2, 0.3, 0.3, 0.2, 0.1, 0.1,
                    0.2, 0.3, 0.4, 0.5, 0.5, 0.4, 0.3, 0.2,
                    0.3, 0.5, 0.5, 0.6, 0.6, 0.5, 0.4, 0.3,
                    0.3, 0.4, 0.5, 0.6, 0.6, 0.5, 0.4, 0.3,
                    0.2, 0.3, 0.4, 0.5, 0.5, 0.4, 0.3, 0.2,
                    0.1, 0.1, 0.2, 0.3, 0.3, 0.2, 0.1, 0.1,
                    0.0, 0.1, 0.1, 0.2, 0.2, 0.1, 0.1, 0.0],
    pieces.PAWN: [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                  0.6, 0.8, 0.8, 0.9, 0.9, 0.8, 0.8, 0.6,
                  0.5, 0.7, 0.7, 0.8, 0.8, 0.7, 0.7, 0.5,
                  0.4, 0.6, 0.6, 0.7, 0.7, 0.6, 0.6, 0.4,
                  0.3, 0.4, 0.5, 0.6, 0.6, 0.5, 0.4, 0.3,
                  0.2, 0.3, 0.4, 0.5, 0.5, 0.4, 0.3, 0.2,
                  0.1, 0.1, 0.2, 0.3, 0.3, 0.2, 0.1, 0.1,
                  0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    pieces.QUEEN: [0.0, 0.1, 0.1, 0.2, 0.2, 0.1, 0.1, 0.0,
                   0.1, 0.1, 0.2, 0.3, 0.3, 0.2, 0.1, 0.1,
                   0.2, 0.3, 0.4, 0.5, 0.5, 0.4, 0.3, 0.2,
                   0.3, 0.5, 0.5, 0.6, 0.6, 0.5, 0.4, 0.3,
                   0.3, 0.4, 0.5, 0.6, 0.6, 0.5, 0.4, 0.3,
                   0.2, 0.3, 0.4, 0.5, 0.5, 0.4, 0.3, 0.2,
                   0.1, 0.1, 0.2, 0.3, 0.3, 0.2, 0.1, 0.1,
                   0.0, 0.1, 0.1, 0.2, 0.2, 0.1, 0.1, 0.0],
    pieces.ROOK: [0.0, 0.1, 0.1, 0.2, 0.2, 0.1, 0.1, 0.0,
                  0.1, 0.1, 0.2, 0.3, 0.3, 0.2, 0.1, 0.1,
                  0.2, 0.3, 0.4, 0.5, 0.5, 0.4, 0.3, 0.2,
                  0.3, 0.5, 0.5, 0.6, 0.6, 0.5, 0.4, 0.3,
                  0.3, 0.4, 0.5, 0.6, 0.6, 0.5, 0.4, 0.3,
                  0.2, 0.3, 0.4, 0.5, 0.5, 0.4, 0.3, 0.2,
                  0.1, 0.1, 0.2, 0.3, 0.3, 0.2, 0.1, 0.1,
                  0.0, 0.1, 0.1, 0.2, 0.2, 0.1, 0.1, 0.0],
    pieces.KING: [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                  0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                  0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                  0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                  0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                  0.1, 0.1, 0.1, 0.0, 0.0, 0.1, 0.1, 0.1,
                  0.6, 0.2, 0.1, 0.1, 0.1, 0.1, 0.2, 0.6,
                  0.9, 0.9, 0.6, 0.2, 0.2, 0.6, 0.9, 0.9],
}

BOARD_MAP_EVAL_BLACK = {
    piece: board_map[::-1] for piece, board_map in BOARD_MAP_EVAL_WHITE.items()
}


class ChessAI(ABC):
    def __init__(self, board, colour, depth=0):
        self.board = board
        self.colour = colour
        self.depth = depth
        self.cache = dict()

    @abstractmethod
    def make_move(self):
        pass


class RandomAI(ChessAI):
    def make_move(self):
        lm = list(self.board.legal_moves())
        move = choice(lm)
        return move


class RandomCaptureAI(ChessAI):
    def make_move(self):
        sleep(1)
        lm = list(self.board.legal_moves())
        captures = [m for m in lm if pieces.BB_SQUARES[m.dest]
                    & self.board.occupied_bb != 0]
        if captures:
            move = choice(captures)
        else:
            move = choice(lm)
        return move


class ProperAI(ChessAI):
    @staticmethod
    def evaluate(board):
        if board.result != '-':
            if board.result == '1/2-1/2':
                return 0
            if board.result == '1-0':
                return float("inf")
            return float("-inf")
        ev = 0
        for piece_type in pieces.PIECE_TYPES:
            bb = board.get_pieces(piece_type, pieces.WHITE)
            for square in pieces.bb_iterator(bb):
                ev += PIECE_EVAL[piece_type] + BOARD_MAP_EVAL_WHITE[piece_type][square]
            bb = board.get_pieces(piece_type, pieces.BLACK)
            for square in pieces.bb_iterator(bb):
                ev -= PIECE_EVAL[piece_type] + BOARD_MAP_EVAL_BLACK[piece_type][square]
        return ev

    def make_move(self):
        maximize = True if self.colour == 0 else False
        move = self.minimax(self.board, self.depth, maximize)
        return move

    def minimax(self, board, depth, maximize, alpha=float("-inf"), beta=float("inf"), max_depth=None):
        if max_depth is None:
            max_depth = depth
        if depth == 0:
            return self.evaluate(board)
        if board.result != '-':
            return self.result_to_evaluation(board.result)
        if maximize:
            max_eval = float("-inf")
            best_move = None
            lm = sorted(board.legal_moves(), key=lambda x: pieces.BB_SQUARES[x.dest] & board.occupied_bb == 0)
            for m in lm:
                board_copy = deepcopy(board)
                board_copy.make_move(m.start, m.dest, m.promotion_piece, check=False)
                st = board_copy.shortcut()
                ev = self.cache.get(st)
                if ev is None or ev[1] < depth - 1:
                    ev = self.minimax(board_copy, depth-1, False, alpha, beta, max_depth)
                    self.cache[st] = (ev, depth-1)
                else:
                    ev = ev[0]
                if ev > max_eval:
                    max_eval = ev
                    best_move = m
                if max_eval > alpha:
                    alpha = max_eval
                if beta <= alpha:
                    break
            if depth == max_depth:
                return best_move or choice(list(board.legal_moves()))
            return max_eval
        else:
            min_eval = float("inf")
            best_move = None
            lm = sorted(board.legal_moves(), key=lambda x: pieces.BB_SQUARES[x.dest] & board.occupied_bb == 0)
            for m in lm:
                board_copy = deepcopy(board)
                board_copy.make_move(m.start, m.dest, m.promotion_piece, check=False)
                st = board_copy.shortcut()
                ev = self.cache.get(st)
                if ev is None or ev[1] < depth - 1:
                    ev = self.minimax(board_copy, depth-1, True, alpha, beta, max_depth)
                    self.cache[st] = (ev, depth-1)
                else:
                    ev = ev[0]
                if ev < min_eval:
                    min_eval = ev
                    best_move = m
                if min_eval < beta:
                    beta = min_eval
                if beta <= alpha:
                    break
            if depth == max_depth:
                return best_move or choice(list(board.legal_moves()))
            return min_eval

    @staticmethod
    def result_to_evaluation(result):
        if result == '1/2-1/2':
            return 0
        if result == '1-0':
            return float("inf")
        if result == '0-1':
            return float("-inf")
        raise ValueError
