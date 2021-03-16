from copy import deepcopy

from django.test import TestCase

from .pieces import Board
from . import chessAI


class LegalMovesTest(TestCase):
    def test_depth_1(self):
        fen = 'rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1'
        board = Board(fen)
        self.assertEqual(len(list(board.legal_moves())), 20)
        fen = '4k3/8/8/8/8/4R3/8/4K3 b - - 0 1'
        board = Board(fen)
        self.assertEqual(len(list(board.legal_moves())), 4)
        fen = '8/8/5q2/K1Q4k/8/8/8/8 b - - 0 1'
        board = Board(fen)
        self.assertEqual(len(list(board.legal_moves())), 7)
        fen = '8/2p3k1/8/KP5r/8/8/8/8 w - - 0 1'
        board = Board(fen)
        self.assertEqual(len(list(board.legal_moves())), 3)
        fen = '8/8/4k3/8/P3K3/8/8/8 w - - 0 1'
        board = Board(fen)
        self.assertEqual(len(list(board.legal_moves())), 6)
        fen = '3k4/8/8/1Pp5/8/8/8/4K3 w - c6 0 1'
        board = Board(fen)
        self.assertEqual(len(list(board.legal_moves())), 7)

    def test_depth_2(self):
        board = Board()
        self.assertEqual(self.number_of_legal_moves(board, 2), 400)

    def test_depth_3(self):
        board = Board()
        self.assertEqual(self.number_of_legal_moves(board, 3), 8_902)

    def test_depth_4(self):
        board = Board()
        self.assertEqual(self.number_of_legal_moves(board, 4), 197_281)

    def test_depth_5(self):
        board = Board()
        self.assertEqual(self.number_of_legal_moves(board, 5), 4_865_609)

    def test_position_5(self):
        fen = 'rnbq1k1r/pp1Pbppp/2p5/8/2B5/8/PPP1NnPP/RNBQK2R w KQ - 1 8'
        board = Board(fen)
        self.assertEqual(self.number_of_legal_moves(board, 1), 44)
        board = Board(fen)
        self.assertEqual(self.number_of_legal_moves(board, 2), 1_486)
        board = Board(fen)
        self.assertEqual(self.number_of_legal_moves(board, 3), 62_379)
        board = Board(fen)
        self.assertEqual(self.number_of_legal_moves(board, 4), 2_103_487)
        #board = Board(fen)
        #self.assertEqual(self.number_of_legal_moves(board, 5), 89_941_194)

    def test_position_3(self):
        fen = '8/2p5/3p4/KP5r/1R3p1k/8/4P1P1/8 w - -'
        board = Board(fen)
        self.assertEqual(self.number_of_legal_moves(board, 1), 14)
        board = Board(fen)
        self.assertEqual(self.number_of_legal_moves(board, 2), 191)
        board = Board(fen)
        self.assertEqual(self.number_of_legal_moves(board, 3), 2812)
        board = Board(fen)
        self.assertEqual(self.number_of_legal_moves(board, 4), 43_238)
        #board = Board(fen)
        #self.assertEqual(self.number_of_en_passants(board, 5), 1165)
        #board = Board(fen)
        #self.assertEqual(self.number_of_captures(board, 5), 52_051)
        board = Board(fen)
        self.assertEqual(self.number_of_legal_moves(board, 5), 674_624)

    def number_of_legal_moves(self, board, depth):
        if depth == 0:
            return 0
        if depth == 1:
            return len(list(board.legal_moves()))
        res = 0
        for move in board.legal_moves():
            board_copy = deepcopy(board)
            board_copy.make_move(
                start=move.start, dest=move.dest,
                promotion_piece=move.promotion_piece,
                check=False
            )
            res += self.number_of_legal_moves(board_copy, depth-1)
        return res

    def number_of_captures(self, board, depth):
        if depth == 0:
            return 0
        if depth == 1:
            return len(list(board.captures()))
        res = 0
        for move in board.legal_moves():
            board_copy = deepcopy(board)
            board_copy.make_move(
                start=move.start, dest=move.dest,
                promotion_piece=move.promotion_piece,
                check=False
            )
            res += self.number_of_captures(board_copy, depth - 1)
        return res

    def number_of_en_passants(self, board, depth):
        if depth == 0:
            return 0
        if depth == 1:
            return len(list(board.en_passants()))
        res = 0
        for move in board.legal_moves():
            board_copy = deepcopy(board)
            board_copy.make_move(
                start=move.start, dest=move.dest,
                promotion_piece=move.promotion_piece,
                check=False
            )
            res += self.number_of_en_passants(board_copy, depth - 1)
        return res


class SpeedTest(TestCase):
    def test_board_string(self):
        board = Board()
        for i in range(100_000):
            str(board)

    def test_first_engine_move(self):
        board = Board()
        ai = chessAI.ProperAI(board, 0, 5)
        move = ai.make_move()

    def test_first_3_engine_moves(self):
        board = Board()
        ai_1 = chessAI.ProperAI(board, 0, 5)
        ai_2 = chessAI.ProperAI(board, 1, 5)
        move = ai_1.make_move()
        board.make_move(move.start, move.dest, move.promotion_piece)
        move = ai_2.make_move()
        board.make_move(move.start, move.dest, move.promotion_piece)
        move = ai_1.make_move()
        board.make_move(move.start, move.dest, move.promotion_piece)