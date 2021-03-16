from typing import Iterable, Iterator, Optional, Tuple, Dict
from copy import deepcopy
from collections import Counter

Bitboard = int
PieceType = int
Colour = int
Square = int


SQUARES = [
    a8, b8, c8, d8, e8, f8, g8, h8,
    a7, b7, c7, d7, e7, f7, g7, h7,
    a6, b6, c6, d6, e6, f6, g6, h6,
    a5, b5, c5, d5, e5, f5, g5, h5,
    a4, b4, c4, d4, e4, f4, g4, h4,
    a3, b3, c3, d3, e3, f3, g3, h3,
    a2, b2, c2, d2, e2, f2, g2, h2,
    a1, b1, c1, d1, e1, f1, g1, h1,
] = range(64)

SQUARE_NAMES = [
    'a8', 'b8', 'c8', 'd8', 'e8', 'f8', 'g8', 'h8',
    'a7', 'b7', 'c7', 'd7', 'e7', 'f7', 'g7', 'h7',
    'a6', 'b6', 'c6', 'd6', 'e6', 'f6', 'g6', 'h6',
    'a5', 'b5', 'c5', 'd5', 'e5', 'f5', 'g5', 'h5',
    'a4', 'b4', 'c4', 'd4', 'e4', 'f4', 'g4', 'h4',
    'a3', 'b3', 'c3', 'd3', 'e3', 'f3', 'g3', 'h3',
    'a2', 'b2', 'c2', 'd2', 'e2', 'f2', 'g2', 'h2',
    'a1', 'b1', 'c1', 'd1', 'e1', 'f1', 'g1', 'h1',
]

BB_SQUARES = [1 << y for y in range(64)]

UNIVERSAL_BB = 0xffff_ffff_ffff_ffff
EMPTY_BB = 0
BB_1 = 0xff00_0000_0000_0000
BB_2 = 0xff_0000_0000_0000
BB_7 = 0xff00
BB_8 = 0xff
BB_CORNERS = 0x8100_0000_0000_0081

PIECE_TYPES = [KING, PAWN, ROOK, BISHOP, KNIGHT, QUEEN] = range(6)
SLIDING_PIECE_TYPES = [ROOK, BISHOP, QUEEN]
STEPPING_PIECE_TYPES = [KING, KNIGHT]
PROMOTION_PIECE_TYPES = [ROOK, BISHOP, KNIGHT, QUEEN]
MOVE_DIAGONALLY = [BISHOP, QUEEN]
MOVE_STRAIGHT = [ROOK, QUEEN]
COLOURS = [WHITE, BLACK] = range(2)

PIECE_NOTATIONS = ['k', 'p', 'r', 'b', 'n', 'q']


def to_uci(sq: Square) -> str:
    return SQUARE_NAMES[sq]


def str_bitboard(bb: Bitboard) -> str:
    st = bin(bb)[2:]
    if len(st) < 64:
        st = '0' * (64 - len(st)) + st
    res = ''
    for i, ch in enumerate(st[::-1]):
        res += ch
        if (i + 1) % 8 == 0:
            res += '\n'
    return res


def lsb(bb: Bitboard) -> Square:
    return (bb & -bb).bit_length() - 1


def shift_bit(bb: Bitboard, n) -> Bitboard:
    if n >= 0:
        return bb << n
    return bb >> -n


def bb_iterator(bb: Bitboard) -> Iterator[Square]:
    next_bit = lsb(bb)
    while next_bit != -1:
        yield next_bit
        bb ^= (1 << next_bit)
        next_bit = lsb(bb)


def distance(sq1: Square, sq2: Square) -> int:
    return max(abs(sq1 // 8 - sq2 // 8), abs(sq1 % 8 - sq2 % 8))


def horizontal_distance(sq1: Square, sq2: Square) -> int:
    return abs(sq1 % 8 - sq2 % 8)


def vertical_distance(sq1: Square, sq2: Square) -> int:
    return abs(sq1 // 8 - sq2 // 8)


def same_diagonal(sq1: Square, sq2: Square) -> bool:
    return horizontal_distance(sq1, sq2) == vertical_distance(sq1, sq2)


def same_line(sq1: Square, sq2: Square) -> bool:
    return horizontal_distance(sq1, sq2) == 0 or vertical_distance(sq1, sq2) == 0


def squares_between(sq1: Square, sq2: Square) -> Bitboard:
    """
    Bitboard of squares between two squares that are either
    on the same line or on the same diagonal
    """
    if not same_line(sq1, sq2) and not same_diagonal(sq1, sq2):
        return BB_SQUARES[sq1] | BB_SQUARES[sq2]
    dist = distance(sq1, sq2)
    if sq1 > sq2:
        sq1, sq2 = sq2, sq1
    dif = sq2 - sq1
    step = dif // dist
    bb = EMPTY_BB
    while sq1 <= sq2:
        bb |= BB_SQUARES[sq1]
        sq1 += step
    return bb


def long_attacks(square: Square, deltas: Iterable[int], occupied: Bitboard) -> Bitboard:
    res = EMPTY_BB
    for delta in deltas:
        cur_square = square + delta
        while 0 <= cur_square <= 63:
            if distance(cur_square, cur_square - delta) > 2:
                break
            res |= BB_SQUARES[cur_square]
            if BB_SQUARES[cur_square] & occupied:
                break
            cur_square += delta
    return res


def short_attacks(square: Square, deltas: Iterable[int]) -> Bitboard:
    return long_attacks(square, deltas, UNIVERSAL_BB)


KING_DELTAS = [-9, -8, -7, -1, 1, 7, 8, 9]
PAWN_DELTAS = [[-9, -7], [7, 9]]
ROOK_DELTAS = [-8, -1, 1, 8]
BISHOP_DELTAS = [-9, -7, 7, 9]
KNIGHT_DELTAS = [-17, -15, -10, -6, 6, 10, 15, 17]
QUEEN_DELTAS = KING_DELTAS

DELTAS = [KING_DELTAS, PAWN_DELTAS, ROOK_DELTAS,
          BISHOP_DELTAS, KNIGHT_DELTAS, QUEEN_DELTAS]


class Piece:
    def __init__(self, piece_type: PieceType, colour: Colour):
        self.piece_type = piece_type
        self.colour = colour


class Move:
    def __init__(self, start: Square, dest: Square,
                 promotion_piece: PieceType = None):
        self.start = start
        self.dest = dest
        self.promotion_piece = promotion_piece

    def __eq__(self, other):
        return (self.start == other.start
                and self.dest == other.dest
                and self.promotion_piece == other.promotion_piece)

    def __repr__(self):
        return f'Move({to_uci(self.start)}, {to_uci(self.dest)}, {self.promotion_piece})'


class Board:
    def __init__(self, fen: str = 'rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1') -> None:
        self.pawns_bb = EMPTY_BB
        self.kings_bb = EMPTY_BB
        self.rooks_bb = EMPTY_BB
        self.knights_bb = EMPTY_BB
        self.bishops_bb = EMPTY_BB
        self.queens_bb = EMPTY_BB
        self.colour_bbs = [None, None]
        self.colour_bbs[WHITE] = EMPTY_BB
        self.colour_bbs[BLACK] = EMPTY_BB
        self.occupied_bb = EMPTY_BB
        self.en_passant_bb = EMPTY_BB
        self.turn = None
        self.castling_bb = EMPTY_BB
        self.result = '-'
        self.halfmove_clock = 0
        self.positions = Counter()

        self.set_up_board(fen)

    def shortcut(self):
        return (self.pawns_bb, self.rooks_bb, self.knights_bb,
                self.bishops_bb, self.kings_bb, self.queens_bb,
                self.colour_bbs[0], self.colour_bbs[1], self.en_passant_bb)

    def __iter__(self):
        for bb, n in zip([self.pawns_bb, self.kings_bb, self.rooks_bb,
                          self.knights_bb, self.bishops_bb, self.queens_bb],
                         ['p', 'k', 'r', 'n', 'b', 'q']):
            for colour in COLOURS:
                sq_iterator = bb_iterator(bb & self.colour_bbs[colour])
                for sq in sq_iterator:
                    notation = n
                    if colour == WHITE:
                        notation = notation.upper()
                    yield sq, notation

    def __repr__(self):
        res = ['*'] * 64
        for piece_type in PIECE_TYPES:
            for square in bb_iterator(self.get_bitboard(piece_type)):
                res[square] = PIECE_NOTATIONS[piece_type]
        for square in bb_iterator(self.colour_bbs[WHITE]):
            res[square] = res[square].upper()
        res = [' '.join(res[8 * i: 8 * i + 8]) for i in range(0, 8)]
        return '\n'.join(res)

    def set_up_board(self, fen: str) -> None:
        board, turn, castling, en_passant, *_ = fen.split()
        cur_sq = 0
        for ch in board:
            if ch == '/':
                continue
            if ch.isdigit():
                cur_sq += int(ch)
            else:
                self.place_piece(ch.lower(), ch.islower(), cur_sq)
                cur_sq += 1

        self.turn = WHITE if turn == 'w' else BLACK

        self.castling_bb = EMPTY_BB
        if 'Q' in castling:
            self.castling_bb |= BB_SQUARES[a1]
        if 'K' in castling:
            self.castling_bb |= BB_SQUARES[h1]
        if 'q' in castling:
            self.castling_bb |= BB_SQUARES[a8]
        if 'K' in castling:
            self.castling_bb |= BB_SQUARES[h1]

        if en_passant != '-':
            self.en_passant_bb = BB_SQUARES[SQUARE_NAMES.index(en_passant)]

        self.positions[str(self)] += 1

    def place_piece(self, notation: str, colour: Colour, square: Square) -> None:
        piece_type = PIECE_NOTATIONS.index(notation)
        if piece_type == PAWN:
            self.pawns_bb |= BB_SQUARES[square]
        elif piece_type == KING:
            self.kings_bb |= BB_SQUARES[square]
        elif piece_type == ROOK:
            self.rooks_bb |= BB_SQUARES[square]
        elif piece_type == KNIGHT:
            self.knights_bb |= BB_SQUARES[square]
        elif piece_type == BISHOP:
            self.bishops_bb |= BB_SQUARES[square]
        elif piece_type == QUEEN:
            self.queens_bb |= BB_SQUARES[square]

        self.colour_bbs[colour] |= BB_SQUARES[square]
        self.occupied_bb |= BB_SQUARES[square]

    def get_bitboard(self, piece_type: PieceType) -> Bitboard:
        if piece_type == PAWN:
            return self.pawns_bb
        elif piece_type == KING:
            return self.kings_bb
        elif piece_type == ROOK:
            return self.rooks_bb
        elif piece_type == KNIGHT:
            return self.knights_bb
        elif piece_type == BISHOP:
            return self.bishops_bb
        elif piece_type == QUEEN:
            return self.queens_bb
        else:
            raise ValueError(f'{piece_type} is not a known piece')

    def remove_piece(self, remove_bb: Bitboard) -> None:
        self.pawns_bb &= ~remove_bb
        self.kings_bb &= ~remove_bb
        self.rooks_bb &= ~remove_bb
        self.knights_bb &= ~remove_bb
        self.bishops_bb &= ~remove_bb
        self.queens_bb &= ~remove_bb
        for colour in COLOURS:
            self.colour_bbs[colour] &= ~remove_bb
        self.occupied_bb &= ~remove_bb

    def add_piece(self, add_bb: Bitboard, piece_type: PieceType,
                  colour: Colour) -> None:
        if piece_type == PAWN:
            self.pawns_bb |= add_bb
        elif piece_type == KING:
            self.kings_bb |= add_bb
        elif piece_type == ROOK:
            self.rooks_bb |= add_bb
        elif piece_type == KNIGHT:
            self.knights_bb |= add_bb
        elif piece_type == BISHOP:
            self.bishops_bb |= add_bb
        elif piece_type == QUEEN:
            self.queens_bb |= add_bb
        else:
            raise ValueError(f'{piece_type} is not a known piece')
        self.colour_bbs[colour] |= add_bb
        self.occupied_bb |= add_bb

    def get_piece_type_at(self, square: Square) -> PieceType:
        bb = BB_SQUARES[square]
        if self.pawns_bb & bb:
            return PAWN
        if self.kings_bb & bb:
            return KING
        if self.rooks_bb & bb:
            return ROOK
        if self.knights_bb & bb:
            return KNIGHT
        if self.bishops_bb & bb:
            return BISHOP
        if self.queens_bb & bb:
            return QUEEN
        raise ValueError(f'No piece at {to_uci(square)}')

    def get_king(self, colour: Colour) -> Bitboard:
        return self.kings_bb & self.colour_bbs[colour]

    def is_attacked(self, square: Square, colour: Optional[Colour] = None) -> bool:
        if colour is None:
            colour = self.turn
        king_bb = self.get_king(colour)
        self.remove_piece(king_bb)

        for piece_type in SLIDING_PIECE_TYPES:
            if (self.get_bitboard(piece_type)
                    & self.colour_bbs[1-colour]
                    & long_attacks(square, DELTAS[piece_type], self.occupied_bb)) != 0:
                self.add_piece(king_bb, KING, colour)
                return True
        for piece_type in STEPPING_PIECE_TYPES:
            if (self.get_bitboard(piece_type)
                    & self.colour_bbs[1-colour]
                    & short_attacks(square, DELTAS[piece_type])) != 0:
                self.add_piece(king_bb, KING, colour)
                return True
        if (self.get_bitboard(PAWN)
                & self.colour_bbs[1 - colour]
                & short_attacks(square, DELTAS[PAWN][colour])) != 0:
            self.add_piece(king_bb, KING, colour)
            return True
        self.add_piece(king_bb, KING, colour)
        return False

    def is_in_check(self) -> bool:
        king_bb = self.get_king(self.turn)
        return self.is_attacked(lsb(king_bb))

    def is_position_legal(self) -> bool:
        enemy_king_bb = self.get_king(1-self.turn)
        return not self.is_attacked(lsb(enemy_king_bb), 1-self.turn)

    def piece_to_block(self) -> Optional[Square]:
        """
        assuming that we are in check,
        return piece that is checking us
        if there is only one,
        return None otherwise
        """
        king_bb = self.get_king(self.turn)
        king_square = lsb(king_bb)
        piece = None
        for piece_type in SLIDING_PIECE_TYPES:
            bb = (self.get_bitboard(piece_type)
                  & self.colour_bbs[1-self.turn]
                  & long_attacks(king_square, DELTAS[piece_type], self.occupied_bb))
            if bb != 0:
                if piece is None:
                    if len(list(bb_iterator(bb))) > 1:
                        return None
                    piece = list(bb_iterator(bb))[0]
                else:
                    return None

        for piece_type in STEPPING_PIECE_TYPES:
            bb = (self.get_bitboard(piece_type)
                  & self.colour_bbs[1-self.turn]
                  & short_attacks(king_square, DELTAS[piece_type]))
            if bb != 0:
                if piece is None:
                    if len(list(bb_iterator(bb))) > 1:
                        return None
                    piece = list(bb_iterator(bb))[0]
                else:
                    return None

        bb = (self.get_bitboard(PAWN)
              & self.colour_bbs[1 - self.turn]
              & short_attacks(king_square, DELTAS[PAWN][self.turn]))
        if bb != 0:
            if piece is None:
                if len(list(bb_iterator(bb))) > 1:
                    return None
                piece = list(bb_iterator(bb))[0]
            else:
                return None
        return piece

    def get_pieces(self, piece_type: PieceType, colour: Colour) -> Bitboard:
        colour_bb = self.colour_bbs[colour]
        if piece_type == KING:
            return self.kings_bb & colour_bb
        if piece_type == PAWN:
            return self.pawns_bb & colour_bb
        if piece_type == ROOK:
            return self.rooks_bb & colour_bb
        if piece_type == KNIGHT:
            return self.knights_bb & colour_bb
        if piece_type == BISHOP:
            return self.bishops_bb & colour_bb
        if piece_type == QUEEN:
            return self.queens_bb & colour_bb

    def pseudo_legal_moves(self) -> Iterator[Move]:
        # rook, bishop and queen moves and captures
        for piece_type in SLIDING_PIECE_TYPES:
            piece_iterator = bb_iterator(self.get_pieces(
                piece_type, self.turn
            ))
            for start in piece_iterator:
                dest_iterator = bb_iterator(long_attacks(
                    start, DELTAS[piece_type], self.occupied_bb
                ) & ~self.colour_bbs[self.turn])
                for dest in dest_iterator:
                    yield Move(start, dest)

        # king and knight moves and captures
        for piece_type in STEPPING_PIECE_TYPES:
            piece_iterator = bb_iterator(self.get_pieces(
                piece_type, self.turn
            ))
            for start in piece_iterator:
                dest_iterator = bb_iterator(short_attacks(
                    start, DELTAS[piece_type]
                ) & ~self.colour_bbs[self.turn])
                for dest in dest_iterator:
                    yield Move(start, dest)

        # pawn moves
        piece_iterator = bb_iterator(self.get_pieces(
            PAWN, self.turn
        ))
        direction = -8 if self.turn == WHITE else 8
        for start in piece_iterator:
            if BB_SQUARES[start + direction] & self.occupied_bb == 0:
                if BB_SQUARES[start + direction] & (BB_1 | BB_8) != 0:
                    for piece_type in PROMOTION_PIECE_TYPES:
                        yield Move(start, start + direction, piece_type)
                else:
                    yield Move(start, start + direction)
                start_rank = BB_2 if self.turn == WHITE else BB_7
                if (BB_SQUARES[start] & start_rank != 0
                        and BB_SQUARES[start + 2 * direction] & self.occupied_bb == 0):
                    yield Move(start, start + 2 * direction)

        # pawn captures
        piece_iterator = bb_iterator(self.get_pieces(
            PAWN, self.turn
        ))
        for start in piece_iterator:
            if 0 <= start + direction + 1 < 64:
                if BB_SQUARES[start + direction + 1] & self.colour_bbs[1-self.turn] != 0:
                    if distance(lsb(BB_SQUARES[start + direction + 1]),
                                lsb(BB_SQUARES[start])) == 1:
                        if BB_SQUARES[start + direction + 1] & (BB_1 | BB_8) != 0:
                            for piece_type in PROMOTION_PIECE_TYPES:
                                yield Move(start, start + direction + 1, piece_type)
                        else:
                            yield Move(start, start + direction + 1)
            if 0 <= start + direction - 1 < 64:
                if BB_SQUARES[start + direction - 1] & self.colour_bbs[1-self.turn] != 0:
                    if distance(lsb(BB_SQUARES[start + direction - 1]),
                                lsb(BB_SQUARES[start])) == 1:
                        if BB_SQUARES[start + direction - 1] & (BB_1 | BB_8) != 0:
                            for piece_type in PROMOTION_PIECE_TYPES:
                                yield Move(start, start + direction - 1, piece_type)
                        else:
                            yield Move(start, start + direction - 1)

        # en passant
        if (shift_bit(self.en_passant_bb, -direction + 1)
                & self.get_pieces(PAWN, self.turn)) != 0:
            if distance(lsb(shift_bit(self.en_passant_bb, -direction + 1)),
                        lsb(self.en_passant_bb)) == 1:
                yield Move(lsb(self.en_passant_bb) - direction + 1, lsb(self.en_passant_bb))
        if (shift_bit(self.en_passant_bb, -direction - 1)
                & self.get_pieces(PAWN, self.turn)) != 0:
            if distance(lsb(shift_bit(self.en_passant_bb, -direction - 1)),
                        lsb(self.en_passant_bb)) == 1:
                yield Move(lsb(self.en_passant_bb) - direction - 1, lsb(self.en_passant_bb))

        # castle
        if self.turn == WHITE:
            if self.castling_bb & BB_SQUARES[h1] != 0:
                if (BB_SQUARES[f1] | BB_SQUARES[g1]) & self.occupied_bb == 0:
                    yield Move(e1, g1)
            if self.castling_bb & BB_SQUARES[a1] != 0:
                if (BB_SQUARES[b1] | BB_SQUARES[c1]
                        | BB_SQUARES[d1]) & self.occupied_bb == 0:
                    yield Move(e1, c1)
        else:
            if self.castling_bb & BB_SQUARES[h8] != 0:
                if (BB_SQUARES[f8] | BB_SQUARES[g8]) & self.occupied_bb == 0:
                    yield Move(e8, g8)
            if self.castling_bb & BB_SQUARES[a8] != 0:
                if (BB_SQUARES[b8] | BB_SQUARES[c8]
                        | BB_SQUARES[d8]) & self.occupied_bb == 0:
                    yield Move(e8, c8)

    def legal_moves(self) -> Iterator[Move]:
        if self.result != '-':
            return
        king_bb = self.get_king(self.turn)
        king_square = lsb(king_bb)
        start_iterator = bb_iterator(self.occupied_bb & self.colour_bbs[self.turn])
        masks = dict()
        for start in start_iterator:
            if king_square == start:
                continue
            masks[start] = UNIVERSAL_BB
            if same_diagonal(start, king_square) or same_line(start, king_square):
                sq_between = squares_between(start, king_square)
                if (sq_between ^ king_bb ^ BB_SQUARES[start]) & self.occupied_bb == 0:
                    dist = distance(king_square, start)
                    diff = start - lsb(king_bb)
                    step = diff // dist
                    cur = start + step
                    while 0 <= cur < 64 and distance(cur, cur-step) < 2:
                        if self.occupied_bb & BB_SQUARES[cur] != 0:
                            piece_bb = self.occupied_bb & BB_SQUARES[cur]
                            break
                        cur += step
                    else:
                        continue
                    if piece_bb & self.colour_bbs[self.turn] != 0:
                        continue
                    piece_type = self.get_piece_type_at(lsb(piece_bb))
                    if same_line(king_square, start):
                        if piece_type in MOVE_STRAIGHT:
                            masks[start] = squares_between(king_square, lsb(piece_bb))
                    else:
                        if piece_type in MOVE_DIAGONALLY:
                            masks[start] = squares_between(king_square, lsb(piece_bb))
            if (self.en_passant_bb != 0 and distance(lsb(self.en_passant_bb), start) == 1
                    and self.get_piece_type_at(start) == PAWN):
                board_copy = deepcopy(self)
                board_copy.make_move(start, lsb(self.en_passant_bb), check=False)
                if not board_copy.is_position_legal():
                    masks[start] &= ~ self.en_passant_bb

        check_mask = UNIVERSAL_BB
        king_can_castle = True
        if self.is_in_check():
            to_block = self.piece_to_block()
            if to_block is None:
                check_mask = EMPTY_BB
            else:
                check_mask = squares_between(king_square, to_block)
            king_can_castle = False
        #plm = sorted(self.pseudo_legal_moves(), key=lambda x: BB_SQUARES[x.dest] & self.occupied_bb == 0)
        for move in self.pseudo_legal_moves():
            if self.get_piece_type_at(move.start) == KING:
                if self.is_attacked(move.dest):
                    continue
                if distance(move.start, move.dest) > 1:
                    if not king_can_castle:
                        continue
                    if self.is_attacked((move.start + move.dest)//2):
                        continue
            else:
                if check_mask & masks[move.start] & BB_SQUARES[move.dest] == 0:
                    continue
            yield move

    def captures(self) -> Iterator[Move]:
        return (move for move in self.legal_moves() if
                (BB_SQUARES[move.dest] & self.occupied_bb != 0))

    def en_passants(self) -> Iterator[Move]:
        return (move for move in self.legal_moves() if
                (move.dest == lsb(self.en_passant_bb)
                 and self.get_piece_type_at(move.start) == PAWN))

    def any_legal_moves(self) -> bool:
        king_sq = lsb(self.get_king(self.turn))
        for sq in bb_iterator(short_attacks(king_sq, KING_DELTAS) & ~self.colour_bbs[self.turn]):
            if not self.is_attacked(sq):
                return True
        for _ in self.legal_moves():
            return True
        return False

    def get_result(self):
        if self.positions[str(self)] >= 3:
            return '1/2-1/2'
        if self.halfmove_clock >= 100:
            return '1/2-1/2'
        if self.any_legal_moves():
            return '-'
        if not self.is_in_check():
            return '1/2-1/2'
        if self.turn == WHITE:
            return '0-1'
        return '1-0'

    def make_move(self, start: Square, dest: Square,
                  promotion_piece: PieceType = None,
                  check: bool = True
                  ) -> Optional[Tuple[Tuple[Square], Tuple[Square],
                                      Tuple[PieceType], Colour]]:

        start_bb = BB_SQUARES[start]
        dest_bb = BB_SQUARES[dest]
        piece_type = self.get_piece_type_at(start)

        """CHANGE THIS PART LATER"""
        if promotion_piece is None:
            if piece_type == PAWN and dest_bb & (BB_1 | BB_8) != 0:
                promotion_piece = QUEEN
        """----------------------"""

        # make sure that the move is legal
        if check:
            if BB_SQUARES[start] & self.colour_bbs[self.turn] == 0:
                return
            move = Move(start, dest, promotion_piece)
            if move not in list(self.legal_moves()):
                return

        colour = self.turn

        # update halfmove clock and clear the positions cache if needed

        if (self.occupied_bb & dest_bb != 0
                or piece_type == PAWN):
            self.halfmove_clock = 0
            self.positions.clear()
        else:
            self.halfmove_clock += 1

        # remove corresponding castling rights
        self.castling_bb &= ~start_bb
        self.castling_bb &= ~dest_bb
        if piece_type == KING:
            row_bb = BB_1 if self.turn == WHITE else BB_8
            self.castling_bb &= ~row_bb

        if piece_type == KING and distance(start, dest) > 1:
            # castle
            if start > dest:
                # long
                remove = (start, dest - 2)
                place = (dest, dest + 1)
                self.remove_piece(BB_SQUARES[dest - 2])
                self.add_piece(BB_SQUARES[dest + 1], ROOK, colour)
            else:
                # short
                remove = (start, dest + 1)
                place = (dest, dest - 1)
                self.remove_piece(BB_SQUARES[dest + 1])
                self.add_piece(BB_SQUARES[dest - 1], ROOK, colour)
            self.remove_piece(start_bb)
            self.add_piece(dest_bb, KING, colour)
            pieces = (KING, ROOK)
        elif piece_type == PAWN and BB_SQUARES[dest] == self.en_passant_bb:
            # en passant
            direction = -8 if colour == WHITE else 8
            remove = (start, dest - direction)
            place = (dest,)
            pieces = (PAWN,)
            self.remove_piece(start_bb)
            self.remove_piece(BB_SQUARES[dest - direction])
            self.add_piece(dest_bb, PAWN, colour)
        elif piece_type == PAWN and dest_bb & (BB_1 | BB_8) != 0:
            # promotion
            remove = (start, dest)
            place = (dest,)
            pieces = (promotion_piece,)
            self.remove_piece(start_bb)
            self.remove_piece(dest_bb)
            self.add_piece(dest_bb, promotion_piece, colour)
        else:
            # regular move or capture
            remove = (start, dest)
            place = (dest,)
            pieces = (piece_type,)
            self.remove_piece(dest_bb)
            self.remove_piece(start_bb)
            self.add_piece(dest_bb, piece_type, colour)

        # add or remove en passant square
        if piece_type == PAWN and distance(start, dest) > 1:
            self.en_passant_bb = BB_SQUARES[(start + dest) // 2]
        else:
            self.en_passant_bb = EMPTY_BB

        self.turn = 1 - self.turn
        self.positions[str(self)] += 1

        self.result = self.get_result()
        return remove, place, pieces, colour
