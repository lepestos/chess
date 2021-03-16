from collections import defaultdict, Counter
from itertools import product
from copy import deepcopy

VALID_SQUARES = {x for x in product(range(8), repeat=2)}


class Piece:
    valid_coordinates = {x for x in product(range(8), repeat=2)}

    def __init__(self, i, j, colour):
        self.i = i
        self.j = j
        self.colour = colour

    def __eq__(self, other):
        if not isinstance(other, self.__class__):
            return False
        if self.colour == other.colour and self.notation == other.notation:
            return True
        return False

    def __repr__(self):
        return f'{self.fen_notation()} at ({self.i}, {self.j})'

    def fen_notation(self):
        if self.colour == 1:
            return self.notation
        return self.notation.upper()

    def long_moves(self, board, directions):
        res = defaultdict(set)
        for x, y in directions:
            cur_i = self.i
            cur_j = self.j
            while True:
                cur_i += x
                cur_j += y
                if (cur_i, cur_j) not in self.valid_coordinates:
                    break
                if board[cur_i, cur_j] is not None:
                    if board[cur_i, cur_j].colour != self.colour:
                        res['captures'].add((cur_i, cur_j))
                    break
                res['shifts'].add((cur_i, cur_j))

        pin = self.pin_type(board)
        if pin != 0:
            moves = self.pin_movement(pin)
            for move_type in res:
                res[move_type].intersection_update(moves)

        check = board.under_check()
        if check != 0:
            if check == 1:
                return defaultdict()
            else:
                king = board[board.kings[self.colour]]
                interp = self.interposing_squares(king, check)
                for move_type in res:
                    res[move_type].intersection_update(interp)

        return res

    def short_moves(self, board, directions):
        res = defaultdict(set)
        for x, y in directions:
            if (self.i + x, self.j + y) in self.valid_coordinates:
                piece = board.get((self.i + x, self.j + y))
                if piece is None:
                    res['shifts'].add((self.i + x, self.j + y))
                else:
                    if piece.colour != self.colour:
                        res['captures'].add((self.i + x, self.j + y))
        return res

    def legal_moves(self, board):
        raise NotImplementedError

    def same_line(self, other):
        if self.i == other.i or self.j == other.j:
            return True
        return False

    def same_diagonal(self, other):
        if abs(self.i - other.i) == abs(self.j - other.j):
            return True
        return False

    def lined_up_with_king(self, board):
        """
        Return True if king of the same colour
        is on the same diagonal or line,
        False otherwise
        """
        king = board[board.kings[self.colour]]
        return self.same_line(king) or self.same_diagonal(king)

    def pin_type(self, board):
        """
        return 0 if the piece is not pinned,
        1 if the piece is pinned along the file,
        2 if it's pinned along the rank,
        3 - along the diagonal, parallel to the main one,
        4 - along the diagonal, parallel to the counterdiagonal
        """
        if not self.lined_up_with_king(board):
            return 0
        king = board[board.kings[self.colour]]
        squares_between = self.squares_between(king)
        for square in squares_between:
            if board[square] is not None:
                return 0
        i = self.i - king.i
        j = self.j - king.j
        if j != 0:
            i //= abs(j)
            j //= abs(j)
        else:
            j //= abs(i)
            i //= abs(i)

        if j == 0:
            potential_pin = 1
        elif i == 0:
            potential_pin = 2
        elif i == j:
            potential_pin = 3
        else:
            potential_pin = 4

        if potential_pin in [1, 2]:
            pinning_pieces = ['r', 'q']
        else:
            pinning_pieces = ['b', 'q']

        cur_i = self.i + i
        cur_j = self.j + j
        while (cur_i, cur_j) in VALID_SQUARES:
            piece = board[cur_i, cur_j]
            if piece is not None:
                if (piece.colour != self.colour
                        and piece.notation in pinning_pieces):
                    return potential_pin
                break
            cur_i += i
            cur_j += j
        return 0

    def pin_movement(self, pin_type):
        """
        return squares, which the movement
        is restricted to if the piece is pinned
        """
        if pin_type == 0:
            return VALID_SQUARES
        if pin_type == 1:
            return {(i, self.j) for i in range(8)}
        if pin_type == 2:
            return {(self.i, j) for j in range(8)}
        if pin_type == 3:
            res = set()
            cur_i = self.i - min(self.i, self.j)
            cur_j = self.j - min(self.i, self.j)
            while (cur_i, cur_j) in VALID_SQUARES:
                res.add((cur_i, cur_j))
                cur_i += 1
                cur_j += 1
            return res
        if pin_type == 4:
            res = set()
            cur_i = self.i - min(self.i, 7 - self.j)
            cur_j = self.j + min(self.i, 7 - self.j)
            while (cur_i, cur_j) in VALID_SQUARES:
                res.add((cur_i, cur_j))
                cur_i += 1
                cur_j -= 1
            return res

    def squares_between(self, other):
        res = set()
        if self.same_line(other) or self.same_diagonal(other):
            inc_i, inc_j = self.i - other.i, self.j - other.j
            if inc_i == 0 and inc_j == 0:
                return set()
            inc = inc_i or inc_j
            inc_i //= abs(inc)
            inc_j //= abs(inc)
            cur_i = other.i + inc_i
            cur_j = other.j + inc_j
            while cur_i != self.i or cur_j != self.j:
                res.add((cur_i, cur_j))
                cur_i += inc_i
                cur_j += inc_j
            return res
        return set()

    @staticmethod
    def interposing_squares(king, adversary):
        """
        Return all the squares between king and
        adversary, including the square of adversary
        """
        return {(adversary.i, adversary.j)}.union(
            king.squares_between(adversary)
        )


PROMOTIONS_PIECES = ['q', 'n', 'b', 'r']


class Pawn(Piece):
    notation = 'p'

    def legal_moves(self, board):
        res = defaultdict(set)
        direct = 1 if self.colour == 1 else -1
        start_row = 1 if self.colour == 1 else 6
        end_row = 7 if self.colour == 1 else 0
        if board[self.i + direct, self.j] is None:
            if self.i + direct == end_row:
                for notation in PROMOTIONS_PIECES:
                    res['promotions'].add(
                        ((self.i + direct, self.j), notation)
                    )
            else:
                res['shifts'].add((self.i + direct, self.j))
                if (self.i == start_row
                        and board[self.i + 2 * direct, self.j] is None):
                    res['shifts'].add((self.i + 2 * direct, self.j))
        for cap_dir in [-1, 1]:
            if (self.i + direct, self.j + cap_dir) in VALID_SQUARES:
                target = board.get((self.i + direct, self.j + cap_dir))
                if target is not None:
                    if target.colour != self.colour:
                        if self.i + direct == end_row:
                            for notation in PROMOTIONS_PIECES:
                                res['capture_promotions'].add(
                                    ((self.i + direct, self.j + cap_dir), notation)
                                )
                        else:
                            res['captures'].add(
                                (self.i + direct, self.j + cap_dir)
                            )
        if board.en_passant:
            if (self.i == board.en_passant[0] - direct
                    and (self.j == board.en_passant[1] - 1
                         or self.j == board.en_passant[1] + 1)):
                res['en_passants'].add(board.en_passant)

        pin = self.pin_type(board)
        if (pin == 0 and len(res['en_passants']) != 0
                and self.same_line(board[board.kings[self.colour]])):
            king = board[board.kings[self.colour]]
            if self.i == king.i:

                pawn_dir = 1 if self.colour == 1 else -1
                pawn = board[board.en_passant[0] - pawn_dir,
                             board.en_passant[1]]
                board[board.en_passant[0] - pawn_dir,
                      board.en_passant[1]] = None

                for square in self.squares_between(king):
                    if board[square] is not None:
                        break
                else:
                    j = self.j - king.j
                    j //= abs(j)
                    cur_i = self.i
                    cur_j = self.j + j
                    while (cur_i, cur_j) in VALID_SQUARES:
                        if board[cur_i, cur_j] is not None:
                            piece = board[cur_i, cur_j]
                            if (piece.colour != self.colour
                                    and piece.notation in ['r', 'q']):
                                res['en_passants'] = set()
                            break
                        cur_j += j

                board[board.en_passant[0] - pawn_dir,
                      board.en_passant[1]] = pawn

        if pin != 0:
            moves = self.pin_movement(pin)
            for move_type in res:
                if move_type in ['promotions', 'capture_promotions']:
                    res[move_type].intersection_update({
                        (move, prom) for move in moves for prom in PROMOTIONS_PIECES
                    })
                else:
                    res[move_type].intersection_update(moves)

        check = board.under_check()
        if check != 0:
            if check == 1:
                return defaultdict()
            else:
                king = board[board.kings[self.colour]]
                interp = self.interposing_squares(king, check)
                for move_type in res:
                    if move_type in ['promotions', 'capture_promotions']:
                        res[move_type].intersection_update({
                            (move, prom) for move in interp for prom in PROMOTIONS_PIECES
                        })
                    else:
                        res[move_type].intersection_update(interp)

        return res


class Rook(Piece):
    notation = 'r'
    directions = [
        (0, 1), (1, 0), (-1, 0), (0, -1)
    ]

    def legal_moves(self, board):
        return self.long_moves(board, self.directions)


class Bishop(Piece):
    notation = 'b'
    directions = [
        (i, j) for i, j in product([-1, 1], repeat=2)
    ]

    def legal_moves(self, board):
        return self.long_moves(board, self.directions)


class Queen(Piece):
    directions = Rook.directions + Bishop.directions
    notation = 'q'

    def legal_moves(self, board):
        return self.long_moves(board, self.directions)


class Knight(Piece):
    directions = [
        (i, (3-abs(i)) * k) for i in [-2, -1, 1, 2] for k in [-1, 1]
    ]
    notation = 'n'

    def legal_moves(self, board):
        if self.pin_type(board) != 0:
            return defaultdict()

        res = self.short_moves(board, self.directions)

        check = board.under_check()
        if check != 0:
            if check == 1:
                return defaultdict()
            else:
                king = board[board.kings[self.colour]]
                interp = self.interposing_squares(king, check)
                for move_type in res:
                    res[move_type].intersection_update(interp)

        return res


class King(Piece):
    directions = [
        (i, j) for i, j in product([-1, 0, 1], repeat=2) if i != 0 or j != 0
    ]
    notation = 'k'

    def legal_moves(self, board):
        res = self.short_moves(board, self.directions)
        k_notation = 'k' if self.colour == 1 else 'K'
        q_notation = 'q' if self.colour == 1 else 'Q'
        row = 0 if self.colour == 1 else 7
        if board.under_check() == 0:
            for notation in [k_notation, q_notation]:
                if notation in board.castling:
                    king = board[row, 4]
                    if king is None:
                        continue
                    rook_column = 7 if notation.lower() == 'k' else 0
                    route_columns = [5, 6] if notation.lower() == 'k' else [2, 3]
                    rook = board[row, rook_column]
                    if rook is None:
                        continue
                    for square in king.squares_between(rook):
                        if board[square] is not None:
                            break
                        if square[1] in route_columns:
                            if board.under_attack(square) != 0:
                                break
                    else:
                        type_ = 'short' if notation.lower() == 'k' else 'long'
                        res['castles'].add(type_)

        # temporarily remove the king from the board
        board[self.i, self.j] = None

        # remove squares, that are under attack
        for move_type in ['shifts', 'captures']:
            rem = set()
            for square in res[move_type]:
                if board.under_attack(square) != 0:
                    rem.add(square)
            res[move_type].difference_update(rem)

        # return the king back to the board
        board[self.i, self.j] = self

        return res


ch_to_cls = {
    'r': Rook,
    'n': Knight,
    'b': Bishop,
    'q': Queen,
    'k': King,
    'p': Pawn
}


class Board:
    def __init__(self, board_map, moves, turn, castling, en_passant, winner, kings, halfmove_clock, cache):
        self.board_map = board_map
        self.moves = moves
        self.turn = turn
        self.castling = castling
        self.en_passant = en_passant
        self.winner = winner
        self.kings = kings
        self.halfmove_clock = halfmove_clock
        self.cache = cache

    @classmethod
    def from_fen(cls, fen):
        # dictionary of piece names pointing to lists of piece objects
        # list of moves that have been played so far
        moves = []
        # map of the pieces currently placed on the board
        # pieces are named according to the FEN notation
        # empty fields are denoted as None
        board_map = [[None]*8 for _ in range(8)]
        cur_i = 0
        cur_j = 0
        board_fen, colour, castling, en_passant, halfmove_clock, move = fen.split()
        halfmove_clock = int(halfmove_clock)
        turn = 2 * int(move) if colour == 'w' else 2 * int(move) + 1
        castling = list(castling)
        en_passant = None if en_passant == '-'\
            else (8 - int(en_passant[1]), ord(en_passant[0]) - 97)
        winner = None
        kings = [None, None]
        cache = Counter()
        for ch in board_fen:
            if ch == '/':
                cur_i += 1
                cur_j = 0
            elif ch.isdigit():
                cur_j += int(ch)
            else:
                if ch.islower():
                    colour = 1
                else:
                    colour = 0
                piece = ch_to_cls[ch.lower()](cur_i, cur_j, colour)
                board_map[cur_i][cur_j] = piece
                if piece.notation == 'k':
                    kings[piece.colour] = (cur_i, cur_j)
                cur_j += 1
        return Board(board_map, moves, turn, castling, en_passant, winner, kings, halfmove_clock, cache)


    def __getitem__(self, key):
        if len(key) != 2:
            raise KeyError('Access board object with board[i, j]')
        x, y = key
        return self.board_map[x][y]

    def __setitem__(self, key, value):
        if len(key) != 2:
            raise KeyError('Access board object with board[i, j]')
        x, y = key
        self.board_map[x][y] = value

    def __repr__(self):
        st = ''
        for row in self.board_map:
            for el in row:
                if el is None:
                    st += '* '
                else:
                    st += el.fen_notation() + ' '
            st += '\n'
        return st

    def __iter__(self):
        for row in self.board_map:
            for piece in self.board_map:
                yield piece

    def get(self, key, default=None):
        try:
            return self[key]
        except IndexError:
            return default

    def typed_move(self, start, type_, move):
        """
        type_ is one of the following:
        shifts, captures, castles, promotions,
        capture_promotions, en_passants
        """
        if type_ == 'shifts':
            return Shift(board=self, start=start, dest=move)
        if type_ == 'captures':
            return Capture(board=self, start=start, dest=move)
        if type_ == 'castles':
            return Castle(board=self, start=start, type_=move)
        if type_ == 'promotions':
            return Promotion(
                board=self, start=start, dest=move[0], prom_piece=move[1]
            )
        if type_ == 'capture_promotions':
            return CapturePromotion(
                board=self, start=start, dest=move[0], prom_piece=move[1]
            )
        if type_ == 'en_passants':
            return EnPassant(board=self, start=start, dest=move)

        raise ValueError(f'Unknown type: {type_}')

    @staticmethod
    def to_coordinates(n):
        if isinstance(n, tuple):
            return n
        return int(n) // 8, int(n) % 8

    def make_move(self, start, dest, prom_piece=None):
        move = self.create_move(start, dest, prom_piece)
        if move == 'prom_piece':
            return 'prom_piece'
        if not move.is_legal() or self.turn % 2 != self[start].colour:
            return 'dispatch'
        self.commit_move(move)
        return 'move'

    def create_move(self, start, dest, prom_piece):
        piece_coord = start
        piece = self[piece_coord]
        if piece is None:
            raise ValueError(f'There is not piece at {piece_coord}')
        target_coord = dest
        target_piece = self[target_coord]
        if target_piece is None:
            # either shift, castle or promotion
            if piece.notation == 'k':
                if ((target_coord == (0, 6) and piece_coord == (0, 4) and piece.colour == 1)
                        or (target_coord == (7, 6) and piece_coord == (7, 4) and piece.colour == 0)):
                    return Castle(board=self, start=start, type_='short')
                if ((target_coord == (0, 2) and piece_coord == (0, 4) and piece.colour == 1)
                        or (target_coord == (7, 2) and piece_coord == (7, 4) and piece.colour == 0)):
                    return Castle(board=self, start=start, type_='long')
            elif piece.notation == 'p':
                if target_coord[0] == 7 or target_coord[0] == 0:
                    if prom_piece is None:
                        return 'prom_piece'
                    return Promotion(board=self, start=start,
                                     dest=target_coord, prom_piece=prom_piece)
                if target_coord == self.en_passant:
                    return EnPassant(board=self, start=start, dest=target_coord)
            return Shift(board=self, start=start, dest=target_coord)
        else:
            if piece.notation == 'p':
                if target_coord[0] == 7 or target_coord[0] == 0:
                    if prom_piece is None:
                        return 'prom_piece'
                    return CapturePromotion(board=self, start=start,
                                            dest=dest, prom_piece=prom_piece)
            return Capture(board=self, start=start, dest=dest)

    def commit_move(self, move):
        self.cache_position()
        move.make()
        self.moves.append(move)
        self.turn += 1
        if hasattr(move, 'enp_square'):
            self.en_passant = move.enp_square
        else:
            self.en_passant = None
        if self[move.dest].notation == 'k':
            self.remove_castling_rights(self.turn % 2 - 1, ['short', 'long'])
        elif self[move.dest].notation == 'r':
            row = 0 if (self.turn % 2 - 1) == 1 else 7
            if move.start[0] == row:
                if move.start[1] == 7:
                    self.remove_castling_rights(self[move.dest].colour, ['short'])
                elif move.start[1] == 0:
                    self.remove_castling_rights(self[move.dest].colour, ['long'])
        if move.dest in [(0, 0), (0, 7), (7, 0), (7, 7)]:
            if move.dest[1] == 7:
                self.remove_castling_rights(self.turn % 2, ['short'])
            elif move.dest[1] == 0:
                self.remove_castling_rights(self.turn % 2, ['long'])
        if (self[move.dest].notation != 'p'
                and not isinstance(move, Capture)
                and not isinstance(move, Promotion)
                and not isinstance(move, CapturePromotion)):
            self.halfmove_clock += 1
        else:
            self.halfmove_clock = 0
        self.winner = self.get_winner()

    def remove_castling_rights(self, colour, types):
        for type_ in types:
            notation = 'k' if type_ == 'short' else 'q'
            if colour == 0:
                notation = notation.upper()
            try:
                self.castling.remove(notation)
            except ValueError:
                pass

    def highlight(self, piece):
        legal_moves = piece.legal_moves(self)
        res = set()
        for legal_move in legal_moves['shifts']:
            res.add(legal_move)
        for legal_move in legal_moves['captures']:
            res.add(legal_move)
        for legal_move in legal_moves['castles']:
            row = 0 if piece.colour == 1 else 7
            column = 2 if legal_move == 'long' else 6
            res.add((row, column))
        for legal_move in legal_moves['promotions']:
            res.add(legal_move[0])
        for legal_move in legal_moves['capture_promotions']:
            res.add(legal_move[0])
        for legal_move in legal_moves['en_passants']:
            res.add(legal_move)
        return res

    def to_iter(self):
        for row in self.board_map:
            for piece in row:
                if piece is not None:
                    notation = piece.notation
                    if piece.colour == 0:
                        notation = notation.upper()
                    yield piece.j + 8 * piece.i, notation

    def compare(self, other):
        res = defaultdict(list)
        for i, (self_row, other_row) in enumerate(
                zip(self.board_map, other.board_map)):
            for j, (x, y) in enumerate(zip(self_row, other_row)):
                if x != y:
                    if y is not None:
                        res['remove'].append((j + 8*i, y.fen_notation()))
                    if x is not None:
                        res['put'].append((j + 8*i, x.fen_notation()))
        return res

    def under_attack(self, square):
        """
        Return 0 if the square is not under attack,
        return the piece it is attacked by if there
        is only one such piece, return 1 otherwise.
        """

        res = 0
        attacking_piece = None
        i, j = square
        e_colour = 1 - self.turn % 2

        # pawn, knight, king attacks
        e_pawn_dir = 1 if self.turn % 2 == 0 else -1
        target_squares_dict = {
            Pawn: [(i - e_pawn_dir, j + k) for k in [-1, 1]],
            Knight: [(i + x, j + y) for x, y in Knight.directions],
            King: [(i + x, j + y) for x, y in King.directions]
        }
        for piece_type, target_squares in target_squares_dict.items():
            for sq in target_squares:
                if sq not in VALID_SQUARES:
                    continue
                piece = self[sq]
                if (piece is not None and piece.notation == piece_type.notation
                        and piece.colour == e_colour):
                    res += 1
                    attacking_piece = piece
                    if res == 2:
                        return 1

        # bishop, rook and queen attacks
        dir_dict = {
            Bishop: ['b', 'q'],
            Rook: ['r', 'q']
        }
        for piece_type, not_list in dir_dict.items():
            for x, y in piece_type.directions:
                cur_i = i + x
                cur_j = j + y
                while (cur_i, cur_j) in VALID_SQUARES:
                    piece = self[cur_i, cur_j]
                    if piece is not None:
                        if (piece.colour == e_colour
                                and (piece.notation in not_list)):
                            res += 1
                            attacking_piece = piece
                            if res == 2:
                                return 1
                        break
                    cur_i += x
                    cur_j += y

        if res == 0:
            return 0
        return attacking_piece

    def under_check(self):
        """
        Return 0 if king is not under check,
        return 1 if king is double checked,
        return the checking piece otherwise
        """
        king = self[self.kings[self.turn % 2]]
        attack = self.under_attack((king.i, king.j))
        return attack

    def all_legal_moves(self):
        res = []
        for row in self.board_map:
            for piece in row:
                if piece is not None and piece.colour == self.turn % 2:
                    for type_, moves in piece.legal_moves(self).items():
                        for move in moves:
                            res.append(self.typed_move((piece.i, piece.j), type_, move))
        return sorted(res, key=lambda x: x.priority)

    def any_legal_moves(self):
        for row in self.board_map:
            for piece in row:
                if piece is not None and piece.colour == self.turn % 2:
                    for type_, moves in piece.legal_moves(self).items():
                        for move in moves:
                            return True
        return False

    def get_winner(self):
        if self.cache[self.position()] >= 2:
            return 2
        if self.halfmove_clock > 100 and self.any_legal_moves():
            return 2
        if self.any_legal_moves():
            return None
        if self.under_check():
            return 1 - self.turn % 2
        return 2

    def position(self):
        """
        fen-like representation
        """
        st = ''
        for row in self.board_map:
            st += '/'
            for piece in row:
                if piece is None:
                    if st[-1].isdigit():
                        n = int(st[-1])
                        st = st[:-1] + str(n)
                    else:
                        st += str(1)
                else:
                    st += piece.fen_notation()
        if self.turn % 2 == 0:
            st += ' w'
        else:
            st += ' b'
        return st

    def cache_position(self):
        self.cache[self.position()] += 1


class Move:
    def __init__(self, *args, **kwargs):
        self.board = kwargs['board']
        self.start = kwargs['start']

    def __repr__(self):
        piece = self.board[self.start]
        return f'{self.__class__} for {piece.fen_notation()} ' \
               f'from {self.start} to {self.dest}'

    def is_check(self):
        board = self.board
        board_copy = deepcopy(self.board)
        self.board = board_copy
        board_copy.commit_move(self)
        if board_copy.under_check() == 0:
            res = False
        else:
            res = True
        self.board = board
        return res


class Shift(Move):
    priority = 6

    def __init__(self, *args, **kwargs):
        self.dest = kwargs['dest']
        super().__init__(self, *args, **kwargs)

    def is_legal(self):
        piece = self.board[self.start]
        return self.dest in piece.legal_moves(self.board)['shifts']

    def make(self):
        piece = self.board[self.start]
        if piece.notation == 'p' and abs(piece.i - self.dest[0]) == 2:
            self.enp_square = ((piece.i + self.dest[0]) // 2, piece.j)
        self.board[piece.i, piece.j] = None
        self.board[self.dest[0], self.dest[1]] = piece
        piece.i = self.dest[0]
        piece.j = self.dest[1]
        if piece.notation == 'k':
            self.board.kings[piece.colour] = (piece.i, piece.j)


class Capture(Move):
    priority = 3

    def __init__(self, *args, **kwargs):
        self.dest = kwargs['dest']
        super().__init__(self, *args, **kwargs)

    def is_legal(self):
        piece = self.board[self.start]
        return self.dest in piece.legal_moves(self.board)['captures']

    def make(self):
        piece = self.board[self.start]
        self.board[self.start] = None
        self.board[self.dest] = piece
        piece.i = self.dest[0]
        piece.j = self.dest[1]
        if piece.notation == 'k':
            self.board.kings[piece.colour] = (piece.i, piece.j)


class Castle(Move):
    priority = 5

    def __init__(self, *args, **kwargs):
        # either 'long' or 'short'
        self.type_ = kwargs['type_']
        super().__init__(self, *args, **kwargs)
        if self.board.turn % 2 == 0:
            if self.type_ == 'short':
                self.dest = (7, 6)
            else:
                self.dest = (7, 2)
        else:
            if self.type_ == 'short':
                self.dest = (0, 6)
            else:
                self.dest = (0, 2)

    def __repr__(self):
        return f'{self.type_} castle for ' \
               f'{self.board[self.start].fen_notation()}'

    def is_legal(self):
        piece = self.board[self.start]
        return self.type_ in piece.legal_moves(self.board)['castles']

    def make(self):
        piece = self.board[self.start]
        row = 0 if piece.colour == 1 else 7
        k_column = 4
        k_dest_column = 2 if self.type_ == 'long' else 6
        r_column = 0 if self.type_ == 'long' else 7
        r_dest_column = 3 if self.type_ == 'long' else 5
        self.board[row, r_dest_column] = self.board[row, r_column]
        self.board[row, k_dest_column] = self.board[row, k_column]
        self.board[row, r_column] = None
        self.board[row, k_column] = None
        self.board[row, r_dest_column].j = r_dest_column
        self.board[row, k_dest_column].j = k_dest_column
        self.board.kings[piece.colour] = (piece.i, piece.j)
        self.dest = (row, k_dest_column)


class Promotion(Move):
    priority = 2
    piece_dict = {
        'q': Queen,
        'n': Knight,
        'b': Bishop,
        'r': Rook
    }

    def __init__(self, *args, **kwargs):
        self.prom_piece = kwargs['prom_piece']
        self.dest = kwargs['dest']
        super().__init__(self, *args, **kwargs)

    def is_legal(self):
        piece = self.board[self.start]
        return ((self.dest, self.prom_piece)
                in piece.legal_moves(self.board)['promotions'])

    def make(self):
        piece = self.board[self.start]
        self.board[piece.i, piece.j] = None
        new_piece = self.piece_dict[self.prom_piece](
            self.dest[0], self.dest[1], piece.colour
        )
        self.board[self.dest] = new_piece


class CapturePromotion(Promotion, Capture):
    priority = 1

    def is_legal(self):
        piece = self.board[self.start]
        return ((self.dest, self.prom_piece)
                in piece.legal_moves(self.board)['capture_promotions'])

    def make(self):
        piece = self.board[self.start]
        self.board[self.start] = None
        new_piece = self.piece_dict[self.prom_piece](
            self.dest[0], self.dest[1], piece.colour
        )
        self.board[self.dest] = new_piece


class EnPassant(Move):
    priority = 4


    def __init__(self, *args, **kwargs):
        self.dest = kwargs['dest']
        super().__init__(self, *args, **kwargs)

    def is_legal(self):
        piece = self.board[self.start]
        return self.dest in piece.legal_moves(self.board)['en_passants']

    def make(self):
        piece = self.board[self.start]
        self.board[self.start] = None
        self.board[self.dest[0], self.dest[1]] = piece
        piece.i = self.dest[0]
        piece.j = self.dest[1]
        direct = 1 if piece.colour == 1 else -1
        self.board[self.dest[0]-direct, self.dest[1]] = None
