from collections import defaultdict
from .coordinates import Coordinates


class Piece:
    # what type of pin the piece is capable of,
    # either 'h' (for 'horizontal'),
    # or 'd' (for 'diagonal'), or both (or neither)
    pins = []

    def __init__(self, i, j, colour):
        self.i = i
        self.j = j
        self.colour = colour

    def go(self, board, dest):
        """
        Determine whether it's a simple move or a capture,
        and call the corresponding method.
        To be overwritten for King and Pawn,
        as there is a castle option for the former
        and promotion for the latter
        """
        if board.turn % 2 != self.colour or dest not in self.legal_moves(board):
            return 'revert'

        board.turn += 1
        board.en_passant = None

        if board[dest] is None:
            self.move(board, dest)
            return 'move'
        else:
            self.capture(board, dest)
            return 'capture'

    def legal_moves(self, board):
        raise NotImplementedError

    def moves_to_avoid_check(self, board, piece_coord):
        """
        'piece' is the piece our king is attacked by
        """
        k_notation = 'k' if self.colour == 1 else 'K'
        king = board.pieces[k_notation][0]
        res = self.interposing_squares(piece_coord, (king.i, king.j))
        return res

    @staticmethod
    def interposing_squares(enemy_square, king_square):
        k_i, k_j = king_square
        e_i, e_j = enemy_square
        dif_i, dif_j = e_i - k_i, e_j - k_j
        if dif_i != 0 and dif_j != 0 and abs(dif_i) != abs(dif_j):
            return {enemy_square}
        if dif_i != 0:
            dif_j //= abs(dif_i)
            dif_i //= abs(dif_i)
        else:
            dif_i //= abs(dif_j)
            dif_j //= abs(dif_j)
        cur_i = e_i
        cur_j = e_j
        res = set()
        while cur_i != k_i or cur_j != k_j:
            res.add((cur_i, cur_j))
            cur_i -= dif_i
            cur_j -= dif_j
        return res


    def move(self, board, dest):
        """
        Move from one location to another,
        no capture is going on.
        To be overwritten for King and Rook,
        as we need to make sure that we lose
        certain castling rights
        """
        board[dest] = board[self.i, self.j]
        board[self.i, self.j] = None
        i, j = dest
        self.i = i
        self.j = j

    def capture(self, board, dest):
        """
        Capture a piece.
        No need to overwrite this for King and Rook
        as self.move() is called here
        """
        cap_p_not = board[dest]
        captured_piece = board.find_piece(dest, cap_p_not)
        captured_piece.get_captured(board, cap_p_not)
        self.move(board, dest)

    def get_captured(self, board, notation):
        """
        Remove the piece from the board.
        To be overwritten for Rook as we need
        to make sure that certain castling rights
        are lost.
        Optionally can be overwritten for King as well,
        for the reason that kings can't really be captured
        """
        board[self.i, self.j] = None
        board.pieces[notation].remove(self)

    def long_moves(self, board, directions):
        res = set()
        i, j = self.i, self.j
        for dir_i, dir_j in directions:
            cur_i = i
            cur_j = j
            while True:
                cur_i += dir_i
                cur_j += dir_j
                if cur_i < 0 or cur_i > 7:
                    break
                if cur_j < 0 or cur_j > 7:
                    break
                if board[cur_i, cur_j] is not None:
                    if board[cur_i, cur_j].islower() != self.colour:
                        res.add((cur_i, cur_j))
                    break
                res.add((cur_i, cur_j))
        return res

    def short_moves(self, board, offsets_i, offsets_j,):
        i, j = self.i, self.j
        res = set()
        for off_i, off_j in zip(offsets_i, offsets_j):
            if (i + off_i > 7 or i + off_i < 0 or
                    j + off_j > 7 or j + off_j < 0):
                continue
            if board[i+off_i, j+off_j] is not None:
                if board[i+off_i, j+off_j].islower() != self.colour:
                    res.add((i+off_i, j+off_j))
            else:
                res.add((i + off_i, j + off_j))
        return res

    def pin_type(self, board):
        """
        Determine whether the piece is pinned to the king,
        and if it is, along which line:
        0: the piece is not pinned
        1: pinned along the file
        2: pinned along the rank
        3: pinned diagonally, parallel to the main diagonal
        4: pinned diagonally, parallel to the counterdiagonal
        """
        k_notation = 'k' if self.colour == 1 else 'K'
        king = board.pieces[k_notation][0]
        i_dif = self.i - king.i
        j_dif = self.j - king.j
        if i_dif == 0 and j_dif == 0:
            # king cannot be pinned to himself
            return 0
        if i_dif != 0 and j_dif != 0 and abs(i_dif) != abs(j_dif):
            return 0
        if i_dif != 0:
            j_dif //= abs(i_dif)
            i_dif //= abs(i_dif)
        else:
            i_dif //= abs(j_dif)
            j_dif //= abs(j_dif)
        # first piece behind
        king_notation = 'k' if self.colour == 1 else 'K'
        pb_n, *_ = self.first_piece_in_direction(board, (-i_dif, -j_dif))
        if pb_n != king_notation:
            return 0
        # first piece in front
        *_, pf = self.first_piece_in_direction(board, (i_dif, j_dif))
        if pf is None:
            return 0
        if pf.colour == self.colour:
            return 0
        potential_pin = 'd' if abs(i_dif) == abs(j_dif) else 'h'
        if potential_pin not in pf.pins:
            return 0
        # now we have determined, that the piece is in fact pinned,
        # so at this point we only need to find out the type of pin
        if j_dif == 0:
            return 1
        if i_dif == 0:
            return 2
        if i_dif == j_dif:
            return 3
        if i_dif == -j_dif:
            return 4
        raise ValueError(
            'Failed to determine whether the piece is pinned or not'
        )

    def first_piece_in_direction(self, board, direction):
        """
        direction = (i, j), where i, j is from {-1, 0, 1}
        and i and j aren't equal to 0 simultaneously.
        Return (notation, piece)
        """
        i, j = direction
        cur_i, cur_j = self.i + i, self.j + j
        while (cur_i in set(range(8))) and (cur_j in set(range(8))):
            if board[cur_i, cur_j] is not None:
                return board[cur_i, cur_j], board.find_piece((cur_i, cur_j))
            cur_i += i
            cur_j += j
        return None, None


class Pawn(Piece):
    def go(self, board, dest):
        promotion = (dest[0] == 0 or dest[0] == 7)

        if board.turn % 2 != self.colour or dest not in self.legal_moves(board):
            return 'revert'

        board.turn += 1

        if board[dest] is None:
            if promotion:
                #self.promote(board, dest)
                #board.en_passant = None
                #return 'promote'
                return 'choose promotion'
            elif board.en_passant == dest:
                self.en_passant(board, dest)
                board.en_passant = None
                return 'en passant'
            else:
                if abs(self.i - dest[0]) == 2:
                    board.en_passant = ((dest[0]+self.i)//2, self.j)
                else:
                    board.en_passant = None
                self.move(board, dest)
                return 'move'
        else:
            board.en_passant = None
            if promotion:
                #self.capture_and_promote(board, dest)
                #return 'capture and promote'
                notation = board[dest]
                cap = board.find_piece(dest, notation)
                cap.get_captured(board, notation)
                return 'choose promotion'
            else:
                self.capture(board, dest)
                return 'capture'

    def promote(self, board, dest, target_notation):
        """
        target_notation is one of the following letters:
        q, n, b, r
        """
        piece_dict = {
            'q': Queen,
            'n': Knight,
            'b': Bishop,
            'r': Rook
        }

        if self.colour == 0:
            p_notation = 'P'
            target_notation = target_notation.upper()
        else:
            p_notation = 'p'

        i, j = dest

        board.pieces[p_notation].remove(self)
        board.pieces[target_notation].append(
            piece_dict[target_notation.lower()](i, j, self.colour)
        )

        board[dest] = target_notation
        board[self.i, self.j] = None

    def capture_and_promote(self, board, dest, target_notation):
        cap_p_not = board[dest]
        captured_piece = board.find_piece(dest, cap_p_not)
        captured_piece.get_captured(board, cap_p_not)
        self.promote(board, dest, target_notation)

    def en_passant(self, board, dest):
        enemy_dir = 1 if self.colour == 0 else -1
        cap_pawn_not = 'p' if self.colour == 0 else 'P'
        i, j = dest
        cap_pawn_coord = (i+enemy_dir, j)
        cap_pawn = board.find_piece(cap_pawn_coord, cap_pawn_not)
        cap_pawn.get_captured(board, cap_pawn_not)
        self.move(board, dest)

    def legal_moves(self, board):
        res = set()
        i = self.i
        j = self.j
        pin_type = self.pin_type(board)
        direct = -1 if self.colour == 0 else 1
        start_pos = 6 if self.colour == 0 else 1
        if pin_type == 1:
            if board[i + direct, j] is None:
                res.add((i + direct, j))
            if i == start_pos:
                if board[i + direct, j] is None and board[i + 2 * direct, j] is None:
                    res.add((i + 2 * direct, j))
            return res
        if pin_type == 2:
            return set()
        if pin_type == 3:
            x = board[i + direct, j + direct]
            if x is not None:
                if x.islower() != self.colour:
                    res.add((i + direct, j + direct))
            if board.en_passant == (i + direct, j + direct):
                res.add((i + direct, j + direct))
            return res
        if pin_type == 4:
            x = board[i + direct, j - direct]
            if x is not None:
                if x.islower() != self.colour:
                    res.add((i + direct, j - direct))
            if board.en_passant == (i + direct, j - direct):
                res.add((i + direct, j - direct))
            return res

        if board[i+direct, j] is None:
            res.add((i+direct, j))
        if j < 7:
            x = board[i+direct, j+1]
            if x is not None:
                if x.islower() != self.colour:
                    res.add((i + direct, j+1))
        if j > 0:
            x = board[i+direct, j-1]
            if x is not None:
                if x.islower() != self.colour:
                    res.add((i+direct, j-1))
        if i == start_pos:
            if board[i+direct, j] is None and board[i+2*direct, j] is None:
                res.add((i+2*direct, j))

        if board.en_passant is not None:
            if (i + direct, j + 1) == board.en_passant:
                k_notation = 'k' if self.colour == 1 else 'K'
                f_p_l, _ = self.first_piece_in_direction(board, (0, -1))
                if f_p_l == k_notation:
                    cur_j = j + 2
                    f_p_r = None
                    while cur_j in set(range(8)):
                        if board[i + direct, cur_j] is not None:
                            f_p_r = board[i + direct, cur_j]
                        cur_j += 1
                    en_notation = 'QR' if self.colour == 1 else 'qr'
                    if f_p_r is not None and f_p_r not in en_notation:
                        res.add((i + direct, j + 1))
                else:
                    res.add((i + direct, j + 1))

            if (i + direct, j - 1) == board.en_passant:
                k_notation = 'k' if self.colour == 1 else 'K'
                f_p_l, _ = self.first_piece_in_direction(board, (0, +1))
                if f_p_l == k_notation:
                    cur_j = j - 2
                    f_p_r = None
                    while cur_j in set(range(8)):
                        if board[i + direct, cur_j] is not None:
                            f_p_r = board[i + direct, cur_j]
                            cur_j -= 1
                    en_notation = 'QR' if self.colour == 1 else 'qr'
                    if f_p_r is not None and f_p_r not in en_notation:
                        res.add((i + direct, j - 1))
                else:
                    res.add((i + direct, j - 1))

        u_c = board.under_check()
        if u_c == 0:
            return res
        if u_c == 1:
            return set()
        res.intersection_update(self.moves_to_avoid_check(board, u_c))
        return res


class Rook(Piece):
    pins = ['h']

    def legal_moves(self, board):
        pin_type = self.pin_type(board)
        if pin_type in {3, 4}:
            return set()
        if pin_type == 1:
            # pinned along the file
            directions = ((-1, 0), (1, 0))
        elif pin_type == 2:
            # pinned along the rank
            directions = ((0, -1), (0, 1))
        else:
            directions = ((1, 0), (-1, 0), (0, -1), (0, 1))
        res = self.long_moves(board, directions)

        u_c = board.under_check()
        if u_c == 0:
            return res
        if u_c == 1:
            return set()
        res.intersection_update(self.moves_to_avoid_check(board, u_c))
        return res

    def move(self, board, dest):
        coord_to_notation = {
            (0, 0): 'q',
            (0, 7): 'k',
            (7, 0): 'Q',
            (7, 7): 'K'
        }
        try:
            notation = coord_to_notation[(self.i, self.j)]
        except KeyError:
            pass
        else:
            try:
                board.castling.remove(notation)
            except ValueError:
                pass
        super().move(board, dest)

    def get_captured(self, board, rook_notation):
        coord_to_notation = {
            (0, 0): 'q',
            (0, 7): 'k',
            (7, 0): 'Q',
            (7, 7): 'K'
        }
        try:
            notation = coord_to_notation[self.i, self.j]
        except KeyError:
            pass
        else:
            try:
                board.castling.remove(notation)
            except ValueError:
                pass
        rook_notation = 'r' if self.colour == 1 else 'R'
        super().get_captured(board, rook_notation)


class Knight(Piece):
    def legal_moves(self, board):
        if self.pin_type(board) != 0:
            return set()
        offsets_i = (-2, -2, -1, -1, 1, 1, 2, 2)
        offsets_j = (1, -1, 2, -2, 2, -2, 1, -1)
        res = self.short_moves(board, offsets_i, offsets_j)

        u_c = board.under_check()
        if u_c == 0:
            return res
        if u_c == 1:
            return set()
        res.intersection_update(self.moves_to_avoid_check(board, u_c))
        return res


class Bishop(Piece):
    pins = ['d']

    def legal_moves(self, board):
        pin_type = self.pin_type(board)
        if pin_type in {1, 2}:
            return set()
        if pin_type == 3:
            # pinned diagonally, parallel to the main diagonal
            directions = ((-1, -1), (1, 1))
        elif pin_type == 4:
            # pinned diagonally, parallel to the counterdiagonal
            directions = ((-1, 1), (1, -1))
        else:
            directions = ((-1, -1), (-1, 1), (1, -1), (1, 1))
        res = self.long_moves(board, directions)

        u_c = board.under_check()
        if u_c == 0:
            return res
        if u_c == 1:
            return set()
        res.intersection_update(self.moves_to_avoid_check(board, u_c))
        return res


class Queen(Piece):
    pins = ['h', 'd']

    def legal_moves(self, board):
        pin_type = self.pin_type(board)
        if pin_type == 1:
            # pinned along the file
            directions = ((-1, 0), (1, 0))
        elif pin_type == 2:
            # pinned along the rank
            directions = ((0, -1), (0, 1))
        elif pin_type == 3:
            # pinned diagonally, parallel to the main diagonal
            directions = ((-1, -1), (1, 1))
        elif pin_type == 4:
            # pinned diagonally, parallel to the counterdiagonal
            directions = ((-1, 1), (1, -1))
        else:
            directions = ((-1, -1), (-1, 1), (1, -1), (1, 1),
                          (1, 0), (-1, 0), (0, -1), (0, 1))
        res = self.long_moves(board, directions)

        u_c = board.under_check()
        if u_c == 0:
            return res
        if u_c == 1:
            return set()
        res.intersection_update(self.moves_to_avoid_check(board, u_c))
        return res


class King(Piece):
    def go(self, board, dest):
        if board.turn % 2 != self.colour or dest not in self.legal_moves(board):
            return 'revert'

        board.turn += 1
        board.en_passant = None

        castle_rank = 0 if self.colour == 1 else 7

        if self.i == castle_rank and self.j == 4:
            if dest[1] in {2, 6}:
                self.castle(board, dest)
                return 'castle'

        if board[dest] is None:
            self.move(board, dest)
            return 'move'
        else:
            self.capture(board, dest)
            return 'capture'

    def move(self, board, dest):
        notations = list('kq') if self.colour == 1 else list('KQ')
        for notation in notations:
            try:
                board.castling.remove(notation)
            except ValueError:
                pass
        super().move(board, dest)

    def castle(self, board, dest):
        self.move(board, dest)
        castle_rank = 0 if self.colour == 1 else 7
        rook_file = 0 if (dest[1] < 4) else 7
        new_rook_file = 3 if (dest[1] < 4) else 5
        rook = board.find_piece((castle_rank, rook_file))
        rook.move(board, (castle_rank, new_rook_file))

    def legal_moves(self, board):
        offsets_i = (-1, -1, -1, 0, 0, 1, 1, 1)
        offsets_j = (-1, 0, 1, -1, 1, -1, 0, 1)
        res = self.short_moves(board, offsets_i, offsets_j)
        k_notation = 'k' if self.colour == 1 else 'K'
        q_notation = 'k' if self.colour == 1 else 'Q'
        castle_rank = 0 if self.colour == 1 else 7
        if k_notation in board.castling:
            if (board[castle_rank, 5] is None
                    and board[castle_rank, 6] is None):
                res.add((castle_rank, 6))
        if q_notation in board.castling:
            if (board[castle_rank, 3] is None
                    and board[castle_rank, 2] is None
                    and board[castle_rank, 1] is None):
                res.add((castle_rank, 2))

        k_notation = 'k' if self.colour == 1 else 'K'
        board[self.i, self.j] = None
        board.pieces[k_notation] = []
        rem = set()
        for square in res:
            if board.attacked_by(square) != 0:
                rem.add(square)
        board[self.i, self.j] = k_notation
        board.pieces[k_notation] = [self]
        res.difference_update(rem)

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
    def __init__(self, fen):
        # dictionary of piece names pointing to lists of piece objects
        self.pieces = defaultdict(list)
        # list of moves that have been played so far
        self.moves = []
        # map of the pieces currently placed on the board
        # pieces are named according to the FEN notation
        # empty fields are denoted as None
        self.board_map = [[None]*8 for _ in range(8)]
        cur_i = 0
        cur_j = 0
        board_fen, colour, castling, en_passant, *_, move = fen.split()
        self.turn = 2 * int(move) if colour == 'w' else 2 * int(move) + 1
        self.castling = list(castling)
        self.en_passant = None
        self.is_over = False
        self.winner = None
        for ch in board_fen:
            if ch == '/':
                cur_i += 1
                cur_j = 0
            elif ch.isdigit():
                cur_j += int(ch)
            else:
                self.board_map[cur_i][cur_j] = ch
                if ch.islower():
                    colour = 1
                else:
                    colour = 0
                self.pieces[ch].append(
                    ch_to_cls[ch.lower()](cur_i, cur_j, colour)
                )
                cur_j += 1

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
                    st += el + ' '
            st += '\n'
        return st

    def find_piece(self, coord, piece_not=None):
        if piece_not is None:
            piece_not = self[coord]
        i, j = coord
        for piece in self.pieces[piece_not]:
            if piece.i == i and piece.j == j:
                return piece
        raise ValueError('No piece found')

    def make_move(self, start, dest):
        """
        'start' and 'dest' are tuples that represent coordinates
        from which and to which move is to be made, respectively
        """
        piece_not = self[start]
        if piece_not is None:
            raise ValueError(f'There is no piece at {start}')
        piece = self.find_piece(start, piece_not)
        res = piece.go(self, dest)
        winner = self.result()
        if winner is not None:
            self.is_over = True
            self.winner = winner
        return res

    def promote(self, start, dest, piece_notation):
        pawn = self.find_piece(start)
        self.en_passant = None
        pawn.promote(self, dest, piece_notation)
        return 'promote'

    def all_legal_moves(self):
        res = set()
        for notation, pieces in self.pieces.items():
            if notation.islower() != self.turn % 2:
                continue
            res = res.union(
                *[{((piece.i, piece.j), legal_move)
                   for legal_move in piece.legal_moves(self)}
                  for piece in pieces]
            )
        return list(res)

    def attacked_by(self, square):
        """
        Return 0 is the square is not attacked by
        any of the enemy pieces, return 1 if it is
        attacked by multiple enemy pieces,
        and return the coordinates of the piece it is
        being attacked by if there is exactly one such piece
        """
        res = 0
        res_piece = None
        enemy_pieces = 'rqbpnk'
        if self.turn % 2 == 1:
            enemy_pieces = enemy_pieces.upper()


        straight_directions = (
            (0, 1), (0, -1), (1, 0), (-1, 0)
        )
        diagonal_directions = (
            (1, 1), (1, -1), (-1, 1), (-1, -1)
        )

        for direct in straight_directions:
            piece_coordinates = self.first_piece_in_direction(square, direct)
            if piece_coordinates is not None:
                piece_notation = self[piece_coordinates]
                if (piece_notation is not None
                        and piece_notation in enemy_pieces[0:2]):
                    res += 1
                    if res > 1:
                        return 1
                    res_piece = piece_coordinates

        for direct in diagonal_directions:
            piece_coordinates = self.first_piece_in_direction(square, direct)
            if piece_coordinates is not None:
                piece_notation = self[piece_coordinates]
                if (piece_notation is not None
                        and piece_notation in enemy_pieces[1:3]):
                    res += 1

                    if res > 1:
                        return 1
                    res_piece = piece_coordinates

        en_pawn_dir = 1 if self.turn % 2 == 0 else -1

        i, j = square
        if (i - en_pawn_dir) in set(range(8)) and (j+1) in set(range(8)):
            if self[i - en_pawn_dir, j + 1] == enemy_pieces[3]:
                res += 1
                if res > 1:
                    return 1
                res_piece = (i - en_pawn_dir, j + 1)
        if (i - en_pawn_dir) in set(range(8)) and (j-1) in set(range(8)):
            if self[i - en_pawn_dir, j - 1] == enemy_pieces[3]:
                res += 1
                if res > 1:
                    return 1
                res_piece = (i - en_pawn_dir, j - 1)

        knight_directions = (
            (-2, -1), (-1, -2), (1, 2), (2, 1),
            (-2, 1), (-1, 2), (1, -2), (2, -1)
        )
        for i_off, j_off in knight_directions:
            if i + i_off not in set(range(8)) or j + j_off not in set(range(8)):
                continue
            if self[i + i_off, j + j_off] == enemy_pieces[4]:
                res += 1
                if res > 1:
                    return 1
                res_piece = (i + i_off, j + j_off)

        king_directions = (
            (-1, -1), (-1, 0), (-1, 1), (0, -1),
            (0, 1), (1, -1), (1, 0), (1, 1)
        )
        for i_off, j_off in king_directions:
            if i + i_off not in set(range(8)) or j + j_off not in set(range(8)):
                continue
            if self[i + i_off, j + j_off] == enemy_pieces[5]:
                res += 1
                if res > 1:
                    return 1
                res_piece = (i + i_off, j + j_off)

        if res_piece is None:
            return 0

        return res_piece

    def first_piece_in_direction(self, square, direction):
        """
        direction = (i, j), where i, j is from {-1, 0, 1}
        and i and j aren't equal to 0 simultaneously.
        Return coordinates or None
        """
        i, j = direction
        cur_i, cur_j = square[0] + i, square[1] + j
        while cur_i in set(range(8)) and cur_j in set(range(8)):
            if self[cur_i, cur_j] is not None:
                return cur_i, cur_j
            cur_i += i
            cur_j += j
        return None

    def under_check(self):
        """
        Return 0 if the king is not under check,
        return 1 if the king is under check and the attack
        cannot be interposed and the attacking piece cannot
        be captured by another piece,
        return the coordinates of the piece it is under
        check because of otherwise
        """
        k_notation = 'k' if self.turn % 2 == 1 else 'K'
        king = self.pieces[k_notation][0]
        return self.attacked_by((king.i, king.j))

    def board_to_iter(self):
        for notation, pieces in self.pieces.items():
            for piece in pieces:
                yield piece.j + 8 * piece.i, notation

    def any_legal_moves(self):
        """
        Return True if there are legal moves that
        can be made, False otherwise
        """
        for notation, pieces in self.pieces.items():
            if notation.islower() != self.turn % 2:
                continue
            for piece in pieces:
                if piece.legal_moves(self):
                    return True
        return False

    def result(self):
        """
        Return None if the game is still running,
        return result otherwise:
        0: white won
        1: black won
        2: draw(stalemate)
        """
        if self.any_legal_moves():
            return None
        if self.under_check() == 0:
            return 2
        return 1 - (self.turn % 2)
