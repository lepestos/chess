class Coordinates:
    number_to_letter = {
        i: chr(i + 97) for i in range(8)
    }

    k_moves = {
        (-1, -1), (-1, 0), (-1, 1), (0, -1),
        (0, 1), (1, -1), (1, 0), (1, 1)
    }

    n_moves = {
        (-2, -1), (-1, -2), (-2, 1), (1, -2),
        (-1, 2), (2, -1), (2, 1), (1, 2)
    }

    r_moves = {
        (-1, 0), (1, 0), (0, -1), (0, 1)
    }

    b_moves = k_moves.difference(r_moves)

    q_moves = r_moves.union(b_moves)

    notation_to_moves = {
        'k': k_moves,
        'n': n_moves,
        'r': r_moves,
        'b': b_moves,
        'q': q_moves
    }

    def __init__(self, x, y=None):
        if y is None:
            self.i = x[0]
            self.j = x[1]
        else:
            self.i = x
            self.j = y

    def __getitem__(self, key):
        if key == 0:
            return self.i
        if key == 1:
            return self.j
        raise KeyError(key)

    def __add__(self, other):
        return Coordinates(self.i + other[0], self.j + other[1])

    def __sub__(self, other):
        return Coordinates(self.i-other[0], self.j-other[1])

    def __floordiv__(self, other):
        return Coordinates(self.i//other, self.j//other)

    def __eq__(self, other):
        return self.i == other.i and self.j == other.j

    def __le__(self, other):
        if self.i <= other.i and self.j <= other.j:
            return True
        if self.i >= other.i and self.j >= other.j:
            return False
        return None

    def __lt__(self, other):
        if self.i <= other.i and self.j <= other.j:
            if self.i != other.i or self.j != other.j:
                return True
            else:
                return False
        if self.i >= other.i and self.j >= other.j:
            return False
        return None

    def __abs__(self):
        return self.__class__(abs(self.i), abs(self.j))

    def __repr__(self):
        return f'Coordinates({self.i}, {self.j})'

    def __hash__(self):
        return self.i ^ self.j

    def __bool__(self):
        return bool(self.i) or bool(self.j)

    def tuple(self):
        return self.i, self.j

    def to_human_notation(self):
        return self.number_to_letter[self.j] + str(8-self.i)

    def long_range_moves(self, notation):
        moves = self.notation_to_moves[notation.lower()]
        res = set()
        for move in moves:
            offset = list(move)
            while (self + offset).is_valid():
                res.add(self + offset)
                offset[0] += move[0]
                offset[1] += move[1]
        return res

    def short_range_moves(self, notation):
        res = {self + x for x in self.notation_to_moves[notation.lower()]}
        return {x for x in res if x.is_valid()}

    def pawn_moves(self, notation):
        colour = notation.islower()
        d_move_rank = 1 if colour == 1 else 6
        direction = 1 if colour == 1 else -1
        res = {self + (direction, 0)}
        if d_move_rank == self.i:
            res.add(self + (2*direction, 0))
        return res

    notation_to_move_type = {
        'r': long_range_moves,
        'n': short_range_moves,
        'b': long_range_moves,
        'k': short_range_moves,
        'q': long_range_moves,
        'p': pawn_moves
    }

    def moves(self, notation):
        """
        takes piece notation as input,
        returns all possible moves from the
        current square for that piece
        (provided that the board is empty)
        """
        return self.notation_to_move_type[notation](self, notation)

    def is_valid(self):
        valid_set = range(8)
        return self.i in valid_set and self.j in valid_set

    def same_line(self, other):
        if self.i == other.i or self.j == other.j:
            return True
        return False

    def same_diagonal(self, other):
        if abs(self.i - other.i) == abs(self.j - other.j):
            return True
        return False

    def squares_between(self, other):
        """
        Return all the squares between
        two given ones, not including them.
        If given squares are not on the same line,
        neither on the same diagonal,
        return empty set.
        """
        res = set()
        if self.same_line(other) or self.same_diagonal(other):
            increment = abs(self-other)
            if not increment:
                return set()
            increment //= (increment.i or increment.j)
            lesser, bigger = (self, other) if self <= other else (other, self)
            cur = lesser + increment
            while cur < bigger:
                res.add((cur.i, cur.j))
                cur += increment
            return res
        return set()



