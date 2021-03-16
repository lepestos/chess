import json

from channels.generic.websocket import AsyncWebsocketConsumer

from .pieces import Board
from . import chessAI


N_TO_LETTER = {
    0: 'a', 1: 'b', 2: 'c', 3: 'd', 4: 'e', 5: 'f', 6: 'g', 7: 'h'
}


class MoveConsumer(AsyncWebsocketConsumer):
    async def connect(self):
        self.room_name = "chess_room"
        self.room_group_name = "chess_group"

        await self.channel_layer.group_add(
            self.room_group_name,
            self.channel_name
        )
        await self.accept()

    async def disconnect(self, close_code):
        # Leave room group
        await self.channel_layer.group_discard(
            self.room_group_name,
            self.channel_name
        )

    async def start_game(self, _):
        self.board = Board()
        msg = 'White to Move' if self.board.turn % 2 == 0 else 'Black to Move'

        await self.channel_layer.group_send(
            self.room_group_name,
            {
                'type': 'send_message',
                'command': 'show_message',
                'message': msg
            }
        )
        for position, piece in self.board:
            await self.channel_layer.group_send(
                self.room_group_name,
                {
                    'type': 'send_message',
                    'command': 'place',
                    'piece': piece,
                    'position': position
                }
            )

        """
        await self.send(text_data=json.dumps({
            'command': 'show_message',
            'message': msg
        }))
        for position, piece in self.board:
            await self.send(text_data=json.dumps({
                'command': 'place',
                'piece': piece,
                'position': position
            }))
        """

    async def ai_vs_ai(self, data):
        self.mode = 'ai_vs_ai'
        player1 = chessAI.ProperAI(self.board, 0, data['depth'])
        player2 = chessAI.ProperAI(self.board, 1, data['depth'])
        self.players = [player1, player2]
        await self.pass_move()

    async def player_vs_ai(self, data):
        self.mode = 'player_vs_ai'
        self.ai = chessAI.ProperAI(self.board, 1, data['depth'])
        await self.pass_move()

    async def player_vs_player(self, _):
        self.mode = 'player_vs_player'
        await self.pass_move()

    async def ai_make_move(self, player):
        move = player.make_move()
        data = dict()
        data['to'] = move.dest
        data['from'] = move.start
        if move.promotion_piece is not None:
            data['prom_piece'] = move.promotion_piece
        await self.move(data)

    @staticmethod
    def to_notation(piece_type: int, colour: int) -> str:
        dct = {
            0: 'k',
            1: 'p',
            2: 'r',
            3: 'b',
            4: 'n',
            5: 'q'
        }
        res = dct[piece_type]
        if colour == 0:
            res = res.upper()
        return res

    async def move(self, data):
        response = self.board.make_move(int(data['from']), int(data['to']))
        if response is None:
            return
        remove, place, pieces, colour = response

        if self.board.result == '-':
            msg = 'White to Move' if self.board.turn % 2 == 0 else 'Black to Move'
        elif self.board.result == '1/2-1/2':
            msg = 'Draw'
        elif self.board.result == '1-0':
            msg = 'White Won!'
        elif self.board.result == '0-1':
            msg = 'Black Won!'
        else:
            raise ValueError(self.board.result)

        await self.channel_layer.group_send(
            self.room_group_name,
            {
                'type': 'send_message',
                'command': 'show_message',
                'message': msg
            }
        )
        await self.channel_layer.group_send(
            self.room_group_name,
            {
                'type': 'send_message',
                'command': 'play_move_sound',
            }
        )
        await self.channel_layer.group_send(
            self.room_group_name,
            {
                'type': 'send_message',
                'command': 'update_board',
                'remove': list(remove),
                'place': list(place),
                'pieces': [self.to_notation(piece, colour) for piece in pieces]
            }
        )
        """
        await self.send(text_data=json.dumps({
            'command': 'show_message',
            'message': msg
        }))

        await self.send(text_data=json.dumps({
            'command': 'play_move_sound'
        }))

        await self.send(text_data=json.dumps({
            'command': 'update_board',
            'remove': list(remove),
            'place': list(place),
            'pieces': [self.to_notation(piece, colour) for piece in pieces]
        }))
        """

    async def send_message(self, event):
        await self.send(text_data=json.dumps(event))

    async def highlight(self, data):
        response = [move.dest for move in self.board.legal_moves()
                    if move.start == int(data['initial_square'])]
        await self.send(text_data=json.dumps({
            'command': 'highlight',
            'tiles': response
        }))

    async def pass_move(self, *_):
        if self.mode == 'ai_vs_ai':
            if self.board.result == '-':
                await self.ai_make_move(self.players[self.board.turn])
        if self.mode == 'player_vs_ai':
            if self.ai.colour == self.board.turn and self.board.result == '-':
                await self.ai_make_move(self.ai)
        if self.mode == 'player_vs_player':
            pass

    commands = {
        'move_request': move,
        'highlight': highlight,
        'start_game': start_game,
        'ai_vs_ai': ai_vs_ai,
        'player_vs_ai': player_vs_ai,
        'player_vs_player': player_vs_player,
        'pass_move': pass_move,
    }

    async def receive(self, text_data):
        text_data_json = json.loads(text_data)
        await self.commands[text_data_json['command']](self, text_data_json)
