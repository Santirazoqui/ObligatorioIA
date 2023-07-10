import math
import random
import sys
from agent import Agent
from board import Board

WIN_SCORE = 100000
LOSE_SCORE = -1000000

class MiniMaxAgent(Agent):
    def __init__(self, minimax_max_depth: int, player=1):
        self.player = player
        self.opponent = 2
        self.max_depth = minimax_max_depth
    
    def next_action(self, obs):
        pos, _ = self.minimax(obs, self.player, self.max_depth, 0, -sys.maxsize, sys.maxsize)
        return pos
    
    def minimax(self, board: Board, current_player: int, max_depth, current_depth, alpha, beta) -> tuple[int,int]:
        #Casos base
        #(1) chequeo si alguien gano o si se lleno el tablero
        if(board.is_final()):
            #Empate:
            if(board._winner == 0):
                return (2, 0)
            #Si gano el oponente:
            elif(board._winner != self.player):
                return (2, LOSE_SCORE)
            #Ganó el jugador
            else:
                return (2, WIN_SCORE)

        #(2) chequear depth y hacer f/eval
        if(current_depth > max_depth):
            val = self.evaluation_Function(board, current_player)
            return (2, val)
            #eturn (0, 0)

        #Casos no base
        possible_actions = board.get_posible_actions()
        ran = random.randint(0, len(possible_actions)-1)

        # tomo una acción random para empezar
        chosen_action = possible_actions[ran]
        
        #Caso juega oponente
        if current_player != self.player: #mini
            value = sys.maxsize
            for action in possible_actions:
                clone_board = board.clone()
                clone_board.add_tile(action, self.opponent)
                _, new_value = self.minimax(clone_board, self.player, max_depth, current_depth + 1, alpha, beta) 
                        #No me importa la nueva accion _ ?
                if (new_value < value):
                    value = new_value
                    chosen_action = action
                #alfa beta prunning - poda beta
                beta = min(value, beta) 
                if alpha >= beta:
                    break

        #Caso juega jugador
        else: #(current_player == self.player) max
            value = -sys.maxsize
            for action in possible_actions:
                clone_board = board.clone()
                clone_board.add_tile(action, self.player)
                _, new_value = self.minimax(clone_board, self.opponent, max_depth, current_depth + 1, alpha, beta)
                if new_value > value:
                    value = new_value
                    chosen_action = action
                #alfa beta prunning - poda alfa
                alpha = max(value, alpha) 
                if alpha >= beta:
                    break
        return chosen_action, value

    def evaluation_Function(self, board: Board, current_player: int):
        eval = self.heuristic_score_lines(board, current_player)
        #eval += self.heuristic_fichas_comibles(board, current_player)
        #eval += self.heuristic_middle_pieces(board)
        return eval
    
    def heuristic_middle_pieces(self, board: Board):
        alfa = 5
        player_score = 0
        opponent_score = 0
        grid = board._grid
        for row in range(board.heigth): 
            if(grid[row][3] == self.player):
                player_score += 10
            if(grid[row][3] != self.player and grid[row][3] != " "):
                opponent_score += 10

        return alfa * (player_score - opponent_score)

    def heuristic_fichas_comibles(self, board: Board, current_player):
        alfa = -5
        player_score = 0
        opponent_score = 0
        grid = board._grid
        for row in range(1, board.heigth - 1):
            for cell in range(1, board.length - 1):
                es_comible = False
                if(grid[row][cell] != " "):
                    #comible horizontal por la derecha
                    if (grid[row][cell - 1] != " " 
                        and grid[row][cell - 1] != grid[row][cell] 
                        and grid[row][cell + 1] == " "):
                        es_comible = True
                    #comible horizontal por la izquierda
                    elif(grid[row][cell + 1] != " " 
                         and grid[row][cell + 1] != grid[row][cell] 
                         and grid[row][cell - 1] == " "):
                        es_comible = True

                    #comer diagonal por la derecha hacia abajo
                    elif (grid[row - 1][cell - 1] != " " 
                          and grid[row - 1][cell - 1] != grid[row][cell] 
                          and grid[row + 1][cell + 1] == " "):
                        es_comible = True
                    #comer diagonal por la derecha hacia arriba  
                    elif (grid[row + 1][cell - 1] != " " 
                          and grid[row + 1][cell - 1] != grid[row][cell] 
                          and grid[row - 1][cell + 1] == " "):
                        es_comible = True
                    #comer diagonal por la izquierda hacia abajo
                    elif(grid[row - 1][cell + 1] != " " 
                         and grid[row - 1][cell + 1] != grid[row][cell] 
                         and grid[row + 1][cell - 1] == " "):
                        es_comible = True
                    #comer diagonal por la izquierda hacia arriba
                    elif(grid[row + 1][cell + 1] != " " 
                         and grid[row + 1][cell + 1] != grid[row][cell] 
                         and grid[row - 1][cell - 1] == " "):
                        es_comible = True

                if(es_comible):
                    if(grid[row][cell] == self.player):
                        player_score += 1
                    else:
                        opponent_score += 1 

        return alfa*(player_score - opponent_score)

    def heuristic_score_lines(self, board: Board, current_player):
        alfa = 10
        player_score = 0
        opponent_score = 0
        grid = board._grid
        #horizontal
        for row in range(board.heigth):
            #agarro el array de la fila
            row_array = list(grid[row,:])
            for cell in range(board.length - 3):
                #tomo las 4 siguientes del array de la fila
                line = row_array[cell: cell + 4]
                #evaluo el puntaje de esa linea y lo sumo al agente correspondiente
                line_score = self.evaluate_line(line, current_player)
                if(cell == self.player):
                    player_score += line_score
                else:
                    opponent_score += line_score
        
        #vertical
        for col in range(board.length):
            col_array = list(grid[:, col])
            for row in range(board.heigth - 3):
                line = col_array[row:row + 4]
                line_score = self.evaluate_line(line, current_player)
                if(cell == self.player):
                    player_score += line_score
                else:
                    opponent_score += line_score

        #diagonal positivo
        for row in range(3, board.heigth):
            for col in range(board.length - 3):
                line = [grid[row-i][col+i] for i in range(4)]
                line_score = self.evaluate_line(line, current_player)
                if(cell == self.player):
                    player_score += line_score
                else:
                    opponent_score += line_score

        #diagonal negativo
        for row in range(3, board.heigth):
            for col in range(3, board.length):
                line = [grid[row-i][col-i] for i in range(4)]
                line_score = self.evaluate_line(line, current_player)
                if(cell == self.player):
                    player_score += line_score
                else:
                    opponent_score += line_score

        return alfa*(player_score - opponent_score)
    
    def evaluate_line(self, line, current_player_piece: int):
        score = 0
        #no considero las que tienen de ambos jugadore, ya que en ese caso la 
        # linea se bloquea y no me sirve

        #linea con 4 (creo que no sirve para nada si igual WIN_SCORE existe)
        if line.count(current_player_piece) == 4:
            score += 100
        #linea con 3 y una vacía
        if line.count(current_player_piece) == 3 and line.count(0) == 1:
            score += 50
        #linea con 2 y 2 vacias
        if line.count(current_player_piece) == 3 and line.count(0) == 1:
            score += 10

        return score


    def heuristic_utility(self, board: Board):
        return 0
    