class TicTacToe:
    def __init__(self):
        self.board = [" " for _ in range(9)]  # 3x3 grid as a list
        self.ai_symbol = "O"
        self.player_symbol = "X"

    def print_board(self):
        for i in range(0, 9, 3):
            print(f"{self.board[i]} | {self.board[i+1]} | {self.board[i+2]}")
            if i < 6: print("---------")

    def is_winner(self, symbol):
        win_combinations = [(0,1,2), (3,4,5), (6,7,8), (0,3,6), (1,4,7), (2,5,8), (0,4,8), (2,4,6)]
        return any(all(self.board[i] == symbol for i in combo) for combo in win_combinations)

    def is_board_full(self):
        return " " not in self.board

    def evaluate(self):
        if self.is_winner(self.ai_symbol): return 10
        if self.is_winner(self.player_symbol): return -10
        return 0

    def minimax(self, depth, is_maximizing):
        score = self.evaluate()
        if score != 0 or self.is_board_full(): return score

        if is_maximizing:
            best = -float('inf')
            for i in range(9):
                if self.board[i] == " ":
                    self.board[i] = self.ai_symbol
                    best = max(best, self.minimax(depth + 1, False))
                    self.board[i] = " "
            return best
        else:
            best = float('inf')
            for i in range(9):
                if self.board[i] == " ":
                    self.board[i] = self.player_symbol
                    best = min(best, self.minimax(depth + 1, True))
                    self.board[i] = " "
            return best

    def find_best_move(self):
        best_score = -float('inf')
        best_move = -1
        for i in range(9):
            if self.board[i] == " ":
                self.board[i] = self.ai_symbol
                score = self.minimax(0, False)
                self.board[i] = " "
                if score > best_score:
                    best_score = score
                    best_move = i
        return best_move

    def play(self):
        while not self.is_board_full():
            self.print_board()
            move = int(input("Enter your move (0-8): "))
            if 0 <= move <= 8 and self.board[move] == " ":
                self.board[move] = self.player_symbol
                if self.is_winner(self.player_symbol):
                    self.print_board()
                    print("You win!")
                    return
            else:
                print("Invalid move, try again.")
                continue
            if not self.is_board_full():
                ai_move = self.find_best_move()
                self.board[ai_move] = self.ai_symbol
                if self.is_winner(self.ai_symbol):
                    self.print_board()
                    print("AI wins!")
                    return
        self.print_board()
        print("It's a draw!")

game = TicTacToe()
game.play()
