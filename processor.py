import numpy as np
import os
import re

import numpy as np

def fen_to_board_tensor(fen):
    fen_parts = fen.split(' ')
    board_str = fen_parts[0]
    board_tensor = np.zeros((8, 8, 20), dtype=np.float32)
    piece_to_index = {
        'P': 0, 'N': 1, 'B': 2, 'R': 3, 'Q': 4, 'K': 5,
        'p': 6, 'n': 7, 'b': 8, 'r': 9, 'q': 10, 'k': 11
    }
    rows = board_str.split('/')
    for row in range(8):
        col = 0
        for char in rows[row]:
            if char.isdigit():
                col += int(char)
            else:
                board_tensor[7-row, col, piece_to_index[char]] = 1.0
                col += 1
    board_tensor[:, :, 12] = 1 if len(fen_parts) > 1 and fen_parts[1] == 'w' else 0
    if len(fen_parts) > 2:
        castling_str = fen_parts[2]
        castling_rights = {
            'K': 13, 'Q': 14, 'k': 15, 'q': 16
        }
        for right in castling_rights:
            board_tensor[:, :, castling_rights[right]] = 1.0 if right in castling_str else 0.0
    if len(fen_parts) > 3 and fen_parts[3] != '-':
        ep_square = fen_parts[3]
        file_map = {'a': 0, 'b': 1, 'c': 2, 'd': 3, 'e': 4, 'f': 5, 'g': 6, 'h': 7}
        rank_map = {'1': 0, '2': 1, '3': 2, '4': 3, '5': 4, '6': 5, '7': 6, '8': 7}
        file = file_map[ep_square[0]]
        rank = rank_map[ep_square[1]]
        board_tensor[rank, file, 17] = 1.0
    if len(fen_parts) > 4:
        halfmove_clock = int(fen_parts[4])
        board_tensor[:, :, 18] = np.log(halfmove_clock + 1) / 10.0
    if len(fen_parts) > 5:
        fullmove_number = int(fen_parts[5])
        board_tensor[:, :, 19] = np.log(fullmove_number + 1) / 10.0
    return board_tensor

class ChessBoardProcessor:

    @staticmethod
    def generate_advanced_training_data(file_path, validation_split=0.2, oversample=True, output_classes=6):
        X = []
        y = []

        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Training data file not found: {file_path}")

        if output_classes == 2:
            state_map = {
            'Check' : 0,
            'Nothing' : 1
            }
        elif output_classes == 4:
            state_map = {
            'Checkmate' : 0,
            'Check' : 1,
            'Stalemate' : 2,
            'Nothing' : 3
            }
        elif output_classes == 6:
            state_map = {
            'Checkmate Black': 0,
            'Checkmate White': 1,
            'Check Black': 2,
            'Check White': 3,
            'Stalemate': 4,
            'Nothing': 5
            }
        else:
            raise ValueError(f"Invalid number of output classes: {output_classes}")

        with open(file_path, 'r') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                
                match = re.match(r'^(.*?)(?:\s+(Checkmate Black|Checkmate White|Check Black|Check White|Stalemate|Nothing))?$', line)
                if not match:
                    continue
                
                fen_state = match.group(1)
                state = match.group(2) if match.group(2) else 'Nothing'
                
                try:
                    board_tensor = fen_to_board_tensor(fen_state)
                    X.append(board_tensor)
                    if output_classes == 2:
                        state = state.replace(' Black', '').replace(' White', '')
                    elif output_classes == 4:
                        state = state.replace(' Black', '').replace(' White', '')
                    y.append(state_map.get(state, output_classes - 1))
                except Exception as e:
                    print(f"Error processing FEN: {fen_state}. Error: {e}")
        
        X = np.array(X)
        y = np.array(y)

        split_index = int(len(X) * (1 - validation_split))
        
        X_train = np.array(X[:split_index])
        X_val = np.array(X[split_index:])
        
        y_train = np.array(y[:split_index])
        y_val = np.array(y[split_index:])
        
        def to_one_hot(labels, num_classes):
            one_hot = np.zeros((labels.size, num_classes))
            one_hot[np.arange(labels.size), labels] = 1
            return one_hot

        y_train = to_one_hot(y_train, num_classes=output_classes)
        y_val = to_one_hot(y_val, num_classes=output_classes)

        return X_train, X_val, y_train, y_val, 0

    @staticmethod
    def generate_line(fen_line, output_classes=6):
        X = []
        y = []

        if output_classes == 2:
            state_map = {
            'Check' : 0,
            'Nothing' : 1
            }
        elif output_classes == 4:
            state_map = {
            'Checkmate' : 0,
            'Check' : 1,
            'Stalemate' : 2,
            'Nothing' : 3
            }
        elif output_classes == 6:
            state_map = {
            'Checkmate Black': 0,
            'Checkmate White': 1,
            'Check Black': 2,
            'Check White': 3,
            'Stalemate': 4,
            'Nothing': 5
            }
        else:
            raise ValueError(f"Invalid number of output classes: {output_classes}")

        line = fen_line.strip()
        
        match = re.match(r'^(.*?)(?:\s+(Checkmate Black|Checkmate White|Check Black|Check White|Stalemate|Nothing))?$', line)
        if not match:
            return None, None
        
        fen_state = match.group(1)
        state = match.group(2) if match.group(2) else 'Nothing'
        
        try:
            board_tensor = fen_to_board_tensor(fen_state)
            X.append(board_tensor)
            
            if output_classes == 2:
                state = state.replace(' Black', '').replace(' White', '')
            elif output_classes == 4:
                state = state.replace(' Black', '').replace(' White', '')
            y.append(state_map.get(state, output_classes - 1))
        except Exception as e:
            print(f"Error processing FEN: {fen_state}. Error: {e}")
        
        X = np.array(X)
        y = np.array(y)

        def to_one_hot(labels, num_classes):
            one_hot = np.zeros((labels.size, num_classes))
            one_hot[np.arange(labels.size), labels] = 1
            return one_hot

        y = to_one_hot(y, num_classes=output_classes)
        
        return X, y
