"""
Board game environments for CRL experiments.

Environments:
1. Full Chess (8x8) via python-chess — proper legal move generation, en passant, castling
2. Go (9x9, 13x13, 19x19) — full rules with captures, ko, scoring
3. Connect-4 (6x7) — classic, as a simpler baseline

All environments follow a unified interface:
  - reset() -> state (np.ndarray)
  - step(action) -> (state, reward, done, info)
  - get_legal_actions() -> list[int]
  - clone() -> env copy
  - state_dim, action_dim properties
"""

import numpy as np
import random
from copy import deepcopy
from typing import Tuple, List, Optional, Dict

try:
    import chess
    HAS_PYTHON_CHESS = True
except ImportError:
    HAS_PYTHON_CHESS = False
    print("WARNING: python-chess not installed. FullChessEnv unavailable. pip install python-chess")


# ============================================================
# Full Chess (8x8) via python-chess
# ============================================================

class FullChessEnv:
    """
    Full 8x8 chess with proper rules via python-chess.
    
    State encoding (AlphaZero-style):
      - 12 binary planes for pieces (6 types x 2 colors): 8x8x12
      - 2 planes for repetition count
      - 1 plane for color to move
      - 1 plane for total move count (normalized)
      - 1 plane for castling rights (P1 kingside, queenside, P2 kingside, queenside as 4 planes)
      Total: 8x8x(12 + 2 + 1 + 1 + 4) = 8x8x20 = 1280
    
    Action encoding:
      We use a flat index over all legal moves. At each step, get_legal_actions()
      returns indices into a canonical move list. For simplicity, we use
      from_square * 64 + to_square + promotion_offset (max 4672 actions).
    """
    
    PIECE_PLANES = {
        chess.PAWN: 0, chess.KNIGHT: 1, chess.BISHOP: 2,
        chess.ROOK: 3, chess.QUEEN: 4, chess.KING: 5
    }
    
    def __init__(self, max_moves: int = 200):
        assert HAS_PYTHON_CHESS, "python-chess required. pip install python-chess"
        self.max_moves = max_moves
        self.n_planes = 20
        self.board_size = 8
        self.state_dim = self.board_size * self.board_size * self.n_planes  # 1280
        # from_sq(64) * to_sq(64) + underpromotion variants
        self.action_dim = 64 * 64 + 3 * 64  # 4288 (covers all possible moves + promotions)
        self.reset()
    
    def reset(self) -> np.ndarray:
        self.board = chess.Board()
        self.move_count = 0
        self.done = False
        self.winner = 0  # 0=ongoing/draw, 1=white, -1=black
        return self._get_state()
    
    def _get_state(self) -> np.ndarray:
        planes = np.zeros((self.n_planes, 8, 8), dtype=np.float32)
        
        # Piece planes (0-11)
        for sq in chess.SQUARES:
            piece = self.board.piece_at(sq)
            if piece is not None:
                row, col = sq // 8, sq % 8
                plane_idx = self.PIECE_PLANES[piece.piece_type]
                if piece.color == chess.BLACK:
                    plane_idx += 6
                planes[plane_idx, row, col] = 1.0
        
        # Repetition planes (12-13)
        if self.board.is_repetition(2):
            planes[12] = 1.0
        if self.board.is_repetition(3):
            planes[13] = 1.0
        
        # Color to move (14)
        planes[14] = 1.0 if self.board.turn == chess.WHITE else 0.0
        
        # Move count normalized (15)
        planes[15] = min(self.move_count / self.max_moves, 1.0)
        
        # Castling rights (16-19)
        planes[16] = float(self.board.has_kingside_castling_rights(chess.WHITE))
        planes[17] = float(self.board.has_queenside_castling_rights(chess.WHITE))
        planes[18] = float(self.board.has_kingside_castling_rights(chess.BLACK))
        planes[19] = float(self.board.has_queenside_castling_rights(chess.BLACK))
        
        return planes.reshape(-1)
    
    def _move_to_action(self, move: chess.Move) -> int:
        base = move.from_square * 64 + move.to_square
        if move.promotion and move.promotion != chess.QUEEN:
            # Underpromotion: knight=0, bishop=1, rook=2
            promo_offset = {chess.KNIGHT: 0, chess.BISHOP: 1, chess.ROOK: 2}
            base = 64 * 64 + promo_offset[move.promotion] * 64 + move.to_square
        return base
    
    def _action_to_move(self, action: int) -> chess.Move:
        if action >= 64 * 64:
            # Underpromotion
            remainder = action - 64 * 64
            promo_type_idx = remainder // 64
            to_sq = remainder % 64
            promo_map = {0: chess.KNIGHT, 1: chess.BISHOP, 2: chess.ROOK}
            promo = promo_map[promo_type_idx]
            # Find from_square from legal moves
            for m in self.board.legal_moves:
                if m.to_square == to_sq and m.promotion == promo:
                    return m
            return None
        else:
            from_sq = action // 64
            to_sq = action % 64
            # Check if this is a promotion move (pawn reaching last rank)
            piece = self.board.piece_at(from_sq)
            promotion = None
            if piece and piece.piece_type == chess.PAWN:
                if (piece.color == chess.WHITE and to_sq >= 56) or \
                   (piece.color == chess.BLACK and to_sq < 8):
                    promotion = chess.QUEEN  # Default promote to queen
            return chess.Move(from_sq, to_sq, promotion=promotion)
    
    def get_legal_actions(self) -> List[int]:
        return [self._move_to_action(m) for m in self.board.legal_moves]
    
    def step(self, action: int) -> Tuple[np.ndarray, float, bool, dict]:
        move = self._action_to_move(action)
        if move is None or move not in self.board.legal_moves:
            # Illegal move — loss
            self.done = True
            current_player = 1 if self.board.turn == chess.WHITE else -1
            self.winner = -current_player
            return self._get_state(), -1.0, True, {'winner': self.winner, 'reason': 'illegal'}
        
        self.board.push(move)
        self.move_count += 1
        
        reward = 0.0
        info = {}
        
        if self.board.is_checkmate():
            self.done = True
            self.winner = -1 if self.board.turn == chess.WHITE else 1  # Loser is to move
            reward = 1.0 if self.winner == (1 if not self.board.turn == chess.WHITE else -1) else -1.0
        elif self.board.is_stalemate() or self.board.is_insufficient_material() or \
             self.board.is_fifty_moves() or self.board.is_repetition(3):
            self.done = True
            self.winner = 0
            reward = 0.0
        elif self.move_count >= self.max_moves:
            self.done = True
            self.winner = 0
            reward = 0.0
        
        # Reward from perspective of the player who just moved
        if self.done and self.winner != 0:
            last_player = 1 if self.board.turn == chess.BLACK else -1  # Player who just moved
            reward = 1.0 if self.winner == last_player else -1.0
        
        info['winner'] = self.winner
        info['move_count'] = self.move_count
        return self._get_state(), reward, self.done, info
    
    def clone(self):
        env = FullChessEnv(self.max_moves)
        env.board = self.board.copy()
        env.move_count = self.move_count
        env.done = self.done
        env.winner = self.winner
        return env


# ============================================================
# Go Environment
# ============================================================

class GoEnv:
    """
    Full Go environment with proper rules.
    
    State encoding:
      - Plane 0: current player's stones
      - Plane 1: opponent's stones
      - Plane 2-3: 2 history planes (previous positions for ko)
      - Plane 4: liberties count (normalized) for current player groups
      - Plane 5: liberties count (normalized) for opponent groups  
      - Plane 6: color to move (all 1s or 0s)
      - Plane 7: valid moves mask
      Total: board_size x board_size x 8
    
    Action: row * board_size + col, or board_size^2 for pass.
    """
    
    def __init__(self, board_size: int = 9, komi: float = 6.5):
        self.board_size = board_size
        self.komi = komi
        self.n_planes = 8
        self.state_dim = board_size * board_size * self.n_planes
        self.action_dim = board_size * board_size + 1  # + pass
        self.reset()
    
    def reset(self) -> np.ndarray:
        self.board = np.zeros((self.board_size, self.board_size), dtype=np.int8)
        self.turn = 1  # 1=black, -1=white
        self.done = False
        self.winner = 0
        self.passes = 0
        self.move_count = 0
        self.max_moves = self.board_size * self.board_size * 4
        self.ko_point = None
        self.previous_board = None
        self.captured = {1: 0, -1: 0}
        self.history = []  # Board history for superko
        return self._get_state()
    
    def _get_state(self) -> np.ndarray:
        planes = np.zeros((self.n_planes, self.board_size, self.board_size), dtype=np.float32)
        
        # Current player's stones
        planes[0] = (self.board == self.turn).astype(np.float32)
        # Opponent's stones
        planes[1] = (self.board == -self.turn).astype(np.float32)
        
        # Previous position (for ko detection)
        if self.previous_board is not None:
            planes[2] = (self.previous_board == self.turn).astype(np.float32)
            planes[3] = (self.previous_board == -self.turn).astype(np.float32)
        
        # Liberty counts
        for i in range(self.board_size):
            for j in range(self.board_size):
                if self.board[i, j] != 0:
                    _, liberties = self._get_group(i, j)
                    lib_count = min(len(liberties) / 8.0, 1.0)  # Normalize
                    if self.board[i, j] == self.turn:
                        planes[4, i, j] = lib_count
                    else:
                        planes[5, i, j] = lib_count
        
        # Color to move
        planes[6] = 1.0 if self.turn == 1 else 0.0
        
        # Valid moves
        for action in self.get_legal_actions():
            if action < self.board_size * self.board_size:
                r, c = action // self.board_size, action % self.board_size
                planes[7, r, c] = 1.0
        
        return planes.reshape(-1)
    
    def _get_group(self, r: int, c: int):
        color = self.board[r, c]
        if color == 0:
            return set(), set()
        group = set()
        liberties = set()
        stack = [(r, c)]
        while stack:
            cr, cc = stack.pop()
            if (cr, cc) in group:
                continue
            group.add((cr, cc))
            for dr, dc in [(-1,0),(1,0),(0,-1),(0,1)]:
                nr, nc = cr + dr, cc + dc
                if 0 <= nr < self.board_size and 0 <= nc < self.board_size:
                    if self.board[nr, nc] == 0:
                        liberties.add((nr, nc))
                    elif self.board[nr, nc] == color and (nr, nc) not in group:
                        stack.append((nr, nc))
        return group, liberties
    
    def _remove_group(self, group):
        count = len(group)
        for r, c in group:
            self.board[r, c] = 0
        return count
    
    def _is_legal(self, r: int, c: int) -> bool:
        if self.board[r, c] != 0:
            return False
        if self.ko_point == (r, c):
            return False
        
        # Try placing
        self.board[r, c] = self.turn
        
        # Check if this move has liberties
        _, liberties = self._get_group(r, c)
        if len(liberties) > 0:
            self.board[r, c] = 0
            return True
        
        # Check if it captures opponent stones
        for dr, dc in [(-1,0),(1,0),(0,-1),(0,1)]:
            nr, nc = r + dr, c + dc
            if 0 <= nr < self.board_size and 0 <= nc < self.board_size:
                if self.board[nr, nc] == -self.turn:
                    group, libs = self._get_group(nr, nc)
                    if len(libs) == 0:
                        self.board[r, c] = 0
                        return True
        
        # Suicide — illegal
        self.board[r, c] = 0
        return False
    
    def get_legal_actions(self) -> List[int]:
        actions = []
        for i in range(self.board_size):
            for j in range(self.board_size):
                if self._is_legal(i, j):
                    actions.append(i * self.board_size + j)
        actions.append(self.board_size * self.board_size)  # Pass
        return actions
    
    def _score_area(self) -> float:
        """Tromp-Taylor area scoring."""
        black_area = 0
        white_area = 0
        visited = set()
        
        for i in range(self.board_size):
            for j in range(self.board_size):
                if self.board[i, j] == 1:
                    black_area += 1
                elif self.board[i, j] == -1:
                    white_area += 1
                elif (i, j) not in visited:
                    # Flood fill empty region
                    region = set()
                    borders = set()
                    stack = [(i, j)]
                    while stack:
                        cr, cc = stack.pop()
                        if (cr, cc) in region:
                            continue
                        if self.board[cr, cc] != 0:
                            borders.add(self.board[cr, cc])
                            continue
                        region.add((cr, cc))
                        visited.add((cr, cc))
                        for dr, dc in [(-1,0),(1,0),(0,-1),(0,1)]:
                            nr, nc = cr + dr, cc + dc
                            if 0 <= nr < self.board_size and 0 <= nc < self.board_size:
                                stack.append((nr, nc))
                    
                    if borders == {1}:
                        black_area += len(region)
                    elif borders == {-1}:
                        white_area += len(region)
        
        return black_area - white_area - self.komi
    
    def step(self, action: int) -> Tuple[np.ndarray, float, bool, dict]:
        self.previous_board = self.board.copy()
        
        if action == self.board_size * self.board_size:  # Pass
            self.passes += 1
            if self.passes >= 2:
                self.done = True
                score = self._score_area()
                self.winner = 1 if score > 0 else (-1 if score < 0 else 0)
        else:
            self.passes = 0
            r, c = action // self.board_size, action % self.board_size
            self.board[r, c] = self.turn
            
            # Capture opponent stones
            captured_count = 0
            captured_point = None
            for dr, dc in [(-1,0),(1,0),(0,-1),(0,1)]:
                nr, nc = r + dr, c + dc
                if 0 <= nr < self.board_size and 0 <= nc < self.board_size:
                    if self.board[nr, nc] == -self.turn:
                        group, libs = self._get_group(nr, nc)
                        if len(libs) == 0:
                            cnt = self._remove_group(group)
                            captured_count += cnt
                            if cnt == 1:
                                captured_point = list(group)[0]
            
            self.captured[self.turn] += captured_count
            
            # Ko
            if captured_count == 1 and captured_point is not None:
                _, my_libs = self._get_group(r, c)
                if len(my_libs) == 1:
                    self.ko_point = captured_point
                else:
                    self.ko_point = None
            else:
                self.ko_point = None
        
        self.move_count += 1
        if self.move_count >= self.max_moves and not self.done:
            self.done = True
            score = self._score_area()
            self.winner = 1 if score > 0 else (-1 if score < 0 else 0)
        
        self.turn *= -1
        state = self._get_state()
        
        reward = 0.0
        if self.done and self.winner != 0:
            reward = self.winner * (-self.turn)  # From perspective of player who just moved
        
        return state, reward, self.done, {'winner': self.winner, 'score': self._score_area() if self.done else None}
    
    def clone(self):
        env = GoEnv(self.board_size, self.komi)
        env.board = self.board.copy()
        env.turn = self.turn
        env.done = self.done
        env.winner = self.winner
        env.passes = self.passes
        env.move_count = self.move_count
        env.ko_point = self.ko_point
        env.previous_board = self.previous_board.copy() if self.previous_board is not None else None
        env.captured = self.captured.copy()
        return env


# ============================================================
# Connect-4
# ============================================================

class ConnectFourEnv:
    """
    Standard Connect-4: 6 rows x 7 columns.
    
    State encoding:
      - Plane 0: current player's pieces
      - Plane 1: opponent's pieces
      - Plane 2: valid moves (top row empty)
      - Plane 3: color to move
      Total: 6x7x4 = 168
    """
    
    def __init__(self):
        self.rows = 6
        self.cols = 7
        self.n_planes = 4
        self.state_dim = self.rows * self.cols * self.n_planes  # 168
        self.action_dim = self.cols  # 7
        self.reset()
    
    def reset(self) -> np.ndarray:
        self.board = np.zeros((self.rows, self.cols), dtype=np.int8)
        self.turn = 1
        self.done = False
        self.winner = 0
        self.move_count = 0
        return self._get_state()
    
    def _get_state(self) -> np.ndarray:
        planes = np.zeros((self.n_planes, self.rows, self.cols), dtype=np.float32)
        planes[0] = (self.board == self.turn).astype(np.float32)
        planes[1] = (self.board == -self.turn).astype(np.float32)
        # Valid columns
        for c in range(self.cols):
            if self.board[0, c] == 0:
                planes[2, :, c] = 1.0
        planes[3] = 1.0 if self.turn == 1 else 0.0
        return planes.reshape(-1)
    
    def get_legal_actions(self) -> List[int]:
        return [c for c in range(self.cols) if self.board[0, c] == 0]
    
    def _check_win(self, r: int, c: int) -> bool:
        color = self.board[r, c]
        for dr, dc in [(0,1),(1,0),(1,1),(1,-1)]:
            count = 1
            for sign in [1, -1]:
                for dist in range(1, 4):
                    nr, nc = r + sign*dr*dist, c + sign*dc*dist
                    if 0 <= nr < self.rows and 0 <= nc < self.cols and self.board[nr, nc] == color:
                        count += 1
                    else:
                        break
            if count >= 4:
                return True
        return False
    
    def step(self, action: int) -> Tuple[np.ndarray, float, bool, dict]:
        col = action
        row = -1
        for r in range(self.rows - 1, -1, -1):
            if self.board[r, col] == 0:
                row = r
                break
        
        if row == -1:
            self.done = True
            self.winner = -self.turn
            return self._get_state(), -1.0, True, {'winner': self.winner, 'reason': 'illegal'}
        
        self.board[row, col] = self.turn
        self.move_count += 1
        
        reward = 0.0
        if self._check_win(row, col):
            self.done = True
            self.winner = self.turn
            reward = 1.0
        elif self.move_count >= self.rows * self.cols:
            self.done = True
            self.winner = 0
        
        info = {'winner': self.winner}
        self.turn *= -1
        return self._get_state(), reward, self.done, info
    
    def clone(self):
        env = ConnectFourEnv()
        env.board = self.board.copy()
        env.turn = self.turn
        env.done = self.done
        env.winner = self.winner
        env.move_count = self.move_count
        return env


# ============================================================
# Data Collection
# ============================================================

def collect_random_games(env_class, n_games: int, verbose: bool = True, **env_kwargs) -> List[Dict]:
    """Collect trajectories from random self-play."""
    trajectories = []
    
    for i in range(n_games):
        env = env_class(**env_kwargs)
        state = env.reset()
        trajectory = {'states': [state], 'actions': [], 'rewards': []}
        
        while not env.done:
            legal = env.get_legal_actions()
            if not legal:
                break
            action = random.choice(legal)
            next_state, reward, done, info = env.step(action)
            trajectory['actions'].append(action)
            trajectory['rewards'].append(reward)
            trajectory['states'].append(next_state)
        
        trajectory['outcome'] = info.get('winner', 0)
        trajectory['length'] = len(trajectory['actions'])
        trajectories.append(trajectory)
        
        if verbose and (i + 1) % max(1, n_games // 10) == 0:
            print(f"  Collected {i+1}/{n_games} games...")
    
    return trajectories


def collect_self_play_games(env_class, policy_fn, n_games: int, 
                           temperature: float = 1.0, **env_kwargs) -> List[Dict]:
    """Collect trajectories from self-play with a learned policy."""
    import torch
    trajectories = []
    
    for i in range(n_games):
        env = env_class(**env_kwargs)
        state = env.reset()
        trajectory = {'states': [state], 'actions': [], 'rewards': []}
        
        while not env.done:
            legal = env.get_legal_actions()
            if not legal:
                break
            
            with torch.no_grad():
                state_t = torch.FloatTensor(state).unsqueeze(0)
                action = policy_fn(state_t, legal, temperature)
            
            next_state, reward, done, info = env.step(action)
            trajectory['actions'].append(action)
            trajectory['rewards'].append(reward)
            trajectory['states'].append(next_state)
            state = next_state
        
        trajectory['outcome'] = info.get('winner', 0)
        trajectory['length'] = len(trajectory['actions'])
        trajectories.append(trajectory)
    
    return trajectories
