"""Flask backend providing a minimal web UI to play 2048 with an AI agent."""

from flask import Flask, request, jsonify, render_template
import os
import numpy as np
from board import Board
from agent import DQNAgent

# Mapping index positions returned by the agent to actual move strings
ACTIONS = ["up", "down", "left", "right"]

# Initialize the Flask application and register templates/static folders
app = Flask(__name__, template_folder="../templates", static_folder="../static")

board = None  # Will hold the Board instance once a game starts
agent = None  # Loaded DQNAgent controlling the AI


def get_valid_moves(b: Board) -> list[int]:
    """Return a list of valid move indices for the current board state."""
    valid_moves: list[int] = []
    for idx, action in enumerate(ACTIONS):
        board_copy = np.copy(b.board)

        if action == "up":
            b.move_up()
        elif action == "down":
            b.move_down()
        elif action == "left":
            b.move_left()
        elif action == "right":
            b.move_right()

        if not np.array_equal(board_copy, b.board):
            valid_moves.append(idx)

        b.board = board_copy

    return valid_moves


@app.route("/")
def index() -> str:
    """Serve the main HTML page."""
    return render_template("index.html")


@app.route("/models")
def models() -> list[str]:
    """Return a list of available model files."""
    files = [f for f in os.listdir("models") if f.endswith(".h5")]
    files.sort()
    return jsonify(files)


@app.route("/start", methods=["POST"])
def start() -> dict:
    """Initialize a new game and optionally load an AI model."""
    global board, agent
    data = request.get_json()
    model_name = data.get("model")

    board = Board()
    board.add_random_tile()
    board.add_random_tile()

    agent = DQNAgent(state_size=16, action_size=4)
    if model_name:
        path = os.path.join("models", model_name)
        agent.load(path)

    # Disable exploration for deterministic play
    agent.epsilon = 0.0

    return jsonify({"board": board.board.tolist()})


@app.route("/move", methods=["POST"])
def move() -> dict:
    """Perform a user-triggered move."""
    global board
    data = request.get_json()
    direction = data.get("direction")
    if direction in ACTIONS and board is not None:
        board.move(direction)
    return jsonify({"board": board.board.tolist()})


@app.route("/ai_move", methods=["POST"])
def ai_move() -> dict:
    """Have the AI choose and execute the next move."""
    global board, agent

    if board is None or agent is None:
        return jsonify({"error": "Game not started"}), 400

    state = board.get_board_state()
    valid_moves = get_valid_moves(board)
    if not valid_moves:
        return jsonify({"board": board.board.tolist(), "action": None})

    action_idx = agent.act(state, valid_moves)
    action = ACTIONS[action_idx]
    board.move(action)

    return jsonify({"board": board.board.tolist(), "action": action})


if __name__ == "__main__":
    # When executed directly, start the development server
    app.run(debug=True)
