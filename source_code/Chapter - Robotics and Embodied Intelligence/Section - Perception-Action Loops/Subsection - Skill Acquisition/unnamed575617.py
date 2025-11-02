def chess_rules(positions):
    # Example function that generates moves based on simple chess rules
    moves = []
    for position, piece in positions.items():
        if piece == 'Rook':
            # Generate rook moves
            moves.extend(generate_rook_moves(position))
        elif piece == 'Knight':
            # Generate knight moves
            moves.extend(generate_knight_moves(position))
    return moves

def generate_rook_moves(position):
    # Simplified function to generate rook moves
    return ['Move up', 'Move down', 'Move left', 'Move right']

def generate_knight_moves(position):
    # Simplified function to generate knight moves
    return ['L-shape move']