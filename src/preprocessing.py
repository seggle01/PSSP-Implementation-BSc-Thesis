import numpy as np

#########  DSSP label → integer mappings (Q3 / Q8) #########
def preprocess_tgt_Q3(seq: str):
    mapping = {'H': 0, 'E': 1, 'C': 2}
    return [mapping[aa] for aa in seq]

def preprocess_tgt_Q8(seq: str):
    # DSSP 8-state alphabet
    mapping = {
        'G': 0,  # 3-turn helix (310 helix)
        'H': 1,  # Alpha helix
        'I': 2,  # 5-turn helix (pi helix)
        'B': 3,  # Beta bridge
        'E': 4,  # Extended strand (beta sheet)
        'S': 5,  # Bend
        'T': 6,  # Turn
        'C': 7   # Coil/loop
    }
    return [mapping[aa] for aa in seq]

#########  Q8 → Q3 reduction (vectorised lookup table) #########
Q8_TO_Q3 = np.array([0, 0, 0, 1, 1, 2, 2, 2], dtype=np.int8)

def reduce_q8_to_q3(labels: np.ndarray) -> np.ndarray:
    """Map a Q8 integer label array to Q3 integer labels."""
    return Q8_TO_Q3[labels]

#########  Integer → DSSP character (inverse of the mappings above) #########
def parse_int_to_q3(i: int) -> str:
    return {0: 'H', 1: 'E', 2: 'C'}[i]

def parse_int_to_q8(i: int) -> str:
    return {0: 'G', 1: 'H', 2: 'I', 3: 'B', 4: 'E', 5: 'S', 6: 'T', 7: 'C'}[i]
