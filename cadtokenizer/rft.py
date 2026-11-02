"""Rejection-sampling fine-tuning (RFT, round 1) for generation.

Sample k candidates per training generation prompt, keep the ones the CAD kernel accepts that also
match the training target structure (no retraced edges, loop count within the training range,
ended by the pair/end token), and add them to the SFT cache as extra training rows.
"""
import random
import sqlite3


def structure(sequence):
    """Loop count and retraced-edge count of a canonical sketch sequence."""
    loops, current = [], []
    for token in sequence.split():
        if token.startswith('<loop_end>'):
            loops.append(current)
            current = []
        elif token.startswith('<'):
            continue
        else:
            parts = token.split(',')
            if parts[0] in ('line', 'arc', 'circle'):
                current.append((parts[0], tuple(parts[1:])))
    retraced = 0
    for loop in loops:
        points = [point for _, point in loop]
        for i in range(len(points) - 2):
            if points[i] == points[i + 2] and loop[i][0] == 'line' and loop[i + 1][0] == 'line':
                retraced += 1
    return len(loops), retraced


def training_generation_rows(database):
    query = '''SELECT e.id, e.instruction, g.sequence FROM examples e JOIN geometry g ON g.id = e.target_id
               WHERE e.task = 'generation' AND e.split = 'train' ORDER BY e.id'''
    with sqlite3.connect(database) as db:
        return db.execute(query).fetchall()


def draw_subset(database, count, seed):
    """Deterministic subset of training generation prompts (id and instruction only)."""
    rows = training_generation_rows(database)
    return [[key, instruction] for key, instruction, _ in random.Random(seed).sample(rows, count)]


def loop_limit(database, quantile=0.99):
    """Upper loop count observed in the training generation targets (99th percentile by default)."""
    counts = sorted(structure(sequence)[0] for _, _, sequence in training_generation_rows(database))
    return counts[int(quantile * (len(counts) - 1))]


def accept(sequence, native_ids, end_id, kernel_valid, limit):
    """A candidate is kept only if the kernel accepts it and its structure matches the training targets."""
    if not kernel_valid or not native_ids or native_ids[-1] != end_id:
        return False
    loops, retraced = structure(sequence)
    return retraced == 0 and 0 < loops <= limit
