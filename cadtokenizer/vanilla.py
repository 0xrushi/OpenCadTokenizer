"""Vanilla-LLaMA control: the same SFT data and selection as the CAD-Tokenizer run, but CAD sequences are
written as plain text and tokenized by the native Llama tokenizer. There is no VQ codebook, alignment or
trailer; the model writes the CAD sequence string, and the kernel checks it directly.
"""
import random
import sqlite3


def prompt_text(task, instruction, source=None):
    if task == 'generation':
        return f'Task: CAD generation\nInstruction: {instruction}\nOutput CAD:\n'
    return f'Task: CAD editing\nInstruction: {instruction}\nOriginal CAD:\n{source}\nOutput CAD:\n'


def selected_training_keys(database, seed, editing_per_generation):
    """Same selection as instruction_data.prepare_sft_tokens: all generations plus sampled editing examples."""
    with sqlite3.connect(database) as db:
        rows = db.execute('SELECT id, task FROM examples WHERE split="train" ORDER BY id').fetchall()
    generations = [key for key, task in rows if task == 'generation']
    edits = [key for key, task in rows if task == 'editing']
    random.Random(seed).shuffle(edits)
    return set(generations + edits[:int(len(generations) * editing_per_generation)])


def example_rows(database, split_filter, keys=None):
    """(id, task, instruction, source_sequence, target_sequence) rows for a split, optionally restricted to keys."""
    query = '''SELECT e.id, e.task, e.instruction, s.sequence, t.sequence
               FROM examples e LEFT JOIN geometry s ON s.id = e.source_id
               JOIN geometry t ON t.id = e.target_id WHERE e.split = ? ORDER BY e.id'''
    with sqlite3.connect(database) as db:
        for row in db.execute(query, (split_filter,)):
            if keys is None or row[0] in keys:
                yield row


def encode_example(tokenizer, task, instruction, source, target, context):
    """Token ids and labels for one example, or None when it does not fit in the context."""
    prompt = tokenizer.encode(prompt_text(task, instruction, source), add_special_tokens=True)
    completion = tokenizer.encode(target, add_special_tokens=False) + [tokenizer.eos_token_id]
    if len(prompt) + len(completion) > context:
        return None
    return {'input_ids': prompt + completion, 'labels': [-100] * len(prompt) + completion}
