from typing import Any, List, Union

import mlx.core as mx
import numpy as np
from mlx_whisper.tokenizer import Tokenizer


def print_sentence(
    tokens: List[int],
    log_probs: Union[np.ndarray, mx.array],
    tokenizer: Tokenizer,
):
    """Pretty-print a sentence's tokens and probabilities."""
    print(f"Sentence log_prob = {mx.sum(log_probs):.4f}:")

    for token_index, token in enumerate(tokens):
        token_str = tokenizer.decode_with_timestamps([token])
        log_prob = float(log_probs[token_index])
        print(f"  token {token:>6}, str={token_str:>25}, prob={np.exp(log_prob):.7f}")

    print()


def print_tokens_side_by_side(
    token1: int,
    token2: int,
    dec1: str,
    dec2: str,
    log_prob1: float,
    log_prob2: float,
):
    """Print a pair of tokens and probabilities side by side for comparison."""
    if token1 is not None:
        print(f"  {token1:>6}, str={dec1:>25}, prob={log_prob1:>9.7f} ", end="   ")
    else:
        print(f"  {' ':>6}      {' ':>25}       {' ':>9} ", end="   ")

    if token2 is not None:
        print(f"  {token2:>6}, str={dec2:>25}, prob={log_prob2:>9.7f} ", end="   ")
    else:
        print(f"")


def print_sentences(
    tokens1: List[int],
    tokens2: List[int],
    log_probs1: List[float],
    log_probs2: List[float],
    tokenizer: Any,
):
    """Print two sentences tokens and probabilities side by side."""
    nll_original = -np.sum(log_probs1)
    nll_target = -np.sum(log_probs2)

    print_tokens_side_by_side(
        token1=" ",
        token2=" ",
        dec1="sentence1",
        dec2="sentence2",
        log_prob1=-nll_original,
        log_prob2=-nll_target,
    )
    for token_index in range(max(len(tokens1[0]), len(tokens2[0]))):
        token1 = target_token = None

        if token_index < len(log_probs1):
            token1 = int(tokens1[0, token_index])
            token1_str = tokenizer.decode_with_timestamps([token1])
            log_prob1 = float(log_probs1[token_index])

        if token_index < len(log_probs2):
            token2 = int(tokens2[0, token_index])
            token2_str = tokenizer.decode_with_timestamps([token2])
            log_prob2 = float(log_probs2[token_index])

        print_tokens_side_by_side(
            token1=token1,
            token2=target_token,
            dec1=token1_str,
            dec2=token2_str,
            log_prob1=log_prob1,
            log_prob2=log_prob2,
        )
    print()
