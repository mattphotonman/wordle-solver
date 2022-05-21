"""Tools for making optimal guesses at Wordle"""
from collections import Counter
import logging
from typing import Iterable, Mapping

import numpy as np
from tqdm import tqdm


logger = logging.getLogger(__name__)


class WordleGreedySolver:
    """Computes the guess in a game of Wordle that minimizes uncertainty
    for the next turn.
    """
    n_ary_conversion = {'b': 0, 'y': 1, 'g': 2}

    def __init__(self, guess_words: Iterable[str], solution_words: Iterable[str]):
        solution_words = {s.lower() for s in solution_words}
        guess_words = {s.lower() for s in guess_words}
        assert solution_words
        self._solution_words = np.array(sorted(solution_words))
        self._guess_words = np.array(sorted(guess_words | solution_words))

        self._response_matrix = np.zeros(
            (len(self._guess_words), len(self._solution_words)), dtype=int
        )
        logger.info("Computing responses for each (guess, solution) pair.")
        pair_iter = (
            (row_idx, guess_word, col_idx, solution_word)
            for row_idx, guess_word in enumerate(self._guess_words)
            for col_idx, solution_word in enumerate(self._solution_words)
        )
        for row_idx, guess_word, col_idx, solution_word in tqdm(
                pair_iter, total=self._response_matrix.size):
            self._response_matrix[row_idx, col_idx] = to_n_ary(
                get_response(guess_word, solution_word),
                self.n_ary_conversion
            )

        self._compute_guess_scores_and_prune()

    def best_guess(self):
        if self._response_matrix.shape[1] == 1:
            # The solution has been determined uniquely.
            # Return it.
            return self._solution_words[0]
        return self._guess_words[np.argmin(self._guess_scores)]

    def add_guess_response(self, guess: str, response: str):
        response = to_n_ary(response, self.n_ary_conversion)
        row_idx = np.searchsorted(self._guess_words, guess)
        if self._guess_words[row_idx] != guess:
            # this guess gave no new information
            return

        solution_inds = self._response_matrix[row_idx] == response
        if not solution_inds.any():
            raise ValueError(f"Response not possible for guess '{guess}'")
        self._response_matrix = self._response_matrix[:, solution_inds]
        self._solution_words = self._solution_words[solution_inds]

        self._compute_guess_scores_and_prune()

    def _compute_guess_scores_and_prune(self):
        self._guess_scores = np.apply_along_axis(
            lambda row: np.unique(row, return_counts=True)[1].max(),
            1,
            self._response_matrix
        )
        
        prune_inds = self._guess_scores == self._response_matrix.shape[1]
        if not prune_inds.all():
            self._response_matrix = self._response_matrix[~prune_inds, :]
            self._guess_words = self._guess_words[~prune_inds]
            self._guess_scores = self._guess_scores[~prune_inds]


def get_response(guess_word: str, solution_word: str) -> str:
    """Returns the response if the given guess is made.

    :param guess_word: the guess being made
    :param solution_word: the word that is the correct solution
        to the puzzle

    :return: a string describing the response you would get for the
        guess `guess_word` if the solution to the puzzle is
        `solution_word`. The response is a string that has the same
        length as both `guess_word` and `solution_word`, and in each
        position has the letter 'g' if the letters are the same in
        both words, 'y' if the letters are different by the letter
        in the `guess_word` occurs somewhere in the `solution_word`,
        and 'b' if the letter in the `guess_word` doesn't occur
        anywhere in the `solution_word`.
    """
    sol_counts = Counter(solution_word)
    response = [None for _ in guess_word]
    # fill in 'g' for all matching characters
    for idx, (guess_char, sol_char) in enumerate(
            zip(guess_word, solution_word)):
        if guess_char == sol_char:
            response[idx] = 'g'
            sol_counts[guess_char] -= 1
    # handle remaining cases of either 'y' or 'b'
    for idx, guess_char in enumerate(guess_word):
        if response[idx] is not None:
            continue
        if sol_counts[guess_char] > 0:
            response[idx] = 'y'
            sol_counts[guess_char] -= 1
            continue
        response[idx] = 'b'

    return ''.join(response)


def to_n_ary(s: str, char_digit_lookup: Mapping[str, int]) -> int:
    """Maps a string to an integer, via the character to digit lookup
    table.

    :param s: the string to convert
    :param char_digit_lookup: a map from characters to integers
    
    :return: an integer created by converting characters to digits
        in an n-ary number system, where n = len(char_digit_lookup)
    """
    num_out = 0
    base = 1
    for character in reversed(s):
        digit = char_digit_lookup[character]
        num_out += digit * base
        base *= len(char_digit_lookup)
    return num_out
