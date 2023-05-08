"""Tools for making optimal guesses at Wordle"""
from collections import Counter
import logging
from typing import Callable, Iterable, List, Mapping, Set

import numpy as np
from tqdm import tqdm


logger = logging.getLogger(__name__)


class WordleGreedySolver:
    """Computes the guess in a game of Wordle that minimizes uncertainty
    for the next turn.

    The class takes as input the allowed set of guess words, and the set
    of possible solution words. It is assumed that all words have the
    same length.

    The class internally computes and updates a matrix `self._response_matrix`,
    where each row represents a potential guess word and each column
    represents a potential solution word. The entries of the matrix are
    integers that encode the response you would get if you guessed the guess
    word given by the row, and if the solution were the solution word
    given by the column.

    self._response_matrix.shape = (
        len(self._guess_words), len(self._solution_words)
    )

    Given the response matrix, the set of optimal guesses at a given time
    can be computed, and is returned by `self.best_guesses()`. Here optimal
    is defined in a greedy sense, meaning that an optimal guess reduces
    uncertainty by the most possible immediately after the guess is made.
    The definition of uncertainty can be specified by passing an
    uncertainty_metric function to `self.best_guesses()`.

    The function `self.add_guess_response` allows the user to input the
    response given for a guess while playing an actual game. This will
    result in an update of the internal state `self._response_matrix`.
    """
    n_ary_conversion = {'b': 0, 'y': 1, 'g': 2}

    def __init__(self, guess_words: Iterable[str], solution_words: Iterable[str]):
        """Creates a WordleGreedySolver instance.

        :param guess_words: set of allowed guess words. Assumed to be all
            the same length.
        :param solution_words: set of words that are possible solutions to
            the puzzle. Assumed to be all the same length, and to
            be the same length as the words in `guess_words`. If any
            solution words are not in the set of guess words, they
            will be added to the guess words.
        """
        solution_words = {s.lower() for s in solution_words}
        guess_words = {s.lower() for s in guess_words}
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

        self._prune()

    def best_guesses(
            self, uncertainty_metric: Callable[[np.ndarray], float]=None
    ) -> Set[str]:
        """Returns the set of optimal guess words in the current state.
        Optimality is defined as minimizing the score computed by the
        `uncertainty_metric` function.

        :param uncertainty_metric: a function that takes in a row of
            `self._response_matrix` and outputs an uncertainty
            score that is a measure of how much uncertainty will
            remain once you get the response to the guess word
            corresponding to the row. The best guess word(s) will
            minimize this score.

        :return: the optimal guess words in the current state
        """
        self._check_non_empty()

        if self._response_matrix.shape[1] == 1:
            # The solution has been determined uniquely.
            # Return it.
            return self._solution_words[0]

        guess_scores = self._compute_guess_scores(uncertainty_metric)
        candidate_inds = np.where(
            guess_scores == guess_scores.min()
        )[0]
        best_candidate_set = (
            set(self._guess_words[candidate_inds]) &
            set(self._solution_words)
        )
        if best_candidate_set:
            return best_candidate_set
        return set(self._guess_words[candidate_inds])

    def best_guess(
            self, uncertainty_metric: Callable[[np.ndarray], float]=None
    ) -> str:
        """Returns an optimal guess word (an arbitrary choice from the
        optimal set obtained by `self.best_guesses()`).

        :param uncertainty_metric: a function that takes in a row of
            `self._response_matrix` and outputs an uncertainty
            score that is a measure of how much uncertainty will
            remain once you get the response to the guess word
            corresponding to the row. The best guess word(s) will
            minimize this score.

        :return: an optimal guess word in the current state
        """
        return self.best_guesses(uncertainty_metric).pop()

    def add_guess_response(self, guess: str, response: str):
        """Input the response received for a given guess while playing
        the game, and update the internal state (`self._response_matrix`)
        accordingly.

        :param guess: the word guessed
        :param response: the response received when making the guess,
            e.g. 'gygbb' means that the first and third letter of the
            guess are correct, the second letter of the guess is
            present in the solution, but at a different position,
            and the fourth and fifth letters in the guess are not
            present in the solution word.
        """
        self._check_non_empty()

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

        self._prune()

    def save(self, outfile):
        np.savez(
            outfile,
            solution_words=self._solution_words,
            guess_words=self._guess_words,
            response_matrix=self._response_matrix,
        )

    @classmethod
    def from_file(cls, infile):
        solver = cls([], [])
        loaded = np.load(infile)
        solver._solution_words = loaded["solution_words"]
        solver._guess_words = loaded["guess_words"]
        solver._response_matrix = loaded["response_matrix"]
        return solver

    def _compute_guess_scores(
            self, uncertainty_metric: Callable[[np.ndarray], float]=None
    ):
        if uncertainty_metric is None:
            uncertainty_metric = largest_solution_set_size
        return np.apply_along_axis(
            uncertainty_metric, 1, self._response_matrix
        )

    def _prune(self):
        if self._response_matrix.shape[0] == 0:
            return

        prune_inds = np.apply_along_axis(
            lambda row: np.unique(row).size == 1, 1, self._response_matrix
        )
        if not prune_inds.all():
            self._response_matrix = self._response_matrix[~prune_inds, :]
            self._guess_words = self._guess_words[~prune_inds]

    def _check_non_empty(self):
        if self._response_matrix.shape[0] == 0:
            raise ValueError("No guess words remaining")

        if self._response_matrix.shape[1] == 0:
            raise ValueError("No solution words remaining")


# Helper functions

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
        both words, 'y' if the letters are different but the letter
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


# Uncertainty metrics
# These are functions that take in a single row of the response matrix
# and return a score representing the remaining uncertainty you will
# have after getting a response if you make the guess that corresponds
# to the row. In general, a good guess will minimize this score.

def largest_solution_set_size(row: np.ndarray) -> float:
    """Returns the size of the largest response group in the row."""
    return np.unique(row, return_counts=True)[1].max()

def average_solution_set_size(row: np.ndarray) -> float:
    """Returns a value proportional to the average response group size
    over all elements in the row.
    """
    counts = np.unique(row, return_counts=True)[1]
    return np.sum(counts ** 2)

def entropy(row: np.ndarray) -> float:
    """Returns a value proportional to the entroy of the row."""
    counts = np.unique(row, return_counts=True)[1]
    return np.sum(counts * np.log(counts))


# Simulation functions

def simulate(solver: WordleGreedySolver, solution_word: str) -> List[str]:
    """Outputs the series of guesses that the given solver would make
    for the given solution word.

    :param solver: a WordleGreedySolver instance
    :param solution_word: the solution to the puzzle

    :return: a list of guess words that the solver would make until
        it makes the correct guess
    """
    guesses = []
    while True:
        guess = solver.best_guess()
        guesses.append(guess)
        if guess == solution_word:
            return guesses
        if len(guesses) > 1 and guesses[-1] == guesses[-2]:
            raise ValueError(
                f"Solver could not solve puzzle. Guesses: {guesses}"
            )
        solver.add_guess_response(
            guess, get_response(guess, solution_word)
        )
