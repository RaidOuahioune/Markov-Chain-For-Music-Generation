import numpy as np
from music21 import  note




class State:
    
    def __init__(self, pitch, duration):
        self.pitch = pitch
        self.duration = duration
        
    def __eq__(self, other):
        if not isinstance(other, State):
            return False
        return self.pitch == other.pitch and self.duration == other.duration
    
    def __hash__(self):
        return hash((self.pitch, self.duration))
    
    
class MarkovChainMelodyGenerator:
    """
    Represents a Markov Chain model for melody generation.
    """

    def __init__(self, states:list[State]):
        self.states = states
        self.initial_probabilities = np.zeros(len(states))
        self.transition_matrix = np.zeros((len(states), len(states)))
        self._state_indexes = {state: i for (i, state) in enumerate(states)}

    def train(self, notes:list[note.Note]):
        """
        Train the model based on a list of notes.
        """
        self._calculate_initial_probabilities(notes)
        self._calculate_transition_matrix(notes)

    def generate(self, length: int)->list[State]:
        """
        Generate a melody of a given length.
        """
        melody = [self._generate_starting_state()]
        for _ in range(1, length):
            melody.append(self._generate_next_state(melody[-1]))
        return melody

    def _calculate_initial_probabilities(self, notes:list[note.Note]):
        """
        Calculate the initial probabilities from the provided notes.
        """
        for note in notes:
            self._increment_initial_probability_count(note)
        self._normalize_initial_probabilities()

    def _increment_initial_probability_count(self, note:note.Note):
        """
        Increment the probability count for a given note.

        Parameters:
            note (music21.note.Note): A note object.
        """
        state = State(note.pitch.nameWithOctave, note.duration.quarterLength)
        self.initial_probabilities[self._state_indexes[state]] += 1

    def _normalize_initial_probabilities(self):
        """
        Normalize the initial probabilities array such that the sum of all
        probabilities equals 1.
        """
        total = np.sum(self.initial_probabilities)
        if total:
            self.initial_probabilities /= total
        self.initial_probabilities = np.nan_to_num(self.initial_probabilities)

    def _calculate_transition_matrix(self, notes:list[note.Note]):
        """
        Calculate the transition matrix from the provided notes.
        """
        for i in range(len(notes) - 1):
            self._increment_transition_count(notes[i], notes[i + 1])
        self._normalize_transition_matrix()

    def _increment_transition_count(self, current_note:note.Note, next_note:note.Note):
        """
        Increment the transition count from current_note to next_note.
        This class assumes we are using a state_order of 1 (i.e., the next state wll be based only the current one).
        """
        state = State(
            current_note.pitch.nameWithOctave,
            current_note.duration.quarterLength,
        )
        next_state = State(
            next_note.pitch.nameWithOctave,
            next_note.duration.quarterLength,
        )
        self.transition_matrix[
            self._state_indexes[state], self._state_indexes[next_state]
        ] += 1

    def _normalize_transition_matrix(self):
        """
        This method normalizes each row of the transition matrix so that the
        sum of probabilities in each row equals 1. This is essential for the rows
        of the matrix to represent probability distributions of
        transitioning from one state to the next.
        """      
        row_sums = self.transition_matrix.sum(axis=1, keepdims=True)
        self.transition_matrix = np.nan_to_num(np.divide(self.transition_matrix, row_sums))


    def _generate_starting_state(self)->State:
        """
        Generate a starting state based on the initial probabilities.

        Returns:
            A state from the list of states.
        """
        initial_index = np.random.choice(
            list(self._state_indexes.values()), p=self.initial_probabilities
        )
        return self.states[initial_index]

    def _generate_next_state(self, current_state:State)->State:
        """
        Generate the next state based on the transition matrix and the current
        state.
        """
        if self._does_state_have_subsequent(current_state):
            index = np.random.choice(
                list(self._state_indexes.values()),
                p=self.transition_matrix[self._state_indexes[current_state]],
            )
            return self.states[index]
        return self._generate_starting_state()

    def _does_state_have_subsequent(self, state:State)->bool:
        """
        Check if a given state has a subsequent state in the transition matrix.
        """
        return self.transition_matrix[self._state_indexes[state]].sum() > 0
