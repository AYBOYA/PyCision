import numpy as np

class DiscreteChain:
    def __init__(self, transition_probabilities_matrix: np.array, labels: np.array):
        """
            Check for valid TPM
        """
        valid_tpm = np.all(np.isclose(np.sum(transition_probabilities_matrix,axis=1),1.0))
        if not valid_tpm:
            raise ValueError("Invalid Transition Probabilities Matrix - Rows Must Sum to 1")
        if transition_probabilities_matrix.shape[0] != transition_probabilities_matrix.shape[1]:
            raise ValueError("Invalid Transition Probabilities Matrix - Matrix Must be Square")
        
        """
            Check for Label congruence with TPM
        """
        if labels.ndim != 1:
            raise ValueError("Invalid Labels - Must be 1D array")
        if labels.shape[0] != transition_probabilities_matrix.shape[0]:
            raise ValueError("Invalid Labels - Must Have Number of Labels Equal to States in TPM")

        self.tpm = transition_probabilities_matrix
        self.labels = labels


    def generate_chain(self, starting_distribution: np.array, chain_length: int) -> np.array:
        """
            Generate Chain Based on TPM

            args:
                chain_length - Length of Chain to generate
        """

        chain = np.zeros(chain_length, dtype = int)
        arranged = np.arange(self.tpm.shape[0])
        
        rng = np.random.default_rng()
        current_state = rng.choice(arranged, p = starting_distribution)
        for i in range(chain_length):
            chain[i] = current_state
            current_state = rng.choice(arranged, p = self.tpm[current_state])
        
        return self.labels[chain.astype(int)]

    
    def print_tpm(self) -> None:
        print(self.tpm)
