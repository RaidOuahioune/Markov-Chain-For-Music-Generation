import markov_model
import utils


def main():
    notes, states = utils.build_corpus("./data/pitches.csv")
    
    model = markov_model.MarkovChainMelodyGenerator(states)
    utils.visualize_pitch(melody=states, title="Original Melody")
    model.train(notes)
    melody = model.generate(100)
    """To compare between the 2 me
    """
    utils.visualize_pitch(melody=melody, title="Generated Melody")
    
                          
if __name__ == "__main__":
    main()