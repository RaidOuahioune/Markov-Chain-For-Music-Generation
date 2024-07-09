
from markov_model import State
from music21 import metadata, note, stream
import pandas as pd
def visualize_pitch(melody:list[State],title):
    """
    Visualize a sequence of (pitch, duration) pairs using music21.
    """
    score = stream.Score()
    score.metadata = metadata.Metadata(title=title)
    part = stream.Part()
    for state  in melody:
        print(state.duration)
        part.append(note.Note(state.pitch, quarterLength=state.duration))
    score.append(part)
    score.show()
    
def build_corpus(file_path:str):
    """
    Build a corpus from a file.
    
    Parameters:
        file_path (str): The path to the  csv file.
    """
    df=pd.read_csv(file_path)
    
    notes=[]
    states=[]
    for row in df.itertuples(index=False):
        notes.append(note.Note(row[0], quarterLength=row[1]))
        states.append(State(row[0], row[1]))
        
    return notes, states



