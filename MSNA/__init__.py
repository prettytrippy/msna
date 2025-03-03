from msna.our_methods.binary_classifier.msna_pipeline import MSNA_pipeline
from msna.msna_common import read_msna
import numpy as np
 
pipeline = MSNA_pipeline(sampling_rate=250, verbose=True)
pipeline.load("msna/our_methods/binary_classifier/pretrained")

predict = pipeline.predict

def get_metadata_string(file_path):
    with open(file_path, "r", encoding="latin-1") as f:
        lines = f.readlines()
    
    for i, line in enumerate(lines):
        if line.startswith("Range="):  
            return "".join(lines[:i + 1]) 
    
    raise ValueError("Could not identify metadata end based on 'Range='")

def create_labchart_file(output_path, metadata, df):
    metadata_lines = metadata.split("\n")
    modified_metadata = "\n".join(metadata_lines)
    
    with open(output_path, "w", encoding="latin-1") as f:
        f.write(modified_metadata) 
    
    no_bust_df = df.drop('BURST', axis=1)
    no_bust_df.to_csv(output_path, sep="\t", mode="a", index=False, header=False, na_rep="NaN")

def write_bursts(infile, outfile):
    df = read_msna(infile)
    predicted_bursts = predict(infile)
    predicted_col = np.zeros(len(df))
    predicted_col[predicted_bursts] = 1

    def make_burst_col(flag):
        if flag == 1:
            return "#11 BURST"
        else:
            return None

    predicted_col = [make_burst_col(i) for i in predicted_col]

    df['Predicted BURST'] = predicted_col
    create_labchart_file(outfile, get_metadata_string(infile), df)

__all__ = ["predict", "write_bursts"]

__version__ = "0.1.0"
__author__ = "Richard Dow"