from msna_pipeline import MSNA_pipeline
 
pipeline = MSNA_pipeline(sampling_rate=250, verbose=True)
pipeline.load("../pretrained/")

__all__ = ["predict"]

predict = pipeline.predict

__version__ = "0.1.0"
__author__ = "Richard Dow"