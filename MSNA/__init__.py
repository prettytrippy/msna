from msna.msna_pipeline import MSNA_pipeline
 
pipeline = MSNA_pipeline(sampling_rate=250, verbose=True)
pipeline.load("msna/pretrained")

predict = pipeline.predict

__all__ = ["predict"]

__version__ = "0.1.0"
__author__ = "Richard Dow"