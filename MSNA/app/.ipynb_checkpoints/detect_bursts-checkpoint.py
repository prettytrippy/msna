import msna
from tkinter import filedialog

filename = filedialog.askopenfilename()
if filename:
    outfile = "".join(filename.split(".")[:-1]) + "_PREDICTED_BURSTS.txt"
    msna.write_bursts(filename, outfile)