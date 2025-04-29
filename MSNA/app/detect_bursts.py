import msna
from tkinter import filedialog
from tkinter.filedialog import askdirectory

def process_filename(name):
    no_extension = "".join(name.split(".")[:-1])
    no_slashes = no_extension.split("/")[-1]
    return no_slashes

if __name__ == "__main__":
    directory = askdirectory()
    filename = filedialog.askopenfilename()

    print(f"\nAutomatically detecting bursts in file {filename}")

    if filename:
        outfile = directory + "/" + process_filename(filename) + "_PREDICTED_BURSTS.txt"
        msna.write_bursts(filename, outfile)

    print(f"\nPredicted bursts written to {outfile}\n")
