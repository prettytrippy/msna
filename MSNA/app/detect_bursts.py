import msna
from tkinter import filedialog
from msna.app.outcomes import burst_frequency, burst_incidence, burst_area

def process_filename(name):
    no_extension = "".join(name.split(".")[:-1])
    no_slashes = no_extension.split("/")[-1]
    return no_slashes

if __name__ == "__main__":
    filename = filedialog.askopenfilename(title="Select Data File")
    directory = filedialog.askdirectory(title="Select Destination Folder")

    if not directory:
        directory = open("directory.txt", 'r').read()
    else:
        with open("directory.txt", 'w') as file:
            file.write(directory)

    if filename:
        print(f"\nAutomatically detecting bursts in file {filename}")

        outfile = directory + "/" + process_filename(filename) + "_PREDICTED_BURSTS.txt"
        df = msna.write_bursts(filename, outfile)
        
        print(f"\nOutcome Measures:\nBurst frequency;Burst incidence;Burst area\n{burst_frequency(df)};{burst_incidence(df)};{burst_area(df)}")

        print(f"\nPredicted bursts written to {outfile}\n")


