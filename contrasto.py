import cv2
from pathlib import Path
import numpy as np
import csv
import pandas as pd

def to_file(message):
    with open('luminosita.csv', 'a') as filecsv:
        writer = csv.writer(filecsv)
        dati = (message)
        writer.writerow(dati)
        filecsv.close()

def clear_file():
    with open('luminosita.csv', 'w') as filecsv:
        writer = csv.writer(filecsv)
        writer.writerow("")
        filecsv.close()

lista_photo = list(range(1, 80)) + list(range(305, 431))

base_folder = Path(__file__).parent.resolve()
script_dir = Path(__file__).parent.resolve()

clear_file()
if __name__ == '__main__':
    data = pd.read_csv("data.csv", index_col='Counter')
    print(data)

    pixelLuminosi = []


    for counter in range(1, 543):
        if counter in lista_photo:

            original = cv2.imread(f"{base_folder}/image/photo/photo_{counter:03d}.jpg", 0)

            """print(original)
            print(np.max(original))
            h = np.histogram(original, bins=255, range = (0, 255))
            print(h)
            print()
            """

            valore = np.count_nonzero(original[original >= 1])
            #print(valore, counter)
            pixelLuminosi.append(valore)
        else:
            pixelLuminosi.append(np.nan)
        
    pixelLuminosi = np.array(pixelLuminosi)
    data["LuminositaNotturna"] = pixelLuminosi
    data.to_csv("data.csv")


