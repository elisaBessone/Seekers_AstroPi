"""import matplotlib
import numpy as np
import cv2
from pathlib import Path

base_folder = Path(__file__).parent.resolve()

script_dir = Path(__file__).parent.resolve()
#img = cv2.imread(f"{base_folder}/image2/ndvi_blackWhite/ndvi_blackWhite_{484:03d}.jpg")
img = cv2.imread(f"{base_folder}/circle_22.jpg", 0)


print (img)

#equazione 
valore = np.count_nonzero(img[img >= (0.5*255)])    #circa 128

print (valore)"""
"""
coordinate = (1080, 785)
color = (255, 0, 0)
radius = 780
thickness = 2
image = cv2.circle(img, coordinate, radius, color, thickness)
#cv2.imshow("Image", image)
cv2.imwrite(f"{base_folder}\\circle_2.jpg", image)"""


import cv2
from pathlib import Path
import numpy as np
import csv
import pandas as pd

#lista_photo = list(range(1, 80)) + list(range(305, 431))

base_folder = Path(__file__).parent.resolve()
script_dir = Path(__file__).parent.resolve()

if __name__ == '__main__':
    data = pd.read_csv("data.csv", index_col='Counter')
    print(data)

    ndvi = []


    for counter in range(1, 543):
        original = cv2.imread(f"{base_folder}\\image\\tagliate\\photo_tagliata_{counter:03d}.jpg", 0)
        #print(original)
        """print(original)
        print(np.max(original))
        h = np.histogram(original, bins=255, range = (0, 255))
        print(h)
        print()
        """
        valore = np.count_nonzero(original[original >= 128])  
        #(0.5*255)])    #circa 128

        #valore = np.count_nonzero(original[original >= 1])
        #print(valore, counter)
        ndvi.append(valore)
        print(valore)
    
        
    ndvi = np.array(ndvi)
    data["ndvi"] = ndvi
    data.to_csv("data.csv")


