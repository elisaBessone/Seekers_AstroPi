import cv2
import numpy as np
from pathlib import Path


base_folder = Path(__file__).parent.resolve()
script_dir = Path(__file__).parent.resolve()

for counter in range(1, 543):
    image = cv2.imread(f"{base_folder}\\image\\ndvi_blackWhite\\ndvi_blackWhite_{counter:03d}.jpg")
    
    #print(image)
    # Create mask and draw circle onto mask
    #image = cv2.imread('circle_2.jpg')
    mask = np.zeros(image.shape, dtype=np.uint8)
    x,y = image.shape[1], image.shape[0]
    cv2.circle(mask, (1080, 785), 780, (255,255,255), -1)


    # Bitwise-and for ROI
    ROI = cv2.bitwise_and(image, mask)

    # Crop mask and turn background white
    mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
    x,y,w,h = cv2.boundingRect(mask)
    result = ROI[y:y+h,x:x+w]
    mask = mask[y:y+h,x:x+w]
    result[mask==0] = (255,0,0)

    cv2.imwrite(f"{base_folder}\\image\\tagliate\\photo_tagliata_{counter:03d}.jpg", result)

    #cv2.imshow('result', result)
    #cv2.imwrite(f".jpg", result)
    #cv2.waitKey()