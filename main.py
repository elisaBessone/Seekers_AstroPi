#!/usr/local/bin/python
import os, sys
#importazione di librerie esterne
from pathlib import Path
from logzero import logger, logfile
from sense_hat import SenseHat
from picamera import PiCamera
from orbit import ISS
from time import sleep
from datetime import datetime, timedelta
import csv

from PIL import Image
from pycoral.adapters import common
from pycoral.adapters import classify
from pycoral.utils.edgetpu import make_interpreter
from pycoral.utils.dataset import read_label_file

#funzione
def create_csv_file(data_file):
    """Create a new CSV file and add the header row"""
    with open(data_file, 'w') as f:
        writer = csv.writer(f)
        header = ("Counter", "Date/time", "Latitude", "Longitude", "Temperature", "Humidity")
        writer.writerow(header)

def add_csv_data(data_file, data):
    """Add a row of data to the data_file CSV"""
    with open(data_file, 'a') as f:
        writer = csv.writer(f)
        writer.writerow(data)

def convert(angle):
    """
    Convert a `skyfield` Angle to an EXIF-appropriate
    representation (rationals)
    e.g. 98' 34' 58.7 to "98/1,34/1,587/10"

    Return a tuple containing a boolean and the converted angle,
    with the boolean indicating if the angle is negative.
    """
    sign, degrees, minutes, seconds = angle.signed_dms()
    exif_angle = f'{degrees:.0f}/1,{minutes:.0f}/1,{seconds*10:.0f}/10'
    return sign < 0, exif_angle


def capture(camera, image):
    """Use `camera` to capture an `image` file with lat/long EXIF data."""
    location = ISS.coordinates()

    # Convert the latitude and longitude to EXIF-appropriate representations
    south, exif_latitude = convert(location.latitude)
    west, exif_longitude = convert(location.longitude)

    # Set the EXIF tags specifying the current location
    camera.exif_tags['GPS.GPSLatitude'] = exif_latitude
    camera.exif_tags['GPS.GPSLatitudeRef'] = "S" if south else "N"
    camera.exif_tags['GPS.GPSLongitude'] = exif_longitude
    camera.exif_tags['GPS.GPSLongitudeRef'] = "W" if west else "E"

    # Capture the image
    #camera.start.preview()
    camera.capture(image)
    #camera.end.preview()
    #sleep(20)


#inizio del programma
#stringa che contiene il nome dela cartella dove si trova il file che sto eseguendo
base_folder = Path(__file__).parent.resolve()

# Set a logfile name
#crea file events.log in base_folder
logfile(base_folder/"events.log")

# Set up Sense Hat
#costruttore della classe SenseHat
sense = SenseHat()

# Set up camera
#costruttore della classe PiCamera
cam = PiCamera()
#attributi tutti pubblici
cam.resolution = (1296, 972)    #I NUMERI SONO LA RISOLUZIONE DELLA CAMERA
                                #CONTROLLARE


# Initialise the CSV file
data_file = base_folder/"data.csv"
create_csv_file(data_file)  

# Initialise the photo counter
counter = 1
# Record the start and current time
start_time = datetime.now()     #data e ora di questo istante
now_time = datetime.now()
# Run a loop for (almost) three hours

script_dir = Path(__file__).parent.resolve()

modelDayNight_file = script_dir/'modelsDayNight/model_edgetpu.tflite' # name of model
dataDayNight_dir = script_dir
labelDayNight_file = dataDayNight_dir/'labelsDayNight.txt' # Name of your label file
imageFile = dataDayNight_dir/'testMare/test2.jpg' # Name of image for classification #mettere image_file

interpreter_DayNight = make_interpreter(f"{modelDayNight_file}")
interpreter_DayNight.allocate_tensors()

size = common.input_size(interpreter_DayNight)

#*******************
#mare
model_sea_file = script_dir/'modelsSea/model_edgetpu.tflite' # name of model
data_sea_dir = script_dir
label_sea_file = data_sea_dir/'labelsSea.txt' # Name of your label file
imageFile = data_sea_dir/'testMare/test2.jpg' # Name of image for classification #metere image_file

interpreter_sea = make_interpreter(f"{model_sea_file}")
interpreter_sea.allocate_tensors()

size = common.input_size(interpreter_sea)

#notte
model_night_file = script_dir/'modelsNight/model_edgetpu.tflite' # name of model
data_night_dir = script_dir
label_night_file = data_night_dir/'labelsNight.txt' # Name of your label file
imageFile = data_night_dir/'testMare/test2.jpg' # Name of image for classification #metere image_file

interpreter_night = make_interpreter(f"{model_night_file}")
interpreter_night.allocate_tensors()

size = common.input_size(interpreter_night)


while (now_time < start_time + timedelta(minutes=178)):
    #se dentro il try si genera un errore qualsiasi il programma non crasha e va a eseguire
    #cio' che o' contenuto dentro except
    #aggiungere filtro telecamera infrarossi
    try:
        # abilita magnetometro, giroscopio e accellerometro
        sense.set_imu_config(True, True, True)

        #magnetometro
        compass = round(sense.compass, 4)

        gyroscope = sense.gyroscope

        accelerometer = sense.accelerometer
        
        # Get coordinates of location on Earth below the ISS
        # calcolo posizione stazione spaziale
        location = ISS.coordinates()
        print(location) #stampa location ogni volta
        # Save the data to the file
        # non ci sono gli array, 3 collezioni: TUPLE, LISTE, DIZIONARI
        data = (    #tupla con 6 elementi
            counter,
            datetime.now(),
            location.latitude.degrees,  # latitudine in gradi
            location.longitude.degrees, # longitudine in gradi
            compass, 
            gyroscope, 
            accelerometer,
        )
        add_csv_data(data_file, data)   # scrive data nel file dei dati
        # Capture image
        # creato il nome del file. 
        image_file = f"{base_folder}/photo_{counter:03d}.jpg"
        # esegue la foto 
        capture(cam, image_file)
        
        #********************************************
        
        image = Image.open(imageFile).convert('RGB').resize(size, Image.ANTIALIAS)

        common.set_input(interpreter_DayNight, image)
        interpreter_DayNight.invoke()
        classes = classify.get_classes(interpreter_DayNight, top_k=1)

        labels = read_label_file(labelDayNight_file)
        for c in classes:
            print(f'{labels.get(c.id, c.id)} {c.score:.5f}')
            x = labels.get(c.id)
        
        #se e' giorno
        if(x == "day"):
            
            image = Image.open(imageFile).convert('RGB').resize(size, Image.ANTIALIAS)

            common.set_input(interpreter_sea, image)
            interpreter_sea.invoke()
            classes = classify.get_classes(interpreter_sea, top_k=1)

            labels = read_label_file(label_sea_file)
            for c in classes:
                print(f'{labels.get(c.id, c.id)} {c.score:.5f}')
                x = labels.get(c.id)
        else:

            image = Image.open(imageFile).convert('RGB').resize(size, Image.ANTIALIAS)

            common.set_input(interpreter_night, image)
            interpreter_night.invoke()
            classes = classify.get_classes(interpreter_night, top_k=1)

            labels = read_label_file(label_night_file)
            for c in classes:
                print(f'{labels.get(c.id, c.id)} {c.score:.5f}')
                x = labels.get(c.id)
            
        
        #************************************************************
        
        # Log event 
        logger.info(f"iteration {counter}")
        counter += 1
        sleep(3)   # pausa di 30 secondi, circa 12sec
        # Update the current time
        now_time = datetime.now()   #ricalcolo ora corrente

    except Exception as e:
        # scrive una stringa dentro il file
        logger.error(f'{e._class.name_}: {e}')
