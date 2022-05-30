from picamera import PiCamera
from time import sleep
fotocamera = PiCamera()
fotocamera.resolution  = (2272, 1704)
fotocamera.framerate = 15
fotocamera.start_preview()
sleep(5)
fotocamera.capture ()
fotocamera.stop preview()
