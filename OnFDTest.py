import cv2
from matplotlib import pyplot as plt 

imcap = cv2.VideoCapture(0) # Initialize camera
imcap.set(3.640) # Set width
imcap.set(4,480) # Set height   

while True:
    success, img = imcap.read() #Set image as frames from camera
    #Image processing
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    d_set = {'.xml' + cv2.data.haarcascades + "haarcascade_frontalface_default.xml"} #Set of data set
    data_set = cv2.CascadeClassifier(d_set) #Initialize reference data set
    d_object = data_set.detectMultiScale(img_gray, minSize = (20,20)) #Detect object from grayscale with a minimum size of 20x20 pixels
    
    for (x, y, w, h) in d_object:
        img = cv2.rectangle (img_rgb, (x, y), (x + h, y + w), (0, 0, 255), 5) #Create boarder lines on detect objects
    
    cv2.imshow('Camera', img) #Display window with the name camera and display image
    if cv2.waitKey(10) & 0xFF == ord('q'): #Stop program when q is pressed
        break

cap.release()
cv2.destroyWindow('Camera')
        
    
