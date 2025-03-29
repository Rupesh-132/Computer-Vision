import cv2
from utils import get_limits
from PIL import Image


yellow = [0,255,255]  # yellow in rgb color space
cap = cv2.VideoCapture(0) # if your laptop has single camera

while True:
    
    ret,frame = cap.read()
    
    if not ret: # Returns false if it fails to read the frames successfully
        break  # Exit if no frame is read
    
    hsvImage = cv2.cvtColor(frame,cv2.COLOR_BGR2HSV)
    
    lowerLimit,upperLimit = get_limits(color=yellow)
    
    mask =  cv2.inRange(hsvImage,lowerLimit,upperLimit)
    
    mask_ = Image.fromarray(mask)
    
    bbox = mask_.getbbox()
    
    if bbox is not None:
        
        x1,y1,x2,y2 = bbox
        
        frame = cv2.rectangle(frame,(x1,y1),(x2,y2),(0,255,0),5)
    print(bbox)
    
    # Visualize the mask
    
    cv2.imshow("mask",frame)
    
    # Visualise the frame
    # cv2.imshow("frame",frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    

cv2.release()
cv2.destroyAllWindows()