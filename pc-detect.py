from mss import mss
import torch
import cv2
import numpy as np

MONITOR_WIDTH = 1920
MONITOR_HEIGHT = 1080
MONITOR_SCALE = 2
region = (int(MONITOR_WIDTH/2-MONITOR_WIDTH/MONITOR_SCALE/2),int(MONITOR_HEIGHT/2-MONITOR_HEIGHT/MONITOR_SCALE/2),int(MONITOR_WIDTH/2+MONITOR_WIDTH/MONITOR_SCALE/2),int(MONITOR_HEIGHT/2+MONITOR_HEIGHT/MONITOR_SCALE/2))

model = torch.hub.load(r'C:\Users\balbo\Desktop\agro-photo\yolov5', 'custom', path=r'C:\Users\balbo\Desktop\agro-photo\out\exp3\weights\best.pt', source='local')
model.conf = 0.5
model.maxdet = 5
model.apm = True 

with mss() as stc:
    while True:
        frame = np.array(stc.grab(region))
        df = model(frame, size=640).pandas().xyxy[0]

        for i in range(0, model.maxdet):
            try:
                xmin = int(df.iloc[i,0])
                ymin = int(df.iloc[i,1])
                xmax = int(df.iloc[i,2])
                ymax = int(df.iloc[i,3])

                cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), (120, 60, 0), 2)

                class_index = int(df.iloc[i, 5])
                class_name = model.names[class_index]
                cv2.putText(frame, class_name, (xmin, ymin - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (120, 60, 0), 2)
            except:
                print("",end="")

        cv2.imshow("Agro Photo",frame)
        if(cv2.waitKey(1) == ord('k')):
            cv2.destroyAllWindows()
            break