import torch
from matplotlib import pyplot as plt
import numpy as np
import cv2
import pandas as pd
import uuid   # Unique identifier
import os
import time

model = torch.hub.load('ultralytics/yolov5', 'custom', path='yolov5/runs/train/exp/weights/last.pt', force_reload=True)

cap = cv2.VideoCapture(0)
var = ""
employee_score = 100;
employee_name = "Digvijay"
employee_id = 123
while cap.isOpened():
    ret, frame = cap.read()
    cv2.rectangle(frame, (0, 0), (525, 73), (245, 117, 16), -1)
    cv2.putText(frame, "Employee Name:", (15, 12),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
    cv2.putText(frame, "Employee I.D.:",
                        (10, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
    cv2.putText(frame, employee_name, (180, 12),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
    cv2.putText(frame, str(employee_id),
                        (180, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
    # Make detections 
    results = model(frame)
    cv2.imshow('YOLO', np.squeeze(results.render()))
    
    df = results.pandas().xyxy[0]
    d_count=df['name'].str.contains('drowsy').sum()
    if d_count>0:
        employee_score = employee_score - d_count
        if d_count==1:
            print("sleeping")
        else:
            print("awake")
    
    
    if cv2.waitKey(10) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()