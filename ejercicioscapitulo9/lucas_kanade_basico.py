#lucas_kanade_basico.py
import cv2
import numpy as np
cap = cv2.VideoCapture(0)
ret, old_frame = cap.read()
old_gray = cv2.cvtColor(old_frame, cv2.COLOR_BGR2GRAY)
p0 = cv2.goodFeaturesToTrack(old_gray, mask=None, maxCorners=100, qualityLevel=0.3, minDistance=7)
while True:
    ret, frame = cap.read()
    frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    p1, st, err = cv2.calcOpticalFlowPyrLK(old_gray, frame_gray, p0, None)
    for i,(new,old) in enumerate(zip(p1[st==1],p0[st==1])):
        a,b = new.ravel(); c,d = old.ravel()
        cv2.line(frame, (int(a),int(b)), (int(c),int(d)), (0,255,0), 2)
    cv2.imshow('Lucas-Kanade', frame)
    if cv2.waitKey(30) & 0xFF == ord('q'): break
    old_gray = frame_gray.copy(); p0 = p1.reshape(-1,1,2)
