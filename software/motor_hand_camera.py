#media pipe

"""
Imports
"""

import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import numpy as np
import serial

import serial.tools.list_ports

ports = serial.tools.list_ports.comports()

for port in ports:
    print(port.device, port.description)

arduino = serial.Serial('COM7', 115200)

#ANLGE Formula
def angle(a, b,c):
    ba = np.array(a)-np.array(b) # MCP - PIPo
    bc = np.array(c) - np.array(b) # DIP - PIP
    
    cos_theta = np.dot(ba,bc) / (
        np.linalg.norm(ba) * np.linalg.norm(bc)
    )

    return np.degrees(np.arccos(cos_theta))

#DISTANCE
def dist(a,b):
    return np.linalg.norm(a-b)


def map_finger(angle, closed_angle, open_angle):
    angle = np.clip(angle, closed_angle, open_angle)
    return int(np.interp(angle, [closed_angle,open_angle],[160, 20]))
    

open_angles = None
closed_angles = None

#load model for hand
base_options = python.BaseOptions(
    model_asset_path = "hand_landmarker.task"
)

options = vision.HandLandmarkerOptions(
    base_options = base_options, 
    num_hands = 2)

detector = vision.HandLandmarker.create_from_options(options)

#Camera

cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

#Convert Frame for Media pipe

    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    mp_image = mp.Image(
        image_format = mp.ImageFormat.SRGB,
        data= rgb_frame
    )

    #result now has hand info
    result = detector.detect(mp_image)

    # if result.hand_landmarks:
    #     for hand_landmarks in result.hand_landmarks:
    #         for landmark in hand_landmarks:
    #             print(landmark.x, landmark.y, landmark.z)
        
    h, w, _ = frame.shape

    # px = int(landmark.x * w)
    # py = int(landmark.y * h)

    if result.hand_landmarks:
        h,w,_ = frame.shape
        for hand_landmarks in result.hand_landmarks:
            for landmark in hand_landmarks:
                cx = int(landmark.x * w) 
                cy = int(landmark.y * h)

                cv2.circle(frame,(cx,cy),5,(0, 255,0),-1)
    if result.hand_world_landmarks:
        lm = result.hand_world_landmarks[0]
        a = [lm[5].x,lm[5].y, lm[5].z]
        b = [lm[6].x,lm[6].y, lm[6].z]
        c = [lm[7].x,lm[7].y, lm[7].z]
        ang1 = angle(a,b,c)

        a = [lm[9].x,lm[9].y, lm[9].z]
        b = [lm[10].x,lm[10].y, lm[10].z]
        c = [lm[11].x,lm[11].y, lm[11].z]
        ang2 = angle(a,b,c)

        a = [lm[13].x,lm[13].y, lm[13].z]
        b = [lm[14].x,lm[14].y, lm[14].z]
        c = [lm[15].x,lm[15].y, lm[15].z]
        ang3 = angle(a,b,c)

        a = [lm[17].x,lm[17].y, lm[17].z]
        b = [lm[18].x,lm[18].y, lm[18].z]
        c = [lm[19].x,lm[19].y, lm[19].z]
        ang4 = angle(a,b,c)

        a = [lm[2].x,lm[2].y, lm[2].z]
        b = [lm[3].x,lm[3].y, lm[3].z]
        c = [lm[4].x,lm[4].y, lm[4].z]
        ang5 = angle(a,b,c)

        current_angles = [ang1, ang2, ang3, ang4, ang5]

        if open_angles is not None and closed_angles is not None:
            index = map_finger(ang1, closed_angles[0], open_angles[0])
            middle = map_finger(ang2, closed_angles[1], open_angles[1])
            ring = map_finger(ang3, closed_angles[2], open_angles[2])
            pinky = map_finger(ang4, closed_angles[3], open_angles[3])
            thumb = map_finger(ang5, closed_angles[4], open_angles[4])

            # packet = f"{index},{middle},{ring},{pinky},{thumb}\n"
            packet = bytes([index, middle,ring, pinky, thumb])
            arduino.write(packet)

            cv2.putText(
                frame,
                f"Index: {index:.1f}", (400,40), cv2.FONT_HERSHEY_COMPLEX, 1, (0,255,0), 2)
            
            cv2.putText(
                frame,
                f"Middle: {middle:.1f}", (400,80), cv2.FONT_HERSHEY_COMPLEX, 1, (0,255,0), 2)
            
            cv2.putText(
                frame,
                f"Ring: {ring:.1f}", (400,120), cv2.FONT_HERSHEY_COMPLEX, 1, (0,255,0), 2)
            
            cv2.putText(
                frame,
                f"Pinky: {pinky:.1f}", (400,160), cv2.FONT_HERSHEY_COMPLEX, 1, (0,255,0), 2)
        
            cv2.putText(
                    frame,
                    f"Thumb: {thumb:.1f}", (400,200), cv2.FONT_HERSHEY_COMPLEX, 1, (0,255,0), 2)

        cv2.putText(
            frame,
            f"Index: {ang1:.1f}", (50,40), cv2.FONT_HERSHEY_COMPLEX, 1, (0,255,0), 2)
        
        cv2.putText(
            frame,
            f"Middle: {ang2:.1f}", (50,80), cv2.FONT_HERSHEY_COMPLEX, 1, (0,255,0), 2)
        
        cv2.putText(
            frame,
            f"Ring: {ang3:.1f}", (50,120), cv2.FONT_HERSHEY_COMPLEX, 1, (0,255,0), 2)
        
        cv2.putText(
            frame,
            f"Pinky: {ang4:.1f}", (50,160), cv2.FONT_HERSHEY_COMPLEX, 1, (0,255,0), 2)

        cv2.putText(
                frame,
                f"Thumb: {ang5:.1f}", (50,200), cv2.FONT_HERSHEY_COMPLEX, 1, (0,255,0), 2)    
    
    cv2.imshow("Hand Tracker", frame)

    key = cv2.waitKey(1) & 0xFF
    
    if key == ord('o'):
        open_angles = current_angles.copy()
        print("Open calibration saved:", open_angles)

    elif key == ord('c'):
        closed_angles = current_angles.copy()
        print("Closed calibration saved:",closed_angles)

    elif key == ord('q'):
        break



    

    

cap.release()
cv2.destroyAllWindows()