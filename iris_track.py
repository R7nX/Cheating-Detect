from operator import rshift
import cv2 as cv 
import numpy as np
import mediapipe as mp 
import pyautogui
import math

mp_face_mesh = mp.solutions.face_mesh
face_cascade = cv.CascadeClassifier('haarcascade_frontalface_alt.xml')

LEFT_EYE =[ 362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385,384, 398 ]
# right eyes indices
RIGHT_EYE=[ 33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161 , 246 ] 
LEFT_IRIS = [474,475, 476, 477]
RIGHT_IRIS = [469, 470, 471, 472]

L_H_LEFT = [33]
L_H_RIGHT = [133]
R_H_LEFT = [362]
R_H_RIGHT = [263]

def euclidian_distance(vec1, vec2):
    x1, y1 = vec1.ravel()
    x2, y2 = vec2.ravel()
    distance = math.sqrt((x2-x1)**2 + (y2-y1)**2)
    return distance

def position(iris_center, right_vec, left_vec):
    center_right_distance = euclidian_distance(iris_center, right_vec)
    left_right_distance = euclidian_distance(right_vec, left_vec)
    ratio = center_right_distance/left_right_distance
    if ratio <= 1.18:
        position = 'left'
    elif ratio > 1.20 and ratio <= 1.33:
        position = 'center'
    else:
        position = 'right'
    return position, ratio
cap = cv.VideoCapture(1)

with mp_face_mesh.FaceMesh(
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
) as face_mesh:
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv.flip(frame, 1)
        rgb_frame = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
        faces = face_cascade.detectMultiScale(rgb_frame, minNeighbors=5, scaleFactor=1.1, minSize=(25, 25))
        if len(faces) > 1:
            print("The student must take the test alone!!!!!!!")

        img_h, img_w = frame.shape[:2]
        results = face_mesh.process(rgb_frame)


        if results.multi_face_landmarks:
            # print(results.multi_face_landmarks[0].landmark)
            mesh_points=np.array([np.multiply([p.x, p.y], [img_w, img_h]).astype(int) for p in results.multi_face_landmarks[0].landmark])
            # print(mesh_points.shape)
            # cv.polylines(frame, [mesh_points[LEFT_IRIS]], True, (0,255,0), 1, cv.LINE_AA)
            # cv.polylines(frame, [mesh_points[RIGHT_IRIS]], True, (0,255,0), 1, cv.LINE_AA)
            (l_cx, l_cy), l_radius = cv.minEnclosingCircle(mesh_points[LEFT_IRIS])
            (r_cx, r_cy), r_radius = cv.minEnclosingCircle(mesh_points[RIGHT_IRIS])
            center_left = np.array([l_cx, l_cy], dtype=np.int32)
            center_right = np.array([r_cx, r_cy], dtype=np.int32)

            mid_cx, mid_cy = (l_cx + r_cx) / 2, (l_cy + r_cy) / 2
            center_mid = np.array([mid_cx, mid_cy], dtype=np.int32)

            cv.circle(frame, center_left, int(l_radius), (255,0,255), 1, cv.LINE_AA)
            cv.circle(frame, center_right, int(r_radius), (255,0,255), 1, cv.LINE_AA)
            cv.circle(frame, mesh_points[R_H_RIGHT][0], 3, (255, 255, 255), -1, cv.LINE_AA)
            cv.circle(frame, mesh_points[L_H_RIGHT][0], 3, (255, 255, 255), -1, cv.LINE_AA)
            pos, ratio = position(center_right, mesh_points[R_H_RIGHT], mesh_points[L_H_RIGHT])
            print(pos, ratio)
            print(mid_cx, mid_cy)

            print(" ")
            if pos != "center":
                if (mid_cx >= 390 and mid_cy >= 247) or (mid_cx <= 235 and mid_cy >= 250):
                    print("look at the screenn!!!!!!!!!")
            else:
                if mid_cy >= 255:
                    print("look at the screen!!!!!!!!!!")



        cv.imshow('img', frame)
        key = cv.waitKey(1)
        if key ==ord('q'):
            break
cap.release()
cv.destroyAllWindows()