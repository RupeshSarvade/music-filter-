import cv2
import mediapipe as mp
import numpy as np
import pygame
import time


timeout = time.time() + 60*5



mp_face_mesh = mp.solutions.face_mesh.FaceMesh()


cap = cv2.VideoCapture(0)


# # set camera frame rate to 30 fps
# cap.set(cv2.CAP_PROP_FPS, 30)


pygame.mixer.init()

pygame.mixer.music.load("audio.mp3")


mouth_open = False

while True:
    ret, frame = cap.read()

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    results = mp_face_mesh.process(frame_rgb)

    # if landmarks are detected
    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            mouth_landmarks = face_landmarks.landmark[10:19]

            mouth_np = np.array([[lmk.x, lmk.y, lmk.z] for lmk in mouth_landmarks])

            # calculate the distance between the upper and lower lip
            upper_lip = mouth_np[2]
            lower_lip = mouth_np[6]
            dist = np.linalg.norm(upper_lip - lower_lip)

            # draw the mouth landmarks on the frame
            # for landmark in mouth_landmarks:
            #     x = int(landmark.x * frame.shape[1])
            #     y = int(landmark.y * frame.shape[0])
            #     cv2.circle(frame, (x, y), 0, (0, 0, 0), -1)


            # if the distance between the upper and lower lip is greater than a threshold value, assume the mouth is open
            if dist > 0.035 and not mouth_open:
                # set the flag to indicate that the mouth is open
                mouth_open = True




                # play the audio file if it's not already playing
                if not pygame.mixer.music.get_busy():
                    pygame.mixer.music.play(-1)  # set the loop to -1 to play the file continuously
            elif dist <= 0.035 and mouth_open:
                mouth_open = False
                cv2.putText(frame, "Mouth Closed", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)



    cv2.imshow('frame', frame)


    if cv2.waitKey(1) & 0xFF == ord('q'):
        break


    # if time.time() > timeout:
    #     break





pygame.mixer.music.stop()
cap.release()
cv2.destroyAllWindows()
