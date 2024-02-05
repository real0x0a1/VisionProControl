#!/usr/bin/python3

# -*- Author: Ali (Real0x0a1) -*-
# -*- Description: This project allows users to control their computer using eye gaze and finger movements. just like Apple VisionPro -*-

import cv2
import mediapipe as mp
import pyautogui
import math

# Initialize camera
cam = cv2.VideoCapture(0)

# Initialize Face Mesh and Hands modules from Mediapipe
face_mesh = mp.solutions.face_mesh.FaceMesh(refine_landmarks=True)
hands = mp.solutions.hands.Hands()

# Get screen size
screen_w, screen_h = pyautogui.size()

# Set the speed factor for mouse movement
speed_factor = 2  # Adjust this value to increase/decrease the speed
scroll_factor = 0.09  # Adjust this value based on scrolling sensitivity

# Initialize variables for scrolling and clicking
prev_thumb_y = 0
scrolling = False
clicking = False

# Main loop
while True:
    # Read a frame from the camera
    _, frame = cam.read()

    # Flip the frame horizontally
    frame = cv2.flip(frame, 1)

    # Convert the BGR frame to RGB
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Process the frame using Face Mesh
    face_output = face_mesh.process(rgb_frame)

    # Process the frame using Hands
    hands_output = hands.process(rgb_frame)

    # Get facial landmark points
    landmark_points = face_output.multi_face_landmarks

    # Get hand landmark points
    hand_landmarks = hands_output.multi_hand_landmarks

    # Get frame dimensions
    frame_h, frame_w, _ = frame.shape

    # Move the mouse cursor based on eye gaze
    if landmark_points:
        landmarks = landmark_points[0].landmark
        for id, landmark in enumerate(landmarks[474:478]):
            x, y = int(landmark.x * frame_w), int(landmark.y * frame_h)
            cv2.circle(frame, (x, y), 3, (0, 255, 0))
            if id == 1:
                screen_x, screen_y = int(
                    landmark.x * screen_w), int(landmark.y * screen_h)
                pyautogui.moveTo(screen_x, screen_y,
                                 duration=0.1 / speed_factor)

    # Check if index finger and thumb are touching or close for scrolling and clicking
    if hand_landmarks:
        for handLms in hand_landmarks:
            index_tip = handLms.landmark[8]  # Index finger tip
            thumb_tip = handLms.landmark[4]  # Thumb tip

            # Calculate the distance between thumb and index finger
            distance = math.sqrt((thumb_tip.x - index_tip.x)
                                 ** 2 + (thumb_tip.y - index_tip.y) ** 2)

            index_x, index_y = int(
                index_tip.x * frame_w), int(index_tip.y * frame_h)
            thumb_x, thumb_y = int(
                thumb_tip.x * frame_w), int(thumb_tip.y * frame_h)

            cv2.circle(frame, (index_x, index_y), 3, (0, 255, 255))
            cv2.circle(frame, (thumb_x, thumb_y), 3, (0, 255, 255))

            # Check the vertical distance between index finger and thumb for scrolling
            thumb_move = thumb_y - prev_thumb_y

            # Check the distance for clicking
            if distance < 0.05:
                if not clicking:
                    clicking = True
                    pyautogui.click()
            else:
                clicking = False

            # Check the distance and movement for scrolling
            if distance < 0.05 and abs(thumb_move):
                scrolling = True
                pyautogui.scroll(thumb_move * scroll_factor)
            else:
                scrolling = False

            prev_thumb_y = thumb_y

    # Display the frame
    cv2.imshow('VISIONPRO', frame)

    # Break the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the camera and close the OpenCV window
cam.release()
cv2.destroyAllWindows()
