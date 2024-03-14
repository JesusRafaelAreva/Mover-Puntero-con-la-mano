import cv2
import mediapipe as mp
import pyautogui

# Initialize the MediaPipe module for hand tracking
mp_hands = mp.solutions.hands
hands = mp_hands.Hands()

# Capture video from the camera
cap = cv2.VideoCapture(0)

# Create a window with an appropriate size to display the image
cv2.namedWindow("Hand Tracking")

# Flag to track if the finger was touching in the previous frame
finger_touching = False

while True:
    ret, frame = cap.read()

    # Convert the frame to RGB (MediaPipe requires input in RGB)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Detect hands in the frame
    results = hands.process(rgb_frame)

    if results.multi_hand_landmarks:
        # For simplicity, only consider the first detected hand
        hand_landmarks = results.multi_hand_landmarks[0]

        # Get the coordinates of the index finger tip (landmark point 8) and thumb tip (landmark point 4)
        index_finger_tip = hand_landmarks.landmark[8]
        thumb_tip = hand_landmarks.landmark[4]

        # Convert coordinates to pixel values
        x_index = int(index_finger_tip.x * frame.shape[1])
        y_index = int(index_finger_tip.y * frame.shape[0])
        x_thumb = int(thumb_tip.x * frame.shape[1])
        y_thumb = int(thumb_tip.y * frame.shape[0])

        # Calculate the distance between index finger and thumb tip
        distance = ((x_thumb - x_index) ** 2 + (y_thumb - y_index) ** 2) ** 0.5

        # If the distance is smaller than a threshold, consider it as a click
        if distance < 30:  # You can adjust the threshold as needed
            if not finger_touching:
                # Perform a click action
                pyautogui.click()
                finger_touching = True
        else:
            finger_touching = False

        # Move the mouse pointer to the detected finger tip position
        pyautogui.moveTo(x_index, y_index)

    # Show the image in the window
    cv2.imshow("Hand Tracking", frame)

    # Exit the loop if the 'q' key is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture and close the window
cap.release()
cv2.destroyAllWindows()
