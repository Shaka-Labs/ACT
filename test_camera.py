import cv2

# Try to open the default camera
cam = cv2.VideoCapture(1)

if not cam.isOpened():
    print("Cannot open camera")
else:
    # Try to capture a frame
    ret, frame = cam.read()
    if not ret:
        print("Can't receive frame (stream end?). Exiting ...")
    else:
        # If a frame is read correctly, save it as 'frame.png'
        cv2.imwrite('frame.png', frame)
        print("Frame captured and saved.")

    # Release the camera
    cam.release()

# Close all OpenCV windows
cv2.destroyAllWindows()
