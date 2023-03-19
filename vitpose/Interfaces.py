import cv2

# Open the webcam
webcam = cv2.VideoCapture(0)

# Open the video file
video = cv2.VideoCapture('recorded video.mp4')

# Create a window to show both videos
cv2.namedWindow('Double Video', cv2.WINDOW_NORMAL)

# Initialize the position and size of the window
cv2.moveWindow('Double Video', 0, 0)
cv2.resizeWindow('Double Video', 1280, 720)

# read the first frame of the video
ret2, firstFrame = video.read()

# Variable to keep track of whether to play the second video or not
play_video = False

while True:
    # Read the next frame from the webcam
    ret1, frame1 = webcam.read()

    # Read the next frame from the video
    ret2, frame2 = video.read()

    # If either video has ended, break out of the loop
    if not ret1 or not ret2:
        break

    # Resize the frames to fit side-by-side in the window
    frame1 = cv2.resize(frame1, (640, 720))
    frame2 = cv2.resize(frame2, (640, 720))
    firstFrame = cv2.resize(firstFrame, (640, 720))

    # if the space bar has been pressed and toggle the play_video variable accordingly
    key = cv2.waitKey(1) & 0xFF
    if key == ord(' '):
        play_video = True

    # Concatenate the two frames horizontally based on the play_second variable
    if play_video:
        double_frame = cv2.hconcat([frame1, frame2])
    else:
        double_frame = cv2.hconcat([frame1, firstFrame])

    # Display the concatenated frame in the window
    cv2.imshow('Double Video', double_frame)

    # Wait for a key press
    if key == ord('q'):
        break

# Release the video capture objects and destroy the window
webcam.release()
video.release()
cv2.destroyAllWindows()