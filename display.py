import cv2


def display_image(image):
    """
        Display an image.

        Args:
            image: A Numpy array representing an image
    """
    cv2.imshow('image', image)
    cv2.waitKey(0)


def display_video():
    """
        Test YOLOv3 object detection model on live camera feed.
    """
    capture = cv2.VideoCapture(0)

    # set height and width of frame
    ret = capture.set(3, 640)
    ret = capture.set(4, 480)

    counter = 0;
    seconds = 0;
    frame_rate = 0;
    while True:
        if(counter == 0):
            start = time.time()
        elif(counter == 10):
            end = time.time()
            seconds = start - end
            frame_rate = 10/seconds
            counter = -1
        ret, frame = capture.read()
        counter += 1
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_size = 1
        font_color = (255, 255, 255)
        font_weight = 2
        font_line = cv2.LINE_AA
        cv2.putText(frame, str(frame_rate), (10, 10), font, font_size, font_color, font_weight, font_line)

        # predict_bounding_boxes function currently not implemented...
        obj_det_frame = predict_bounding_boxes(frame)
        cv2.imshow('frame', obj_det_frame)

        if (cv2.waitKey(1) & 0xFF == ord('q')):
            break

    capture.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    display_video()
