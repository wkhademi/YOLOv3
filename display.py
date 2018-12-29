import cv2


def display_image(image):
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

    while(True):
        ret, frame = capture.read()

        # predict_bounding_boxes function currently not implemented...
        obj_det_frame = predict_bounding_boxes(frame)
        cv2.imshow('frame', obj_det_frame)

        if (cv2.waitKey(1) & 0xFF == ord('q')):
            break

    capture.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    display_video()
