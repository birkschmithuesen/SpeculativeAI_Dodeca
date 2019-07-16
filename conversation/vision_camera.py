"""
This module contains objects needed to capture camera frames, send osc output and further
functionality needed for https://github.com/birkschmithuesen/SpeculativeArtificialIntelligence .
"""
import platform
import cv2
from PIL import Image

class Camera():
    """
    Camera contains all functionality necessary to capture frames
    from a camera for further processing in keras
    """

    def __init__(self, frame_width, frame_height, fps=120):
        """
        Creates new camera object with given configuration.
        :param frame_width: Width of camera frame
        :param frame_height: Height of camera frame
        :param fps: Frames per second setting of camera
        :return: New Camera object
        """
        self.frame_width = frame_width
        self.frame_height = frame_height
        id = 0
        if any(platform.win32_ver()):
            self.video_capture = cv2.VideoCapture(id + cv2.CAP_DSHOW)
        else:
            self.video_capture = cv2.VideoCapture(id)
        if not self.video_capture.isOpened():
            raise Exception("Could not open video device")
        # Set properties. Each returns === True on success (i.e. correct resolution)
        self.video_capture.set(cv2.CAP_PROP_FRAME_WIDTH, frame_width)
        self.video_capture.set(cv2.CAP_PROP_FRAME_HEIGHT, frame_height)
        self.actual_frame_width = self.video_capture.get(cv2.CAP_PROP_FRAME_WIDTH)
        self.actual_frame_height = self.video_capture.get(cv2.CAP_PROP_FRAME_HEIGHT)
        self.video_capture.set(cv2.CAP_PROP_FPS, fps)
        self.video_capture.set(cv2.CAP_PROP_SETTINGS,0)
        print("Initializing camera")
        print("actual_frame_width:" + str(self.actual_frame_width))
        print("actual_frame_height:" + str(self.actual_frame_height))


    def show_capture(self):
        """
        Shows video frames output in window. Stop by pressing 'q' on keyboard.
        """
        for frames in self:
            (frame, pil_im) = frames
            # Display the resulting frame
            frame = self.crop_frame(frame)
            cv2.imshow('frame', frame)
            if cv2.waitKey(20) & 0xFF == ord('q'):
                break
        cv2.destroyAllWindows()
        self.release()

    def crop_frame(self, frame):
        """
        Crop given opencv frame to set camera frame size
        :param frame: Frame to be cropped
        :return: cropped frame
        """
        x = (self.actual_frame_width - self.frame_width)/2
        y = (self.actual_frame_height - self.frame_height)/2
        x = int(x)
        y = int(y)
        return frame[y:y+self.frame_height, x:x+self.frame_width]

    def release(self):
        """
        Release the camera capture.
        """
        self.video_capture.release()

    def __iter__(self):
        return self

    def __next__(self):
        """
        This will enable iterating over camera frames.
        :return: Tupel consisting of next opencv frame and python image library frame
        """
        ret, frame = self.video_capture.read()
        frame = self.crop_frame(frame)
        cv2_im = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil_im = Image.fromarray(cv2_im)
        return frame, pil_im
