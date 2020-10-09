"""
This module contains objects needed to capture camera frames, send osc output and further
functionality needed for https://github.com/birkschmithuesen/SpeculativeArtificialIntelligence .
"""
import os
import platform
import cv2
import numpy as np
from PIL import Image

def tx2_usb_reset():
    """
    reset all usb ports on Jetson Tx2 (power off/on)
    """
    for usb_port in ["001/011"]:
        os.system("usbreset /dev/bus/usb/{}".format(usb_port))

class Camera():
    """
    Camera contains all functionality necessary to capture frames
    from a camera for further processing in keras
    """

    def __init__(self, frame_width, frame_height, frame_section_width, frame_section_height, fps=120):
        """
        Creates new camera object with given configuration.
        :param frame_width: Width of camera frame
        :param frame_height: Height of camera frame
        :param frame_section_width: Width of camera frame section that will be scaled
        :param frame_section_height: Height of camera frame section that will be scaled
        :param fps: Frames per second setting of camera
        :return: New Camera object
        """
        self.frame_width = frame_width
        self.frame_height = frame_height
        self.frame_section_width = frame_section_width
        self.frame_section_height = frame_section_height
        self.get_video_capture()
        # Set properties. Each returns === True on success (i.e. correct
        # resolution)
        #self.video_capture.set(cv2.CAP_PROP_FRAME_WIDTH, frame_section_width)
        #self.video_capture.set(cv2.CAP_PROP_FRAME_HEIGHT, frame_section_height)
        self.actual_frame_width = self.video_capture.get(
            cv2.CAP_PROP_FRAME_WIDTH)
        self.actual_frame_height = self.video_capture.get(
            cv2.CAP_PROP_FRAME_HEIGHT)
        self.video_capture.set(cv2.CAP_PROP_FPS, fps)
        self.video_capture.set(cv2.CAP_PROP_SETTINGS, 0)
        self.video_capture.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.25) #0.25 is off, 0.75 is on
        self.video_capture.set(cv2.CAP_PROP_EXPOSURE, 0.005) #0.00175 for sunlight / 
        self.video_capture.set(cv2.CAP_PROP_BRIGHTNESS, 0.5)
        mask_base = np.zeros((self.frame_height, self.frame_width), np.uint8)
        self.circle_mask = cv2.circle(mask_base, (int(self.frame_height/2), int(self.frame_width/2)), int(self.frame_height/2), (255, 255, 255), thickness=-1)
        print("Initializing camera")
        print("actual_frame_width:" + str(self.actual_frame_width))
        print("actual_frame_height:" + str(self.actual_frame_height))
        print("actual exposure:" + str(self.video_capture.get(cv2.CAP_PROP_EXPOSURE)))
        print("actual brightness:" + str(self.video_capture.get(cv2.CAP_PROP_BRIGHTNESS)))

    def get_video_capture(self):
        id = 1
        if any(platform.win32_ver()):
            self.video_capture = cv2.VideoCapture(id + cv2.CAP_DSHOW)
            print("Detected Windows platform")
        elif any((any(x) for x in platform.mac_ver())):
            id = 0
            self.video_capture = cv2.VideoCapture(id)
            print("Detected Mac platform")
        else:
            self.video_capture = cv2.VideoCapture(id)
            print("Detected other platform (likely Linux)")
        if not self.video_capture.isOpened():
            raise Exception("Could not open video device")

    def show_capture(self):
        """
        Shows video frames output in window. Stop by pressing 'q' on keyboard.
        """
        for frames in self:
            (frame, pil_im) = frames
            # Display the resulting frame
            frame = self.crop_frame(frame)
            cv2.namedWindow("dodeca", cv2.WND_PROP_FULLSCREEN)
            cv2.setWindowProperty("dodeca",cv2.WND_PROP_FULLSCREEN,cv2.WINDOW_FULLSCREEN)
            cv2.imshow("dodeca", frame)
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
        x = (self.actual_frame_width - self.frame_section_width) / 2
        y = (self.actual_frame_height - self.frame_section_height) / 2
        x = int(x)
        y = int(y)
        cropped_frame = frame[y:y + self.frame_section_height, x:x + self.frame_section_width]
        resized_frame = cv2.resize(cropped_frame, (self.frame_height, self.frame_width), interpolation=cv2.INTER_AREA)
        masked_frame = cv2.bitwise_and(resized_frame, resized_frame, mask=self.circle_mask)
        return masked_frame


    def release(self):
        """
        Release the camera capture.
        """
        self.video_capture.release()

    def restart(self):
        """
        Restart the camera and if that doesn't work reset the usb port
        """
        if hasattr(self, "restarted"):
            print("Resetting USB port")
            tx2_usb_reset()
        try:
            print("Get video capture")
            self.get_video_capture()
        except Exception as e:
            print("Can not get video capture")
            print(e)

        self.restarted = True

    def __iter__(self):
        return self

    def __next__(self):
        """
        This will enable iterating over camera frames.
        :return: Tupel consisting of next opencv frame and python image library frame
        """
        ret, frame = None, None
        for i in range(60):
            ret, frame = self.video_capture.read()
            if frame is not None:
                break
            if i is 59:
                print("Restarting camera")
                self.restart()
        frame = self.crop_frame(frame)
        cv2_im = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil_im = Image.fromarray(cv2_im)
        return frame, pil_im
