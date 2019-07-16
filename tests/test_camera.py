import pytest
import cv2
from conversation.vision_camera import Camera

class TestCamera(object):
    def test_init_params(self):
        cam = Camera(320, 240, 30)
        assert cam
        assert cam.video_capture.get(cv2.CAP_PROP_FRAME_WIDTH) == 320
        assert cam.video_capture.get(cv2.CAP_PROP_FRAME_HEIGHT) == 240
        assert int(cam.video_capture.get(cv2.CAP_PROP_FPS)) == 30
        cam.release()

        cam = Camera(240, 480, 30)
        assert cam
        assert cam.video_capture.get(cv2.CAP_PROP_FRAME_WIDTH) == 640
        assert cam.video_capture.get(cv2.CAP_PROP_FRAME_HEIGHT) == 480
        assert int(cam.video_capture.get(cv2.CAP_PROP_FPS)) == 30
        cam.release()

    def test_cropping(self):
        cam = Camera(120, 80, 30)
        for frames in cam:
            cv2_im, pil_im = frames
            height, width, channels = cv2_im.shape
            assert width == 120
            assert height == 80
            assert channels == 3
            break
        cam.release()

    def test_iterator(self):
        cam = Camera(320, 180, 30)
        cam_itr = iter(cam)
        assert cam_itr
        for frames in cam_itr:
            cv2_im, pil_im = frames
            height, width, channels = cv2_im.shape
            assert width == 320
            assert channels == 3
            break
        cam.release()
