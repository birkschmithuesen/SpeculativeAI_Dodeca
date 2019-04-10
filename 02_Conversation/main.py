"""
Main file that executes the visual conversation part.
"""
from conversation.camera import Camera

camera = Camera(320, 180)
camera.show_capture()
