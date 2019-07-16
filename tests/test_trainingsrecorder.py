import os
import pytest
from pythonosc import udp_client
import trainingsrecorder
import time

PYTEST_CSV_PATH = "./data/pytest_trainingsset_dodeca.csv"

def file_len(fname):
    length = 0
    with open(fname) as f:
        for line in f:
            length += 1
    return length

class TestTrainingsrecorder(object):
    def test_osc_receiving(self):
        CLIENT = udp_client.SimpleUDPClient("localhost", trainingsrecorder.OSC_PORT)
        trainingsrecorder.start_recording()
        time.sleep(0.5)
        assert len(trainingsrecorder.trainingsset) == 0
        CLIENT.send_message("/record_sound_vector", [1,1,1,1,1])
        time.sleep(0.5)
        assert len(trainingsrecorder.trainingsset) == 1
        for i in range(2):
            CLIENT.send_message("/record_sound_vector", [1,1,1,1,1])
            time.sleep(0.05)
        assert len(trainingsrecorder.trainingsset) == 3
    def test_file_saving(self):
        trainingsrecorder.TRAININGS_SET_PATH = PYTEST_CSV_PATH
        if os.path.exists(PYTEST_CSV_PATH):
            os.remove(PYTEST_CSV_PATH)
        trainingsrecorder.stop_recording()
        time.sleep(0.5)
        assert trainingsrecorder.stop_event.isSet()
        trainingsrecorder.process_trainingsset()
        trainingsrecorder.save_to_disk()
        assert os.path.exists(PYTEST_CSV_PATH)
        assert file_len(PYTEST_CSV_PATH) == 3
        os.remove(PYTEST_CSV_PATH)
