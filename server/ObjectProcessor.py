import os
import sys
import torch
from detectors.PannsInference import PannsInference
import logging

trackers_path = os.path.join(os.path.dirname(__file__), 'trackers')
sys.path.append(trackers_path)

class ObjectProcessor:
    def __init__(self, prefix):
        self.model = None  
        self.detector = None
        self.prefix = prefix
        self.panns = PannsInference()

    def load_model(self, model):
        # print(f"model name in object processor is {model}")
        self.detector = model


    def detect_objects(self, audio_path):
        # print(f"detector inside objectprocessor {self.detector}")
        
        # return detections
        if self.detector[:2]==f"{self.prefix}":
            # print("inside if")
            logging.info(f"ObjectProcessor: Model Loaded: {self.detector}")
            abs_path = os.path.dirname(os.path.abspath(__file__))
            model_path = os.path.join(abs_path, "checkpoints", "panns", self.detector[3:])
            model = self.panns.load_model(self.detector[3:])
            detections = self.panns.audio_tagging(audio_path, model_path, model)
            return detections
        
        elif self.detector[:3]=="sed":
            # print("inside if")
            logging.info(f"ObjectProcessor: Model Loaded: {self.detector}")
            abs_path = os.path.dirname(os.path.abspath(__file__))
            model_path = os.path.join(abs_path, "checkpoints", "yolo", self.detector[4:])
            model = self.panns.load_model(self.detector[4:])
            detections = self.panns.sound_event_detection(audio_path, model_path, model)
            return detections

        else:
            logging.error("ObjectProcessor: Model not available")
            raise ValueError(f"Model not available : {self.detector}")

