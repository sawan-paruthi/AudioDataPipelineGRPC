import librosa
import numpy as np
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'panns_inference'))

sys.path.append(os.path.join(os.path.dirname(__file__)))
from panns_inference import *

from ModelConfig import ModelConfig

class PannsInference():

    def __init__(self):
        self.model_config = ModelConfig()
        

    def load_model(self, model_name):
        try:
            model_class_name = model_name[:-14]
            model = self.model_config.get_model(model_class_name)
            return model
        except:
            raise ValueError("Error Loading Model, Model Class not available")
 
        
    def sound_event_detection(self, audio_path, checkpoint_path, model):
        
        (audio, _) = librosa.core.load(audio_path, sr=32000, mono=True)
        audio = audio[None, :] 

        sed = SoundEventDetection(model=model, checkpoint_path=checkpoint_path)
        framewise_output = sed.inference(audio)

        frame_time = 10 / 1000  
        threshold = 0.01  

        event_detections = []
        for class_idx in range(len(labels)):
            active = False
            start_time = None

            for frame_idx, probability in enumerate(framewise_output[0, :, class_idx]):
                time_stamp = frame_idx * frame_time  

                if probability > threshold and not active:
                    active = True
                    start_time = time_stamp  

                elif probability < threshold and active:
                    active = False
                    end_time = time_stamp  
                    event_detections.append((labels[class_idx], start_time, end_time))


        print("\nSound Events Detected:")
        for label, start, end in event_detections:
            print(f"{label}: {start:.2f}s - {end:.2f}s")

    def audio_tagging(self, audio_path, checkpoint_path, model):
        
        print('------ Audio tagging ------')
        (audio, _) = librosa.core.load(audio_path, sr=32000, mono=True)
        audio = audio[None, :] 

        at = AudioTagging(model=model, checkpoint_path=checkpoint_path, weights_only=False)
        clipwise_output, embedding = at.inference(audio)

        tagging_results = {labels[i]: float(clipwise_output[0, i]) for i in range(len(labels))}
        sorted_results = sorted(tagging_results.items(), key=lambda x: x[1], reverse=True)  # Sort by confidence score

        print("\n Audio Tagging Results:")
        for label, score in sorted_results[:10]:  
            print(f"{label}: {score:.4f}")

if __name__ == "__main__":
    audio_path = 'bike.wav'
    panns = PannsInference()
    # model.sound_event_detection(audio_path)
    checkpoint_path = 'F:\\work\\AudioDetectorGRPC\\server\\checkpoints\\panns\\Cnn10_mAP=0.380.pth'
    model = panns.load_model('Cnn10_mAP=0.380.pth')
    panns.audio_tagging(audio_path, checkpoint_path, model) 
    # model.sound_event_detection(audio_path, checkpoint_path)  
    




