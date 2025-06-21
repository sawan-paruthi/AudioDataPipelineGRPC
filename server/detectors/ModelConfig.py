import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'panns_inference'))
from panns_inference import *


class ModelConfig:
    def __init__(self):
        pass

    def get_model(self, name):
        model_map = {
            
            'Cnn6': lambda: Cnn6(sample_rate=8000, window_size=1024, 
                                hop_size=320, mel_bins=64, fmin=50, fmax=4000, 
                                classes_num=527),

            'Cnn10': lambda: Cnn10(sample_rate=8000, window_size=1024, 
                                hop_size=320, mel_bins=64, fmin=50, fmax=4000, 
                                classes_num=527),
            
            'Cnn14': lambda: Cnn14(sample_rate=8000, window_size=1024, 
                                hop_size=320, mel_bins=64, fmin=50, fmax=8000, 
                                classes_num=527),

            'Cnn14_8k': lambda: Cnn14_8k(sample_rate=8000, window_size=256, 
                                        hop_size=80, mel_bins=64, fmin=50, fmax=4000, 
                                        classes_num=527),
            'Cnn14_16k': lambda: Cnn14_16k(sample_rate=16000, window_size=512, 
                                        hop_size=160, mel_bins=64, fmin=50, fmax=8000, 
                                        classes_num=527),

            'Cnn14_DecisionLevelAtt': lambda: Cnn14_DecisionLevelAtt(sample_rate=16000, window_size=1024, 
                                                                    hop_size=160, mel_bins=64, fmin=50, fmax=8000, 
                                                                    classes_num=527),

            'Cnn14_DecisionLevelMax': lambda: Cnn14_DecisionLevelMax(sample_rate=32000, window_size=1024, 
                                                                    hop_size=320, mel_bins=64, fmin=50, fmax=14000, 
                                                                    classes_num=527, interpolate_mode='nearest'),

            'Cnn14_emb512': lambda: Cnn14_emb512(sample_rate=16000, window_size=1024, 
                                                hop_size=160, mel_bins=64, fmin=50, fmax=8000, 
                                                classes_num=527),
                                                
            'Cnn14_emb128': lambda: Cnn14_emb128(sample_rate=16000, window_size=1024, 
                                                hop_size=160, mel_bins=64, fmin=50, fmax=8000, 
                                                classes_num=527),

            'Cnn14_emb32': lambda: Cnn14_emb32(sample_rate=8000, window_size=1024, 
                                            hop_size=80, mel_bins=64, fmin=50, fmax=4000, 
                                            classes_num=527),

            'DaiNet19': lambda: DaiNet19(sample_rate=8000, window_size=1024, 
                                        hop_size=160, mel_bins=64, fmin=50, fmax=4000, 
                                        classes_num=527),

            'LeeNet11': lambda: LeeNet11(sample_rate=8000, window_size=1024, 
                                        hop_size=160, mel_bins=64, fmin=50, fmax=4000, 
                                        classes_num=527),

            'LeeNet24': lambda: LeeNet24(sample_rate=8000, window_size=1024, 
                                        hop_size=160, mel_bins=64, fmin=50, fmax=4000, 
                                        classes_num=527),

            'MobileNetV1': lambda: MobileNetV1(sample_rate=8000, window_size=1024, 
                                        hop_size=160, mel_bins=64, fmin=50, fmax=4000, 
                                        classes_num=527),

            'MobileNetV2': lambda: MobileNetV2(sample_rate=8000, window_size=1024, 
                                        hop_size=160, mel_bins=64, fmin=50, fmax=4000, 
                                        classes_num=527),

            'Res1dNet31': lambda: Res1dNet31(sample_rate=8000, window_size=1024, 
                                            hop_size=160, mel_bins=64, fmin=50, fmax=4000, 
                                            classes_num=527),

            'Res1dNet51': lambda: Res1dNet51(sample_rate=8000, window_size=1024, 
                                            hop_size=160, mel_bins=64, fmin=50, fmax=4000, 
                                            classes_num=527),

            'ResNet22': lambda: ResNet22(sample_rate=8000, window_size=1024, 
                                        hop_size=160, mel_bins=64, fmin=50, fmax=4000, 
                                        classes_num=527),

            'ResNet38': lambda: ResNet38(sample_rate=8000, window_size=1024, 
                                        hop_size=160, mel_bins=64, fmin=50, fmax=4000, 
                                        classes_num=527),

            'ResNet54': lambda: ResNet54(sample_rate=8000, window_size=1024, 
                                        hop_size=160, mel_bins=64, fmin=50, fmax=4000, 
                                        classes_num=527),

            'Wavegram_Cnn14': lambda: Wavegram_Cnn14(sample_rate=8000, window_size=1024, 
                                                    hop_size=160, mel_bins=64, fmin=50, fmax=4000, 
                                                    classes_num=527),

            'Wavegram_Logmel_Cnn14': lambda: Wavegram_Logmel_Cnn14(sample_rate=8000, window_size=1024, 
                                                                    hop_size=320, mel_bins=64, fmin=50, fmax=4000, 
                                                                    classes_num=527),
        }

        if name not in model_map:
            raise ValueError(f"Model '{name}' not found in configs")

        return model_map[name]()
