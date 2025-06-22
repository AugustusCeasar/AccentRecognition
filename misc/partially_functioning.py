import argparse
import torchaudio
import requests

import os
from speechbrain.pretrained.interfaces import foreign_class

def _hide_seek(obj):
    """wrapper to turn requests object to a torchaudio compatible format"""
    class _wrapper:
        def __init__(self, obj):
            self.obj = obj

        def read(self, n):
            return self.obj.read(n)

    return _wrapper(obj)

def default_audio_preprocess(audio):
    """For a simple demo, prioritizing a single audio channel is a workable solution. 
    More complex options such as cdownmixing could be considered if it was determined to significanlty benefit an interaction with specific audio sources."""
    return audio[0, :]



def get_accent_estimate_from_url(url: str, max_frames=9000) -> tuple[str, float]:
    """I would like to use this method, rather than relying on downloading the file to temp. 
    However, torchaudio sometimes inexplicably failed to parse some of my test files when read directly from url, without apparent pattern.
    Given more time I would investigate why, for now we simply download the file using curl.
    """

    with requests.get(url, stream=True) as response:
    
        signal, fs = torchaudio.load(_hide_seek(response.raw), num_frames=max_frames)
        signal = default_audio_preprocess(signal)
        
        classifier = foreign_class(source="Jzuluaga/accent-id-commonaccent_xlsr-en-english", pymodule_file="custom_interface.py", classname="CustomEncoderWav2vec2Classifier")

        out_prob, score, index, text_lab =  classifier.classify_batch(signal)
        
        return text_lab[0], score[0]
    
    
def get_accent_estimate_from_file(fn: str, max_frames=9000) -> tuple[str, float]:
    classifier = foreign_class(source="Jzuluaga/accent-id-commonaccent_xlsr-en-english", pymodule_file="custom_interface.py", classname="CustomEncoderWav2vec2Classifier")

    signal, fs = torchaudio.load(fn, num_frames=max_frames)
    signal = default_audio_preprocess(signal)
    
    out_prob, score, index, text_lab =  classifier.classify_batch(signal)
        
    return text_lab[0], score[0]
    

def main():
    parser = argparse.ArgumentParser(prog='Accent Detection',
                                     description='Provide a url to a video file on the internet and recieve an estimation of what accent the main speaker is using.')
    parser.add_argument('-url', default="", nargs='?',
                        help="url pointing to a video (or audio) file")
    parser.add_argument('-file', default="", nargs='?',
                        help="path to a dowloaded video (or audio) file")
    parser.add_argument('--remove-file', action="store_true",
                        help="delete -file after evaluating it. For use in the main script that generates a temp input file.")
                        
    args = parser.parse_args()
        
    if args.url:
        accent, confidence = get_accent_estimate_from_url(args.url)
        
        print(f"The provided video is likely in an accent from {accent}. The system is {confidence*100:.1f}% confident.")
    
    if args.file and os.path.isfile(args.file):
        accent, confidence = get_accent_estimate_from_file(args.file)
        
        print(f"The provided file is likely in an accent from {accent}. The system is {confidence*100:.1f}% confident.")
        
        if args.remove_file:
            os.remove(args.file)
        

if __name__ == "__main__":
    main()