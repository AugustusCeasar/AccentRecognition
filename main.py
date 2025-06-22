import argparse
import statistics
import os, warnings

with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    import torchaudio
    from speechbrain.pretrained.interfaces import foreign_class

def default_audio_preprocess(audio):
    """For a simple demo, prioritizing a single audio channel is a workable solution. 
    More complex options such as cdownmixing could be considered if it was determined to significanlty benefit an interaction with specific audio sources."""
    return audio[0, :]

READABLE_LABEL_DICT = {"us": "the us", "indian": "india", "african": "africa", "newzealand": "new zealand", "hongkong": "hong kong", "southatlandtic": "the south atlantic"}

def get_accent_estimate_from_file(fn: str, max_frames=54000, large_file_handling=False) -> tuple[str, float]:
    output_labels = []
    output_scores = []
    file_length = torchaudio.info(fn).num_frames

    if file_length == 0:
        raise ValueError(f"File {fn} has no readable audio channels")
    
    for start_frame in range(0, file_length if large_file_handling else max_frames, max_frames):
        print(f"{start_frame} - {start_frame+max_frames}")
        classifier = foreign_class(source="Jzuluaga/accent-id-commonaccent_xlsr-en-english", pymodule_file="custom_interface.py", classname="CustomEncoderWav2vec2Classifier")


        signal, fs = torchaudio.load(fn, num_frames=max_frames, frame_offset=start_frame)
        signal = default_audio_preprocess(signal)
        

        
        out_prob, score, index, text_lab =  classifier.classify_batch(signal)
        output_labels.append(text_lab[0])
        output_scores.append(score[0].item())
    
    final_label = max(set(output_labels), key=output_labels.count) # most common entry, defaulting to the first in ties
    final_score = statistics.mean(output_scores)
    return final_label, final_score
    

def main():
    parser = argparse.ArgumentParser(prog='Accent Detection',
                                     description='Provide a url to a video file on the internet and recieve an estimation of what accent the main speaker is using.')
    parser.add_argument('file',
                        help="path to a dowloaded video (or audio) file")
    parser.add_argument('--remove-file', action="store_true",
                        help="delete -file after evaluating it. For use in the main script that generates a temp input file.")
    parser.add_argument('-max-frames', default=54000, nargs="?",
                        help="to avoid memory errors only a certain number of frames can be analyzed at once. Default is 30 mins at 30FPS.")
    parser.add_argument('--large-file-handling', action="store_true",
                        help="Should your file be larger than `-max-frames`, but you still need to analyze the accent of the entire length set this flag.\n"
                             "This will cause the program to piecewise analyze each segment and output the most common accent.")
    
                        
    args = parser.parse_args()

    if args.file and os.path.isfile(args.file):
        print("EVALUATING INPUT....")
        
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            accent, confidence = get_accent_estimate_from_file(args.file, max_frames=args.max_frames, large_file_handling=args.large_file_handling)

        
        print(f"\nThe provided file is likely in an accent from {READABLE_LABEL_DICT[accent] if accent in READABLE_LABEL_DICT else accent }. The system is {confidence*100:.1f}% confident.")
        
        if args.remove_file:
            os.remove(args.file)
        

if __name__ == "__main__":
    main()