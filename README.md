# Accent Recognition Tool

`Based off Ravanelli et. al @ Huggingface "[CommonAccent](https://huggingface.co/Jzuluaga/accent-id-commonaccent_xlsr-en-english)", through [Speechbrain](https://github.com/speechbrain/speechbrain)`

## ACCENTS
This tool recognises the English accents from the following places:
- us
- england
- australia
- indian
- canada
- bermuda
- scotland
- african
- ireland
- new zealand
- wales
- malaysia
- philippines
- singapore
- hongkong
- south atlandtic

## SETUP
The inference script uses Python, requiring version number between 3.10 and 3.12 (inclusive). The latest python version 3.13 is incompatible with necessary aspects of Speechbrain.

See and install requirements.txt for required python libraries. Notably, a specific older version of Speechbrain is required through this demo, if you have used Speechbrain for other projects make sure to run this installation in a virtual env or downgrade your Speechbrain to versiom `0.5.16`
```pip install requirements.txt```

One of 3 audio processing tools is needed to interface with Speechbrain:
 - FFMPEG: recommended option. installable through `sudo apt install ffmpeg` (Linux), `brew install ffmpeg` (Mac), `conda install conda-forge::ffmpeg` (Windows, exec install also available at [ffmpeg.org](https://ffmpeg.org/download.html) )
 - SoX (not supported on Windows)
 - SoundFile
 (use the latter two only if they are already installed in your machine)
 
You will also need spare diskspace to download the model parameters, this is done automatically the first time any of the scripts are run.

For use of `detect_accent_of_url_video`, curl is also required.

## COMMAND LINE USE
From URL. To detect the accent of a video on the internet use:
```detect_accent_of_url_video.bat https://ik.imagekit.io/kiu67jgp1/speaking_sample.mp4?tr=orig``` 
(where the link points to a sample video of myself uploaded to imagekit, replace with your own url as necessary)

If you have a local file you want to analse, you can use the main python script directly
```python main.py speaking_sample.mp4```





 
 