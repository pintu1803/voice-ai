#create separate dir for this project, have own venv
#python -m venv .venv    
#.\.venv\Scripts\activate

#install the requirement dependencies and faster whisper base model
#python -m pip install -r requirements.txt
#hf download Systran/faster-whisper-base --local-dir models/faster-whisper-base  

#wav writing expects common PCM format (pulse code modulation - raw audio):
# uint8, int16 (standard), int32, float32

#whisper uses compute_type="int8", these 2 are different things.

# Make sure dictate/ has an empty __init__.py so it’s treated as a package.

#for local testing, run like this: python -m dictate.dictate
#from the root dir

