#create separate dir for this project, have own venv
#python -m venv .venv    
#.\.venv\Scripts\activate

#install the requirement dependencies and faster whisper base model
#python -m pip install -r requirements.txt
#download faster whisper model in local space - command below:
#hf download Systran/faster-whisper-base --local-dir models/faster-whisper-base  

#wav writing expects common PCM format (pulse code modulation - raw audio):
# uint8, int16 (standard), int32, float32
#each audio sample is represented by int16 number

#whisper model weights are stored (loaded) in quantised form, to consume min possible space
#whisper uses compute_type="int8", these 2 are different things.

#To make each dir (folder) a package, add an empty init file in them
# Make sure dictate/ has an empty __init__.py so it’s treated as a package.

#for local testing, run like this: python -m dictate.dictate
#from the root dir

#Load the heavy models once on startup and keep reusing them.
#To achieve this, encapsulate the code in a class and create an instance variable to store the heavy model

#Original sound captured by mic has 2D shape, each sample/row considers all channels
#Sample is stored as int16 number, -32768 to 32767 range
#Transformer encoder accepts only normalized tensor, to make computation faster