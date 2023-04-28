# image-validator

Checks that subjects are standing upright with exposed limbs and facing the camera.

There are two files in this repository; dataset-parser.py utilises the functions found in analysis-functions.py. In total this repo is a general outline of how the functions should be used. Feel free to update as required for code clarity, improvements in function, etc.

We're currently sitting at a false positive generation rate of about 6%. A good bit of this can be addressed by tuning the constants in analysis-functions.py but thought should also go into improving analysis logic - which was hastily constructed. We should replace it over time.

In order to run this code you'll need to clone the repo, create a virutal enviroment in the repo root, activate it in a terminal instance, install the required packages into it by backing back out to the root and using `python -m pip -r requirements.txt`. 

When you want to run the program, you will need to do so inside a terminal instance with the venv activated or from inside an IDE (like Visual Studio Code) with the appropriate extensions to identify your venv and enter it for you. 
