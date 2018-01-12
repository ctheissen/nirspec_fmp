# nirspec_pip
The nirspec_pip is to make the required changes before processing [NSDRP] (https://github.com/Keck-DataReductionPipelines/NIRSPEC-Data-Reduction-Pipeline). More features are developing and will be updated in the future.

Authors:
* Dino Chih-Chun Hsu (UCSD)
* Adam Burgasser, PI (UCSD)

Dependencies:
* astropy
* numpy
* scipy
* pandas
* matplotlib

# Code Setup:
Before using the NSDRP, one needs to modify the data headers with modifyHeader.py, setting up the parameters in the code. One can also subtract darks from flats with subtractDark.py. The setup is similar to modifyHeader.py.

The recipe is described below:
* Use modifyHeader.py to add the keywords required by the NSDRP.
* Use subtractDark.py to subtract darks from flats and return a combined flat.
* Run the NSDRP in the terminal.
