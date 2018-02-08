# nirspec_pip
The nirspec_pip is to make the required changes before processing [NSDRP](https://github.com/Keck-DataReductionPipelines/NIRSPEC-Data-Reduction-Pipeline). More features are developing and will be updated in the future.

Authors:
* Dino Chih-Chun Hsu (UCSD)
* Adam Burgasser, PI (UCSD)

Dependencies:
* astropy
* numpy
* scipy
* pandas
* matplotlib

## Code Setup:
Before using the NSDRP, one needs to modify the data headers with addKeyword in addKeyword.py, using the example txt file to set up the information such as file numbers of darks, flats, and sources. 

One can also subtract darks from flats with subtractDark.py.

The recipe is described below:
* If you are reducing your own data (not available on the Keck Observatory Archive (KOA)), use addKeyword.py to add the keywords required by the NSDRP.
* Use subtractDark.py to subtract darks from flats and return a combined flat.
* Run the [NSDRP](https://github.com/chihchunhsu/NIRSPEC-Data-Reduction-Pipeline) in the terminal.
