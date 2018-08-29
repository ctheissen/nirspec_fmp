# nirspec_fmp (NIRSPEC Forward Modeling Pipeline)
The nirspec_fmp is a forward-modeling pipeline for the NIRSPEC spectrometer, which is intended to make required adjustments before reducing private data using [NIRSPEC-Data-Reduction-Pipeline(NSDRP)](https://github.com/Keck-DataReductionPipelines/NIRSPEC-Data-Reduction-Pipeline), to perform telluric wavelength calibrations, and to forward model spectral data. The code is currently being developed.

Authors:
* Dino Chih-Chun Hsu (UCSD)
* Adam Burgasser, PI (UCSD)
* Chris Theissen (BU, UCSD)
* Jessica Birky (UCSD)

## Code Setup:
Dependencies:
* astropy
* numpy
* scipy
* matplotlib
* pandas

Download the nispec_fmp and the forked and modified version of the NSDRP to your computer.

Set up the environment variables in the `.bashrc` or `.bash_profile`

```
export PYTHONPATH="/path/to/nirspec_fmp:${PYTHONPATH}"
```

## Reducing the data using NSDRP:
To add required keywords to the headers before reducing private data using [NSDRP](https://github.com/Keck-DataReductionPipelines/NIRSPEC-Data-Reduction-Pipeline), use the addKeyword function and the [input](https://github.com/chihchunhsu/nirspec_fmp/blob/master/input_reduction.txt) text file:
```
>>> import nirspec_fmp
>>> nirspec_fmp.addKeyword(file='input_reduction.txt')
```
The example txt file is for setting up the data information such as file numbers of darks, flats, arcs, and sources. 

Note that you don't need to perform this task if you download the data directly from the Keck Observatory Archive (KOA).

To reduce the data, use the forked [NSDRP](https://github.com/chihchunhsu/NIRSPEC-Data-Reduction-Pipeline) on the command line:

```
$ python ~/path/to/NIRSPEC-Data-Reduction-Pipeline/nsdrp.py rawData/ reducedData/ -oh_filename ~/path/to/NIRSPEC-Data-Reduction-Pipeline/ir_ohlines.dat -spatial_jump_override -verbose -debug
```

, where the directory rawData/ is the path to the raw data is, and reducedData/ is the path where you want to store the reduce data.

## Dark Subtraction:
You can also optionally subtract the dark frames using subtractDark.py before running the NSDRP. This may be put into the NSDRP in the future.

## Defringe Flats:
The algorithm follows Rojo & Harrington (2006) to remove fringe patterns from flat files, but it does not smooth over the adjacent data points. This may cause a problem when observed spectra are close to the edge of the order. The example and sample outputs are upder the example folder.

## Wavelength Calibration using Telluric Standard Spectra:
The algorithm follows Blake at el. (2010) to cross-correlate the ESO atmospheric model and an observed telluric spectrum, fit the residuals, and iterate the process until the standard deviation of the residual reaches a mininum. The example and sample outputs are upder the example folder.

<!---*## Forward Modeling Science Spectra:---> 
