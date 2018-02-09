# nirspec_pip
The nirspec_pip is to make required adjustments before reducing private data using [NIRSPEC-Data-Reduction-Pipeline(NSDRP)](https://github.com/Keck-DataReductionPipelines/NIRSPEC-Data-Reduction-Pipeline), and to forward model the NIRSPEC spectral data.

Authors:
* Dino Chih-Chun Hsu (UCSD)
* Adam Burgasser, PI (UCSD)

## Code Setup:
Dependencies:
* astropy
* numpy
* matplotlib
<!---* scipy---> 
<!---* pandas---> 

Download the nispec_pip and the forked and modified NSDRP to your computer.

Set up the environment variables in the `.bashrc` or `.bash_profile`

```
export PYTHONPATH="/Users/dinohsu/projects/nirspec_pip:${PYTHONPATH}"
```

<!---*## Downloading the data from the Keck Observatory Archive (KOA)---> 

## Reducing the data using NSDRP:
To add required keywords to the headers before reducing private data using [NSDRP](https://github.com/Keck-DataReductionPipelines/NIRSPEC-Data-Reduction-Pipeline), use the addKeyword function and the [input](https://github.com/chihchunhsu/nirspec_pip/blob/master/input_reduction.txt) text file:
```
import nirspec_pip
nirspec_pip.addKeyword(file='input_reduction.txt')
```
The example txt file is for setting up the data information such as file numbers of darks, flats, arcs, and sources. 

Note that you don't need to perform this task if you download the data directly from the KOA.

To reduce the data, use the forked [NSDRP](https://github.com/chihchunhsu/NIRSPEC-Data-Reduction-Pipeline) on the command line:

```
$ python ~/path/to/NIRSPEC-Data-Reduction-Pipeline/nsdrp.py rawData/ reducedData/ -oh_filename ~/path/to/NIRSPEC-Data-Reduction-Pipeline/ir_ohlines.dat -spatial_jump_override -verbose -debug
```

, where the directory rawData/ is the path to the raw data is, and reducedData/ is the path where you want to store the reduce data.

You can also optionally subtract the dark frames using subtractDark.py before running the NSDRP.

<!---*## Forward Modeling the Spectra:---> 
