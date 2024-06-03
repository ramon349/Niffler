# The Niffler PNG Extractor

The PNG Extractor converts a set of DICOM images into png images, extract metadata in a privacy-preserving manner.


## Configuring Niffler PNG Extractor

Find the config.json file in the folder and modify accordingly *for each* Niffler PNG extractions.

* *DICOMHome*: The folder where you have your DICOM files whose metadata and binary imaging data (png) must be extracted.

* *OutputDirectory*: The root folder where Niffler produces the output after running the PNG Extractor.


* *SaveBatchSize*: During extraction as metadata is being extracted it will be saved in batches of size N specified by this parameter

* *NumProcesses*: How many of the CPU cores to be used for the Image Extraction.

* *PublicHeadersOnly*:  Only extract public headers if set to true. Otherwise extract all the dicom tags 

  
* *ApplyVOILUT*: Apply windowing transform and/or manufacturer specific transforms before converting to png 


*  *SpecificHeadersOnly* : If you want only certain attributes in extracted csv, Then set this value to true and write the required attribute names in featureset.txt. Default value is _false_. Do not delete the featureset.txt even if you don't want this only specific headers


## Running the Niffler PNG Extractor
```bash

$ python3 ImageExtractor.py --ConfigPath path2config.json

# With Nohup
$ nohup python3 ImageExtractor.py --ConfigPath path2config.json > UNIQUE-OUTPUT-FILE-FOR-YOUR-EXTRACTION.out &

```
Check that the extraction is going smooth with no errors, by,

```
$ tail -f UNIQUE-OUTPUT-FILE-FOR-YOUR-EXTRACTION.out
```

## The output files and folders

In the OutputDirectory, there will be several sub folders and directories.

* *metadata.csv*: The metadata from the DICOM images in a csv format. Now also contains png path in the png_path column 

* *ImageExtractor.out*: The log file.

* *extracted-images*: The folder that consists of extracted PNG images

* *failed-dicom*: The folder that consists of the DICOM images that failed to produce the PNG images upon the execution of the Niffler PNG Extractor. Failed DICOM images are stored in 4 sub-folders named 1, 2, 3, and 4, categorizing according to their failure reason.


## Running the Niffler PNG Extractor with Slurm

This feature is deprecated for now 



## Troubleshooting

If you encounter your images being ending in the failed-dicom/3 folder (the folder signifying base exception), check the
ImageExtractor.out.

Check whether you still have conda installed and configured correctly (by running "conda"), if you observe the below error log:

"The following handlers are available to decode the pixel data however they are missing required dependencies: GDCM (req. GDCM)"

The above error indicates a missing gdcm, which usually happens if either it is not configured (if you did not follow the
installation steps correctly) or if conda (together with gdcm) was later broken (mostly due to a system upgrade or a manual removal of conda).

Check whether conda is available, by running "conda" in terminal. If it is missing, install [Anaconda](https://www.anaconda.com/distribution/#download-section).
 
If you just installed conda, make sure to close and open your terminal. Then, install gdcm.

```
$ conda install -c conda-forge -y gdcm 
```

If a ```MemoryError``` pops up while extracting metadata and mapping dataframes, mentioned here in the [issue - 307](https://github.com/Emory-HITI/Niffler/issues/307).

Reduce the no. of DICOM datapoints in the provided cohort/iteration or increase the no. of cohorts while extracting DICOMs through Cold Extraction.
