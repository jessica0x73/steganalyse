# An Investigation into Steganalysis Techniques in Media File Forensics and the Use of Machine Learning in Identifying Affected Files


## Description

Steganalyse.py is a Python tool that uses SVMs and Logistic Regression to classify image and video files as being either clean or steganographic.

The features that have been successfully utilised are:
- Lyu and Farid's (2003, quoted in Schaathun, 2012a, p. 96; 2006, quoted in Schaathun, 2012a, p. 96) higher order statistics for image steganalysis, through the `features` package of the `pysteg` (Schaathun, 2012b) Python module.
- Zhang, Cao and Zhao's (2017) features for video steganalysis based on "near perfect estimation for local optimality", through Zhang's (2019) `NPELO` extractor.


## Requirements

- Runs only on UNIX systems, due to dependence on `subprocess` module (tested on Ubuntu 18.04)
- Videos must be a must be a raw H.264 bitstream for steganalysis
- Python 3
- Python 2.7
- `wine` Python module ([more info](https://wiki.winehq.org/Ubuntu))
- `pysteg` Python module (Schaathun, 2012b)
- `NPELO` extractor (Zhang, 2019)
- `curl` [if using get-training-images.sh]
- `pwgen` [if using get-training-images.sh]
- `steghide` [if using get-training-images.sh]

See requirements-p3.txt for required Python 3 modules, and requirements-p2.txt for required Python 2.7 modules.


## Usage

### Main program: steganalyse.py

Note that train-classifiers.py (and get-training-images.sh, if an image dataset is required) needs to be run first, and the .joblib classifiers should be in the same directory as steganalyse.py.

```console
$ python3 ./steganalyse.py -h

usage: steganalyse.py [-h] [-f FILENAMES [FILENAMES ...]] [-t TEXT_FILE]

A program to detect image or video steganography

optional arguments:
  -h, --help            show this help message and exit
  -f FILENAMES [FILENAMES ...], --filenames FILENAMES [FILENAMES ...]
                        Name(s) of file(s) to analyse
  -t TEXT_FILE, --text-file TEXT_FILE
                        Get filenames from a list in a .txt file

```

### Creating & training classifiers: train-classifiers.py

Training data should be segmented into folders as follows:

* training-data/
  + images/
    - clean/
    - stego/
  + videos/
    - clean/
    - stego/

```console
$ python3 ./train-classifiers.py -h

usage: train-classifiers.py [-h] dir_location

A script to extract image & video features, & train machine learning
classifiers [SVM & Logistic Regression].

positional arguments:
  dir_location  Directory location of training data in quotation marks

optional arguments:
  -h, --help    show this help message and exit

```

### Gathering image files: get-training-images.sh

Input: .txt file list of image URLs.

```console
$ ./get-training-images.sh url-list.txt

```

Due to availability of existing video datasets and video steganography tools, the video files must be gathered and prepared manually.


## References

Schaathun, H. G. (2012a) Machine learning in image steganalysis, Chichester: Wiley.

Schaathun, H. G. (2012b) *pysteg* [Python library]. Available at: http://www.ifs.schaathun.net/pysteg/ (Accessed: 11 October 2018).

Zhang, H., Cao, Y. and Zhao, X. (2017) 'A steganalytic approach to detect motion vector modification using near-perfect estimation for local optimality', *IEEE Transactions on Information Forensics and Security*, 12(2), pp. 465-478. doi: [10.1109/TIFS.2016.2623587](https://doi.org/10.1109/TIFS.2016.2623587)

Zhang, H. (2019) 'Feature extractors for video steganalysis', *GitHub repository*, GitHub. Available at: https://github.com/zhanghong863/Feature-Extractors-for-Video-Steganalysis (Accessed: 1 February 2019).
