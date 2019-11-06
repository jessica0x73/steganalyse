import argparse
import sys
import warnings
import os.path
import fleep
import pathlib
import magic
import subprocess
import json
import re
import math
import pandas
import scipy.stats
from sklearn import svm  # note: this is needed for the imported classifiers
import joblib
from tabulate import tabulate


# --------------------------------------------

# CLASSES

# --------------------------------------------


class File:
    """
    Attributes:
        file_name: A string containing the file name
        file_type: A string containing the file type (image, video, other)
        file_extension: A string containing the file extension
        file_size: A float containing the size of the file in bytes
        features: A dict containing each type of feature from file for ML -> {feature_set_name: [features]}
        classification: A dict containing of structure { classifier : prediction, etc }
    """

    def __init__(self, file_name):
        self.file_name = file_name
        self.file_type = ''
        self.file_extension = ''
        self.file_size = ''
        self.features = {}
        self.classification = {}

    def set_file_type(self, file_type):
        self.file_type = file_type

    def set_file_extension(self, file_extension):
        self.file_extension = file_extension

    def set_file_size(self, file_size):
        self.file_size = file_size

    def add_features(self, feature_source, feature_list):
        self.features[feature_source] = feature_list

    def update_file(self, file_type, file_extension, file_size):
        self.file_type = file_type
        self.file_extension = file_extension
        self.file_size = file_size

    def set_classification(self, classifier, prediction):
        self.classification[classifier] = prediction


# --------------------------------------------

# FUNCTIONS

# --------------------------------------------


# FUNCTION: MACHINE LEARNING CLASSIFIER
def classify_using_ml(file):
    print('[*] File: {}'.format(file.file_name))
    if file.file_type == 'image':
        # create data frame of features from File object
        features = pandas.DataFrame(file.features, index=[file.file_name])
        # predict classification using svm
        svm_prediction = img_svm_classifier.predict(features)
        if svm_prediction == 0:
            file.set_classification('svm', 'clean')
        else:
            file.set_classification('svm', 'stego')
        # predict classification using lr
        lr_prediction = img_lr_classifier.predict(features)
        if lr_prediction == 0:
            file.set_classification('lr', 'clean')
        else:
            file.set_classification('lr', 'stego')
    elif file.file_type == 'video':
        # create data frame of features from File object
        features = pandas.DataFrame.from_dict(file.features, orient='index')
        # predict classification using svm
        svm_prediction = vid_svm_classifier.predict(features)
        if 1 in svm_prediction:
            file.set_classification('svm', 'stego')
        else:
            file.set_classification('svm', 'clean')
        # predict classification using svm
        lr_prediction = vid_lr_classifier.predict(features)
        if 1 in lr_prediction:
            file.set_classification('lr', 'stego')
        else:
            file.set_classification('lr', 'clean')
    return file


# FUNCTION: GET NPELO FEATURES
def get_npelo_features(file):
    # set up files for bash cmds
    input_file = file.file_name
    output_file = 'temp-features.csv'
    extractor = 'NPELO_extractor/extractor.exe'
    bash_cmd = 'wine {} -s -t 12 -i {} -o {}'.format(extractor, input_file, output_file)

    if os.path.exists(output_file):
        os.remove(output_file)
    print('... Calling subprocess ')
    video_extraction_process = subprocess.Popen([bash_cmd], stdin=subprocess.PIPE, stdout=subprocess.PIPE, shell=True)
    output, error = video_extraction_process.communicate()
    decoded_output = output.decode('utf-8')

    print('... Handling frames')
    # get number of frames decoded & number of expected csv lines
    frames = 0
    for line in decoded_output.splitlines():
        if 'frames are decoded' in line:
            # print(line)
            frames = int(re.search(r'\d+', line).group())
    expected_lines = math.ceil(frames / 12)

    # set up column names for pandas
    col_names = []
    for i in range(36):
        col_i = 'NPELO_{}'.format(i + 1)
        col_names.append(col_i)

    # set up row names for pandas
    row_names = {}
    for i in range(expected_lines):
        row_i = '{}_f{}'.format(input_file, i + 1)
        row_names[i] = row_i

    # get data from csv
    temp_csv = pandas.read_csv(output_file, sep=' ', names=col_names, index_col=False)
    temp_csv.rename(index=row_names, inplace=True)

    print('... Handling features')
    features_dict = {}
    for row_name in row_names.values():
        features_dict[row_name] = {}
        for col_name in col_names:
            features_dict[row_name][col_name] = temp_csv.loc[row_name, col_name]

    # remove temp-features.csv
    if os.path.exists(output_file):
        os.remove(output_file)

    # add features to file object
    file.features.update(features_dict)

    return file


# FUNCTION: GET FARID FEATURES (36 PER COLOUR CHANNEL)
def get_farid_features(file):

    # create subprocess to handle features - pysteg is a python 2 package/collection and so needs to be run in p2
    p2_process = subprocess.Popen(['./p2-img-feature-extraction.py', 'farid', file.file_name], stdout=subprocess.PIPE)
    stdout, _ = p2_process.communicate()

    # the first 254 chars of the output are not needed
    decoded_stdout = stdout[254:].decode('utf-8')

    # split output into strings per channel
    segmented = decoded_stdout.split('\n')

    # json.loads converts string representation of list into actual list object
    farid_r = json.loads(segmented[0])
    farid_g = json.loads(segmented[1])
    farid_b = json.loads(segmented[2])

    # start dict
    farid_dict = {}

    # populate dict
    counter = 1
    for feature_value in farid_r:
        feature_name = 'farid_r_{}'.format(counter)
        farid_dict[feature_name] = feature_value
        counter = counter + 1
    counter = 1
    for feature_value in farid_g:
        feature_name = 'farid_g_{}'.format(counter)
        farid_dict[feature_name] = feature_value
        counter = counter + 1
    counter = 1
    for feature_value in farid_b:
        feature_name = 'farid_b_{}'.format(counter)
        farid_dict[feature_name] = feature_value
        counter = counter + 1

    # add dict to file object
    file.features.update(farid_dict)

    return file


# FUNCTION: PERFORM STEGANALYSIS
def perform_steganalysis(file_list):
    print('\n=== Performing steganalysis ===\n')

    # get features for each file
    print('Extracting features (this may take a while) ... ')
    file_number = 1
    for file in file_list:
        print('[*] File {} of {}: {} ({})'.format(file_number, len(file_list), file.file_name, file.file_type))
        if file.file_type == 'image':
            file = get_farid_features(file)
        elif file.file_type == 'video':
            file = get_npelo_features(file)
        file_number = file_number + 1
    print('Feature extraction complete!\n')

    # classify each file
    print('Classifying files ...')
    for file in file_list:
        file = classify_using_ml(file)
    classifications = {}
    for file in file_list:
        classifications[file.file_name] = file.classification
    print('Classifications complete!')

    # save classifications to file
    cols = ['File name', 'SVM Classification', 'LR Classification']
    classifications_df = pandas.DataFrame.from_dict(classifications, orient='index')
    classifications_df = classifications_df.reset_index()
    classifications_df.index += 1
    classifications_df.columns = cols
    classifications_df.to_csv(output_file)

    # output table to stdout
    print('\n=== Classifications ===\n')
    print(tabulate(classifications_df, headers=cols, tablefmt='psql'))
    print('\nClassification information also saved to {}\n'.format(output_file))


# FUCNTION: GET FILE TYPE OF INPUT FILE
def get_file_type(file_name):
    with open(file_name, 'rb') as file:
        file_info = fleep.get(file.read(128))
    if file_info.type_matches('raster-image') or file_info .type_matches('raw-image'):
        file_type = 'image'
        file_extension = file_info.extension[0]
    elif file_info.type_matches('video'):
        file_type = 'video'
        file_extension = file_info.extension[0]
    else:
        h264_flag = 'H.264'
        magic_info = magic.from_file(file_name)
        if h264_flag in magic_info:
            file_type = 'video'
            file_extension = 'h264'
        else:
            file_type = 'other'
            file_extension = pathlib.Path(file_name).suffix  # get file extension from pathlib instead
    return file_type, file_extension


# FUNCTION: FIND INPUT FILE IN FILESYSTEM
def find_file(file_name):
    if os.path.isfile(file_name):
        return True
    else:
        return False


# FUNCTION: RUN FUNCTION FOR MAIN
def run(filenames):
    print('\n === RUNNING PROGRAM ===\n')

    # start file list & counters
    input_file_list = []
    input_file_count = 0
    valid_file_count = 0

    # fill file list
    for file_name in filenames:
        input_file_count = input_file_count + 1
        if find_file(file_name):  # try to find file, if file can be found:
            new_file = File(file_name)  # create File object
            file_type, file_extension = get_file_type(file_name)  # get file type and file extension
            if file_type != 'other':
                valid_file_count = valid_file_count + 1
                file_size = os.path.getsize(file_name)  # get file size
                new_file.update_file(file_type, file_extension, file_size)  # update new_file with new info
                input_file_list.append(new_file)  # add file object to file list

    # output (for testing)
    print('[*] {} input files\n[*] {} valid images/videos in input files'.format(input_file_count, valid_file_count))

    # perform actual steganalysis
    perform_steganalysis(input_file_list)


# MAIN FUNCTION: GLOBAL CODE
if __name__ == '__main__':
    warnings.simplefilter('ignore', UserWarning)  # ignore UserWarnings - this is for farid features

    # argument parsing
    parser = argparse.ArgumentParser(description='A program to detect image or video steganography')
    parser.add_argument('-f', '--filenames', action="store", nargs='+', help='Name(s) of file(s) to analyse')
    parser.add_argument('-t', '--text-file', action='store', help='Get filenames from a list in a .txt file')
    args = parser.parse_args()

    # check for classifiers
    img_svm_joblib = 'img-svm.joblib'
    vid_svm_joblib = 'vid-svm.joblib'
    img_lr_joblib = 'img-lr.joblib'
    vid_lr_joblib = 'vid-lr.joblib'
    if not (find_file(img_svm_joblib) and find_file(vid_svm_joblib)
            and find_file(img_lr_joblib) and find_file(vid_lr_joblib)):
        print('Classifiers not found!')
        sys.exit(1)

    # load in classifiers
    img_svm_classifier = joblib.load(img_svm_joblib)
    vid_svm_classifier = joblib.load(vid_svm_joblib)
    img_lr_classifier = joblib.load(img_lr_joblib)
    vid_lr_classifier = joblib.load(vid_lr_joblib)
    print('[*] Classifiers successfully loaded')

    # set up file array
    input_files = []

    # set up output file
    output_file = 'classifications.csv'

    # handle arguments
    if args.filenames:
        input_files = args.filenames
    elif args.text_file:
        if os.path.isfile(args.text_file):
            text_file = open(args.text_file, 'r')
            for file_name in text_file.readlines():
                input_files.append(file_name.rstrip())
    else:
        parser.print_help(sys.stderr)
        sys.exit(1)

    # run main program
    run(input_files)
