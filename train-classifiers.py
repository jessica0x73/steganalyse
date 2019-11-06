import sys
import os
import argparse
import pathlib
import glob
import fleep
import magic
import subprocess
import re
import math
import pandas
import json
import csv
from sklearn.model_selection import train_test_split
from sklearn import svm, metrics, preprocessing, linear_model
import joblib


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


def create_lr_classifier(file_type):
    if file_type == 'image':
        csv_file = './img-features.csv'
        joblib_file = 'img-lr.joblib'
    else:
        csv_file = './vid-features.csv'
        joblib_file = 'vid-lr.joblib'

    print('=== Handling Logistic Regression for {} files ... ==='.format(file_type))

    print('[*] Reading {} ... '.format(csv_file))
    training_data = pandas.read_csv(csv_file)

    print('[*] Getting x and y ... ')
    x = training_data.drop(['file_name', 'class'], axis=1)
    y = training_data['class']

    print('[*] Scaling x ... ')
    scaler = preprocessing.MinMaxScaler(feature_range=(0, 1))
    x = scaler.fit_transform(x)

    print('[*] Splitting data ... ')
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)

    print('[*] Setting classifier ... ')
    classifier = linear_model.LogisticRegression()

    print('[*] Training classifier (note: this may take a while) ... ')
    classifier.fit(x_train, y_train)

    print('[*] Finding accuracy ... ')
    print('Accuracy: {}'.format(classifier.score(x_test, y_test)))

    print('[*] Saving classifier as {} ... \n'.format(joblib_file))
    joblib.dump(classifier, joblib_file)


def create_svm_classifier(file_type):
    if file_type == 'image':
        csv_file = './img-features.csv'
        joblib_file = 'img-svm.joblib'
    else:
        csv_file = './vid-features.csv'
        joblib_file = 'vid-svm.joblib'

    print('=== Handling SVM for {} files ... ==='.format(file_type))

    print('[*] Reading {} ... '.format(csv_file))
    training_data = pandas.read_csv(csv_file)

    print('[*] Getting x and y ... ')
    x = training_data.drop(['file_name', 'class'], axis=1)
    y = training_data['class']

    print('[*] Splitting data ... ')
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)

    print('[*] Setting classifier ... ')
    classifier = svm.SVC(kernel='linear')

    print('[*] Training classifier (note: this may take a while) ... ')
    classifier.fit(x_train, y_train)

    print('[*] Getting predictions ... ')
    y_pred = classifier.predict(x_test)

    print('[*] Finding accuracy ... ')
    print('... Accuracy: {}'.format(metrics.accuracy_score(y_test, y_pred)))

    print('[*] Saving classifier as {} ... \n'.format(joblib_file))
    joblib.dump(classifier, joblib_file)


# --------------------------------------------


# FUNCTION: WRITE IMAGE FEATURES TO CSV
def write_img_csv(stego_files_features, clean_files_features):
    # set file name
    output_file = 'img-features.csv'
    # set up lists for csv
    list_of_dicts = []
    feature_types = []
    # update class of stego files
    for file in stego_files_features:
        temp_dict = {}
        temp_dict['file_name'] = file.file_name
        for feature_type, feature_list in file.features.items():
            temp_dict[feature_type] = feature_list
            if feature_type not in feature_types:
                feature_types.append(feature_type)
        temp_dict['class'] = 1
        list_of_dicts.append(temp_dict)
    # update class of clean files
    for file in clean_files_features:
        temp_dict = {}
        temp_dict['file_name'] = file.file_name
        for feature_type, feature_list in file.features.items():
            temp_dict[feature_type] = feature_list
            if feature_type not in feature_types:
                feature_types.append(feature_type)
        temp_dict['class'] = 0
        list_of_dicts.append(temp_dict)
    # add features to csv for processing
    with open(output_file, 'w', newline='') as csv_file:
        # set fieldnames
        fieldnames = ['file_name']
        for feature_type in feature_types:
            fieldnames.append(feature_type)
        fieldnames.append('class')
        # set up writer
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        # begin writing
        writer.writeheader()
        for dict_item in list_of_dicts:
            writer.writerow(dict_item)
    # update user again
    print('[*] Extracted image features can be found in {}.'.format(output_file))


# FUNCTION: WRITE TO CSV
def write_vid_csv(stego_files_features, clean_files_features):
    # set file name
    output_file = 'vid-features.csv'
    # set up lists for csv
    list_of_dicts = []
    feature_types = []
    # update class of stego files
    for file in stego_files_features:
        for extracted_frame in file.features:
            temp_dict = {}
            temp_dict['file_name'] = extracted_frame
            for feature_type, feature_list in file.features[extracted_frame].items():
                temp_dict[feature_type] = feature_list
                if feature_type not in feature_types:
                    feature_types.append(feature_type)
            temp_dict['class'] = 1
            list_of_dicts.append(temp_dict)
    # update class of clean files
    for file in clean_files_features:
        for extracted_frame in file.features:
            temp_dict = {}
            temp_dict['file_name'] = extracted_frame
            for feature_type, feature_list in file.features[extracted_frame].items():
                temp_dict[feature_type] = feature_list
                if feature_type not in feature_types:
                    feature_types.append(feature_type)
            temp_dict['class'] = 0
            list_of_dicts.append(temp_dict)
    # add features to csv for processing
    with open(output_file, 'w', newline='') as csv_file:
        # set fieldnames
        fieldnames = ['file_name']
        for feature_type in feature_types:
            fieldnames.append(feature_type)
        fieldnames.append('class')
        # set up writer
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        # begin writing
        writer.writeheader()
        for dict_item in list_of_dicts:
            writer.writerow(dict_item)
    # update user again
    print('[*] Extracted video features can be found in {}.'.format(output_file))


# --------------------------------------------


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


# FUNCTION: GET FARID FEATURES
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
        farid_dict[feature_name] = float(feature_value)
        counter = counter + 1
    counter = 1
    for feature_value in farid_g:
        feature_name = 'farid_g_{}'.format(counter)
        farid_dict[feature_name] = float(feature_value)
        counter = counter + 1
    counter = 1
    for feature_value in farid_b:
        feature_name = 'farid_b_{}'.format(counter)
        farid_dict[feature_name] = float(feature_value)
        counter = counter + 1

    # add dict to file object
    file.features.update(farid_dict)

    # return file
    return file


# FUNCTION: PERFORM STEGANALYSIS
def perform_steganalysis(file_list, group_type):
    # update user on progress
    print('\n=== Performing feature extraction on {} files (this will take a while) ... ==='.format(group_type))
    # get features for each file
    file_number = 1
    for file in file_list:
        print('[*] {} of {} files'.format(file_number, len(file_list)))
        if file.file_type == 'image':
            file = get_farid_features(file)
        elif file.file_type == 'video':
            file = get_npelo_features(file)
        file_number = file_number + 1
    # update user again
    print('=== Steganalysis complete! ===')
    # return files
    return file_list


# --------------------------------------------


# FUNCTION: GET FILE TYPE OF INPUT FILE
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


# FUNCTION: GET LIST OF FILES
def get_file_lists(dir_location):
    file_names = glob.glob("{}/*".format(dir_location))
    file_list = []
    for file_name in file_names:
        if find_file(file_name):  # try to find file, if file can be found:
            new_file = File(file_name)  # create File object
            file_type, file_extension = get_file_type(file_name)  # get file type and file extension
            if file_type != 'other':
                file_size = os.path.getsize(file_name)  # get file size
                new_file.update_file(file_type, file_extension, file_size)  # update new_file with new info
                file_list.append(new_file)  # add file object to file list
    return file_list


# --------------------------------------------


# FUNCTION: FEATURE EXTRACTION
def extract_features(dir_location, file_type):
    # get file lists of File objects
    stego_files = get_file_lists("{}/stego".format(dir_location))
    clean_files = get_file_lists("{}/clean".format(dir_location))
    print('[*] Number of stego {} files: {}'.format(file_type, len(stego_files)))
    print('[*] Number of clean {} files: {}'.format(file_type, len(clean_files)))
    # get features for stego files
    stego_files_features = perform_steganalysis(stego_files, 'stego')
    # get features for clean files
    clean_files_features = perform_steganalysis(clean_files, 'clean')
    # return files
    return stego_files_features, clean_files_features


# FUNCTION: RUN PROGRAM
def run(dir_location):
    # get dir paths
    img_dir = "{}/images".format(dir_location)
    vid_dir = "{}/videos".format(dir_location)
    # extract features
    print('\n===== EXTRACTING IMAGE FEATURES =====\n')
    img_stego_features, img_clean_features = extract_features(img_dir, 'image')
    print('\n===== EXTRACTING VIDEO FEATURES =====\n')
    vid_stego_features, vid_clean_features = extract_features(vid_dir, 'video')
    # write to csvs
    print('\n===== WRITING FEATURES TO DISK =====\n')
    write_img_csv(img_stego_features, img_clean_features)
    write_vid_csv(vid_stego_features, vid_clean_features)
    # create & train svm
    print('\n===== CREATING & TRAINING SVMs =====\n')
    create_svm_classifier('image')
    create_svm_classifier('video')
    print('\n===== CREATING & TRAINING LOGISTIC REGRESSION CLASSIFIERS =====\n')
    create_lr_classifier('image')
    create_lr_classifier('video')


# MAIN FUNCTION: GLOBAL VARIABLES
if __name__ == '__main__':
    # argument parsing
    parser = argparse.ArgumentParser(description='A script to extract image & video features, '
                                                 '& train machine learning classifiers [SVM & Logistic Regression]. ')
    parser.add_argument('dir_location', action="store", help='Directory location of training data in quotation marks')
    args = parser.parse_args()
    # handle arguments
    print('Searching for directory ...')
    if args.dir_location:
        if os.path.exists(args.dir_location):
            print('Location found!')
            run(args.dir_location)
        else:
            print('Location not found!')
    else:
        parser.print_help(sys.stderr)
        sys.exit(1)
