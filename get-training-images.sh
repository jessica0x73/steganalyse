#!/bin/bash
# Bash script to download & format steganalysis training images from list of URLs
# Jessica Sammon
# 05/04/2019

# check arguments
if [ -z "$*" ]; then
  echo "Error! Missing argument (URL list). Check input and try again."
  exit 1
elif [ ! -f $1 ]; then
  echo "Error! URL list not found. Check input and try again."
  exit 1
fi

# check for folders
if [ ! -d "training-data" ]; then
  mkdir "training-data"
  mkdir "training-data/clean"
  mkdir "training-data/stego"
fi
if [ ! -d "training-data/clean" ]; then
    mkdir "training-data/clean"
fi
if [ ! -d "training-data/stego" ]; then
    mkdir "training-data/stego"
fi

# set up url lists urls
url_list="$1"
stego_url_list="stego_url_list.txt"
stego_files_list="stego_files_list.txt"
clean_url_list="clean_url_list.txt"
clean_files_list="clean_files_list.txt"

# shuffle given url list and sort out into clean/stego
files_to_get=1500
echo "Shuffling stego URLs ... "
shuf -n $files_to_get $url_list | awk '{print $2}' > $stego_url_list
echo "Shuffling clean URLs ... "
shuf -n $files_to_get $url_list | awk '{print $2}' > $clean_url_list
echo "Shuffling complete!"

echo -e "\n--------------------------------------------------"
echo -e "--------------------------------------------------\n"

# get images from urls for stego images
file_number=1
echo "Getting images from stego URLs ... "
while read line; do
  echo "$file_number / $files_to_get"
  curl -s -m 30 "${line}" > training-data/stego/${line##*/}
  echo "${line##*/}" >> $stego_files_list
  file_number=$(( $file_number + 1 ))
done < $stego_url_list
echo "Stego images complete!"

echo -e "\n--------------------------------------------------"
echo -e "--------------------------------------------------\n"

# get images from urls for clean images
file_number=1
echo "Getting images from clean URLs ... "
while read line; do
  echo "$file_number / $files_to_get"
  curl -s -m 30 "${line}" > training-data/clean/${line##*/}
  echo "${line##*/}" >> $clean_files_list
  file_number=$(( $file_number + 1 ))
done < $clean_url_list
echo "Clean images complete!"

echo -e "\n--------------------------------------------------"
echo -e "--------------------------------------------------\n"

echo "All images gathered!"

echo -e "\n--------------------------------------------------"
echo -e "--------------------------------------------------\n"

# remove empty and non-jpeg images
echo "Removing empty and non-jpeg files ... "
counter=0
for filename in training-data/stego/*
do
  type="$(file -b $filename)"
  if [[ $type != JPEG* ]]; then
    rm $filename
  fi
done
counter=0
for filename in training-data/clean/*
do
  type="$(file -b $filename)"
  if [[ $type != JPEG* ]]; then
    rm $filename
  fi
done
echo "Empty and non-jpeg files removed!"

echo -e "\n--------------------------------------------------"
echo -e "--------------------------------------------------\n"

# display final numbers of gathered files
stego_file_count="$(ls -l training-data/stego/ | grep ^- | wc -l)"
clean_file_count="$(ls -l training-data/clean/ | grep ^- | wc -l)"
echo "Total images gathered for stego files: $stego_file_count"
echo "Total images gathered for clean files: $clean_file_count"

echo -e "\n--------------------------------------------------"
echo -e "--------------------------------------------------\n"

# start steganography process
echo "Beginning steganography on stego files ... "

# check for folders
if [ ! -d "secrets" ]; then
  mkdir "secrets"
fi

# create passwords/passphrases for steghide
echo "Creating passwords ... "
pwd_file="secrets/passwords.txt"
for i in {1..25}; do
  pwd_length="$(shuf -i 8-36 -n 1)"
  pwgen -s $pwd_length >> $pwd_file
done

# create secret files for embedding for steghide
echo "Creating secret files ... "
for i in {1..50}; do
  file_name="secrets/secret_file_$i.txt"
  file_length="$(shuf -i 25-200 -n 1)"
  pwgen -s $file_length > $file_name
done

# perform steganography using steghide - remove files if too small for steganalysis
echo "Performing steganography ... "
csv_file="secrets/pwd-info.csv"
echo -e "filename\tsecret\tpassword" > $csv_file
for filename in training-data/stego/*; do
  echo $filename
  secret_file_number="$(shuf -i 1-50 -n 1)"
  secret_file="secrets/secret_file_$secret_file_number.txt"
  password="$(shuf -n 1 $pwd_file)"
  if steghide embed -q -cf $filename -ef $secret_file -p $password; then
    echo "Success"
    echo -e "$filename\t$secret_file\t$password" >> $csv_file
  else
    echo -e "Fail. \nRemoving file $filename"
    rm $filename
  fi
done
