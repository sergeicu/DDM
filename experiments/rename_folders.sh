#!/bin/bash

# Directory to scan, replace "/path/to/directory" with your actual directory path
directory="."

# Loop through each subdirectory in the given directory
find "$directory" -mindepth 1 -maxdepth 1 -type d | while read folder; do
  
  # Calculate the size of the folder in kilobytes (kB)
  folder_size=$(du -s "$folder" | cut -f1)
  
  # If the folder size is less than 1024kB (1MB), print the folder name and its size in MB
  if [ "$folder_size" -lt 1024 ]; then

    # Skip folders already starting with "REMOVE_"
    if [[ "$(basename "$folder")" == REMOVE_* ]]; then
      echo "Folder already renamed $folder" 


      continue
    fi

    echo "Will rename: $folder (Size: $(bc <<< "scale=2; $folder_size/1024")MB)"
    # uncomment the below to run 
    #     mv "$folder" "${folder%/*}/REMOVE_${folder##*/}"

  else
    echo "$folder  -> $(bc <<< "scale=2; $folder_size/1024") MB"

  fi
done
