#!/bin/bash
# Usage: ./makevideo.sh <image_directory> <image_pattern> <duration_per_image> <output_video>
# Example: ./makevideo.sh progress "groepsfoto_resized_smaller_scaled_smaller_dithered_uniform_*.png" 0.04 output.mp4

if [ "$#" -ne 4 ]; then
    echo "Usage: $0 <image_directory> <image_pattern> <duration_per_image> <output_video>"
    exit 1
fi

# Assign input arguments to descriptive variables
image_dir="$1"
image_pattern="$2"
duration="$3"
output="$4"

# Remove any existing filelist
rm -f filelist.txt

# Generate filelist.txt by finding images that match the argument pattern,
# sorting them in natural order, and outputting lines with a duration for each image.
# Note: The find command uses "$image_pattern" directly.
for img in $(find "$image_dir" -maxdepth 1 -type f -name "$image_pattern" | sort -V); do
    echo "file '$img'" >> filelist.txt
    echo "duration $duration" >> filelist.txt
done

# For the concat demuxer, the last file should be listed without a duration.
# Remove the last duration line.
sed -i '$d' filelist.txt

# Run ffmpeg to create the video.
ffmpeg -f concat -safe 0 -i filelist.txt -c:v libx264 "$output"
