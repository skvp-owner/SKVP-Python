# SKVP-Python
A Python package for working with SKVP videos and Objects

## Prerequisite

* Python 3
* Numpy
* Scipy

It is recommended to use Anaconda, so Numpy and Scipy are automatically available

## Installation

Currently there is no automatic intallation script for SKVP-Python. In order to make your project work with SKVP-Python, please do one of the following **options**:

1. Download this repository and add it to the PYTHONPATH environment variable, so that Python can import files from it

2. Download this repository and add it to the `sys.path` array in your Python application, so that Python can import files from it

3. Copy the file `skvp.py` into your central Python lib directory


## Usage

### Usage Example
```
import skvp

# Load an SKVP formatted video from a given path. The file can be compressed with '.gz'
vid = skvp.load('/path/to/skvp_vid.skvp.gz')

# Temporally median filter the joint locations, into a new video
filtered = skvp.median(vid)

# Temporally scale the video into a new video of 100 frames
length_scaled = skvp.create_length_scaled_video(vid, num_frames = 100)

# Save the manipulated video, uncompressed
skvp.dump(length_scaled, '/path/to/new_video.skvp')

# Save the manipulated video, compressed
skvp.dump(length_scaled, '/path/to/new_video.skvp.gz')
```

### API

| Function | Description |
|:---------:|-------------|
| skvp.dump(skvp_video, ostream_or_filename) | Save an SKVP video into a file, given a path or an open stream |
| skvp.dump(skvp_video) | Save an SKVP video file content into a string |
| skvp.load(istream_or_filename) | Load an SKVP video from a path or an open stream |
| skvp.loads(video_string) | Load an SKVP video from a string |

