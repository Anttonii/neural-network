# Changelog

Notable changes and additions will be logged into this file.

## [0.0.4] - 2024-17-10

### Changes
 - The simple neural network model now batches input resulting in higher accuracy and faster training.
 - Updated README.md with new plots and table of recorded data over different parameters

## [0.0.3] - 2024-16-10

### Added

 - Convolutional neural network model built with Pytorch.
 - README.md now includes results for convolutional neural network.
 - `test-best` command also now works with convolutional neural network.

### Changes

 - `server.py` now has two modes for the simple neural network and for the convolutional neural network
   - Determining which neural network is used depends on the `method` json field that should be present in a request.
 - K-fold cross validation is now in util.py
 - K-fold now uses newer numpy random number generation method.

## [0.0.2] - 2024-14-10

### Added
 - K-fold cross validation
 - `predict_with_confidence` function that returns the prediction with percentual confidence.
   - Server now also returns the confidence with the prediction.

### Changes
 - Formatting with `autopep8`
 - Cleaned up parts of the code
 - Numpy random number generation now no longer uses legacy code
 - Cleaned up Typer commands
 - Updated README.md
 - Moved utility functions to `util.py`

## [0.0.1]

### Added
 - Initial project release