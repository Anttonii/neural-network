# Changelog

Notable changes and additions will be logged into this file.

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