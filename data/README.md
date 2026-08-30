# CICIDS2017 data

Place CICIDS2017 CSV files under `data/raw/`, or pass one or more CSV paths to the preparation command. Raw and processed data are excluded from Git.

Use either the eight original daily CSV files or one merged CSV, not both in the same run. The loader strips whitespace from headers and labels, normalizes known CICIDS2017 label spelling variants, converts feature columns to numeric values, and records rejected and duplicate rows. If one exact feature vector has multiple labels, every row with that vector is excluded and reported; the pipeline does not choose a preferred label.

The project does not download or redistribute CICIDS2017. Record the source URL and SHA-256 checksums for files used in a reported experiment.
