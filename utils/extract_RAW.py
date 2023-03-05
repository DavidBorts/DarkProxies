# Various helper scripts to extract RAW file metadata using RawPy

import rawpy

# Code adapted from: https://stackoverflow.com/a/67280050/20394190
def get_cfa(raw_file_path):
    raw = rawpy.imread(raw_file_path)

    cfa = "".join([chr(raw.color_desc[i]) for i in raw.raw_pattern.flatten()])
    return cfa