import collections
import math
import os
import random
import zipfile
import numpy as np
import tensorflow as tf

url = 'http://mattmahoney.net/dc/'

def maybe_download(filename, expected_bytes):
    if not os.path.exists(filename):
        filename, _ = urllib.request.urlretrieve(url + )