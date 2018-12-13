# Checks what devices are available.
# Quick check to see if GPU is available.
import tensorflow as tf
from tensorflow.python.client import device_lib

if __name__=='__main__':
    device_lib.list_local_devices()
