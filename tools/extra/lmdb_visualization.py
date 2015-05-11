#!/usr/bin/env python
# This script is used to visualizing lmdb generated from caffe. The value 
# of the lmdb is string form of Datum.
# Here I assume the Datum storing a label and a gray image. And assuming 
# key has mnist feature, a 8-int stringi
# Todo: support RGB/RGBA image
# Todo: support multithreading when drawing images.
# Todo: support other type of keys
# Todo: support commandline arguments
import lmdb
from caffe.proto import caffe_pb2 as cp
import matplotlib.pyplot as plt
import numpy as np
import ipdb

#ipdb.set_trace()
mdb_path = "/home/pris/development/caffe/examples/mnist/mnist_train_lmdb"
env = lmdb.open(mdb_path, max_dbs=0)
txn = env.begin()
cur = txn.cursor()

def show(da):
    im = np.array([ord(i) for i in da.data])
    im = im.reshape([da.channels, da.width, da.height])
    plt.figure()
    plt.imshow(im[0,:,:],cmap='gray')
    plt.title("label: %d" % da.label)
    plt.show()



while True:
    print "<<< Input a key: (or type 'quit')"
    key = raw_input()
    if key == "quit":
        break
    if not cur.set_key(key.zfill(8)):
        print "Error: key %s not fount!" %key
        continue
    print cur.key()
    datum = cp.Datum.FromString(cur.value())
    show(datum)
    #ipdb.set_trace()
#cur.close() # There are no close attribute in Cursor object?
#txn.close()
env.close()
