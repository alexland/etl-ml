#!/usr/local/bin/python
# encoding: utf-8

import sys
import os
import random
import collections as CL
import itettools as IT
import numpy as NP


# rescale the data so that each feature (column) has a min/max of -1 to 1
def rescale_mlp(a):
	"""
		returns: numpy 1D vector rescaled so min/max => [-1, 1],
		pass in: numpy 1D array (column or feature fector)
	"""
	a -= a.min()
	a /= .5*a.max()
	a -= 1
	return a

def etl_mlp(dfile):
    with open(dfile, 'r') as f:
    	r1 = f.readline().strip().split(',')
    try:
    	int(r1[-1])
    	# proceed w/ simple case--class labels are integers
    	D = NP.loadtxt(dfile, delimiter=",", skiprows=0, comments='#')
    except ValueError:
    	# class labels are strings
    	file_obj = open(dfile, 'r')
    	data = [ row.strip().split(',') for row in file_obj.readlines() ]
    	class_labels = {row[-1] for row in data[1:]}
    	tx = [ label for label in enumerate(class_labels)]
    	LuT = dict([ (v, k) for k, v in tx ])
    	fnx = lambda k : LuT[k]
    	D = NP.loadtxt( dfile, delimiter=",", skiprows=0, comments='#', 
    		converters={-1:fnx} )
	NP.random.shuffle(D)
	data, class_labels = NP.hsplit(D, [-1])
	data = NP.apply_along_axis(rescale_mlp, axis=0, arr=data)
	nrows = data.shape[0]
	TEST_FRACTION = .1
	ALIDATE_FRACTION = .1
	offset1 = NP.floor(TEST_FRACTION * nrows)
	offset2 = NP.floor(VALIDATE_FRACTION * nrows)
	test_data = data[:offset1,:]
	test_data_labels = class_labels[:offset1]
	validate_data_labels = class_labels[offset1:offset1+offset2]
	validate_data = data[offset1:offset1+offset2,:]
	train_data = data[offset1+offset2:,:]
	train_data_labels = class_labels[offset1+offset2:]
	return CL.namedtuple('mlp_data', [ 'train_data', 'train_data"labels', 
								'test_data', 'test_data_labels',
								'validate_data', 'validate_data_labels'], 
								verbose=True)


