#!/usr/local/bin/python
# encoding: utf-8

# TODO: add 'converters' to NP.loadtxt call in case class labels are strings

import sys
import os
import random
import Collections as CL
import itettools as IT
import numpy as NP


data_path = "/data"
dfile = os.path.join(data_path, "iris.csv")

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

# separate the feature vectors & class labvels:
data, class_labels = NP.hsplit(D, [-1])

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

data = NP.apply_along_axis(rescale_mlp, axis=0, arr=data)

# split data into training, test, & validation groups
# 80-10-10 fractions

nrows = data.shape[0]
TEST_FRACTION = .1
VALIDATE_FRACTION = .1

size_validate = NP.floor(VALIDATE_FRACTION * nrows)
size_test = NP.floor(TEST_FRACTION * nrows)
size_train = nrows - (size_validate + size_test)

train_data = data[:size_train,:]
validate_data = data[ size_train : size_train+size_validate,: ]
test_data = data[ size_train+size_validate:,:]


