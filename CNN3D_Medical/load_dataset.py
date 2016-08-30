
import SimpleITK as sitk
import theano 
from theano import tensor as T 
import numpy as np 
import os 

def shared_dataset(trainCTPath, trainMaskPath, testCTPath, testMaskPath, filterSize, padSize, borrow=True):
	""" Function that loads the dataset into shared variables

	The reason we store our dataset in shared variables is to allow
	Theano to copy it into the GPU memory (when code is run on GPU).
	Since copying data into the GPU is slow, copying a minibatch everytime
	is needed (the default behaviour if the data is not in a shared
	variable) would lead to a large decrease in performance.
	"""
	
	fileTrainCTList = os.listdir(trainCTPath)
	#fileTestCTList = os.listdir(testCTPath)
	
	noTrain = int(fileTrainCTList.__len__()/2)
	#noTest = int(fileTestCTList.__len__()/2)

	for file in fileTrainCTList:
		if file.endswith(".mhd"):
			imgFilename = trainCTPath + file
			imgInput = sitk.ReadImage(imgFilename)
			img = sitk.GetArrayFromImage(imgInput)
			Width = img.shape[0]
			Height = img.shape[1]
			Depth = img.shape[2]
			break
			

	#no_Layers = 2;
	

	dtrain_x = np.zeros([noTrain, Width+2*padSize[0,0], Height+2*padSize[0,1], Depth+2*padSize[0,2]], 'float32')
	dtrain_y = np.zeros([noTrain, Width, Height, Depth], 'float32')

	for i in range(1, noTrain+1):
		imgFilename = trainCTPath + "liver-orig-resamp" + str(i).zfill(3) + ".mhd"
		mskFilename = trainMaskPath + "liver-seg-resamp" + str(i).zfill(3) + ".mhd"

		print "processing " + imgFilename + " ..."
		print (" ")

		imgInput = sitk.ReadImage(imgFilename)
		mskInput = sitk.ReadImage(mskFilename)
		
		#Extraction of image ROI
		aa = sitk.GetArrayFromImage(imgInput)
	#    Input size is increased to compensate for convlution size reduction
		dtrain_x[i-1, :, :, :] = np.array(np.pad(aa, ((padSize[0,0],), (padSize[0,1],), (padSize[0,2],)), 'symmetric'), dtype=np.float32)
	#   dtrain_x[i-1, :, :, :] = sitk.GetArrayFromImage(imgInput)
		dtrain_y[i-1, :, :, :] = sitk.GetArrayFromImage(mskInput)
		dtrain_y[i-1, :, :, :] = np.array(dtrain_y[i-1, :, :, :], dtype=np.float32)
		dtrain_y[i-1, :, :, :] = np.logical_and(dtrain_y[i-1, :, :, :], dtrain_y[i-1, :, :, :])

	shared_x = theano.shared(np.asarray(dtrain_x,dtype=theano.config.floatX),borrow=borrow)  
	shared_y = theano.shared(np.asarray(dtrain_y,dtype=theano.config.floatX),borrow=borrow) 
		
	# When storing data on the GPU it has to be stored as floats
	# therefore we will store the labels as ``floatX`` as well
	# (``shared_y`` does exactly that). But during our computations
	# we need them as ints (we use labels as index, and if they are
	# floats it doesn't make sense) therefore instead of returning
	# ``shared_y`` we will have to cast it to int. This little hack
	# lets ous get around this issue
	return shared_x, T.cast(shared_y, 'float32'), dtrain_y
	