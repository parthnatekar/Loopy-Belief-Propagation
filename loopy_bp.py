import numpy as np
import pandas as pd
from scipy.stats import gamma
import matplotlib.pyplot as plt
import cv2
import argparse
from matplotlib.ticker import StrMethodFormatter

parser = argparse.ArgumentParser()

parser.add_argument('-p', action='store', required = True, dest='image_path',
                    help='Path to Image (can be either csv or png/jpg')

parser.add_argument('-m', action='store', required = False, dest='mode',
                    help='Mode: Sequential or Synchronous', default = 'sync')

parser.add_argument('-n', action='store', required = False, dest='add_noise',
                    help='Whether to add noise to image (do not use if image is already noisy, useful for testing)', default = False)

results = parser.parse_args()	

class LBP:

	def __init__(self, impath, iterations, params, add_noise = False):
		'''
		impath: Path to image, may be png or csv
		iterations: Number of iterations
		params: Theta and Gamma values
		add_noise: Whether to add external noise to input image. Useful for testing.
		'''
		self.impath = impath
		self.iterations = iterations
		self.params = params
		self.add_noise = add_noise
		self.compatibility_inter = np.array([[1.0, self.params['theta']], [self.params['theta'], 1.0]])
		self.compatibility_outer = np.array([[1.0, self.params['gamma']], [self.params['gamma'], 1.0]])
		self.preprocess()

	def preprocess(self):

		if self.impath[-3:] == 'png' or self.impath[-3:] == 'jpg':
			image = cv2.imread(self.impath, cv2.IMREAD_GRAYSCALE)/255
			image = cv2.threshold(image, 0.5, 1, cv2.THRESH_BINARY)[1]

		elif self.impath[-3:] == 'csv':
			im_csv = np.loadtxt(self.impath, delimiter=',')
			image = np.zeros((int(np.max(im_csv[:,0]))+1, int(np.max(im_csv[:,1]))+1))
			for j in range(len(im_csv)):
				image[int(im_csv[j,0]), int(im_csv[j,1])] = im_csv[j,2]

		self.noiseless_image = image.copy()

		if self.add_noise:

			flip_prob = 0.3
			for j in range(image.shape[0]*image.shape[1]):
				idx = np.unravel_index(j, (image.shape[0], image.shape[1]))
				thresh = np.random.random_sample()
				if thresh < flip_prob:
					image[idx] = 1-image[idx]

		self.image = image
		self.height, self.width = self.image.shape

	# Function to get neighbours. Only used for sequential message passing
	def get_neighbours(self, idx):

		neighbours = []
		if idx+self.size < self.size**2:
			neighbours.append(idx+self.size)
		if idx-self.size > 0:
			neighbours.append(idx-self.size)
		try:
			if np.unravel_index(idx-1, (self.size,self.size))[0] ==  np.unravel_index(idx, (self.size,self.size))[0]:
				neighbours.append(idx-1)
		except:
			pass
		try:
			if np.unravel_index(idx+1, (self.size,self.size))[0] ==  np.unravel_index(idx, (self.size,self.size))[0]:
				neighbours.append(idx+1)
		except:
			pass
		return(neighbours)

	# Function to perform sequential message passing
	def messages_seq(self):

		factor_messages = np.ones((self.size**2, self.size**2, 2))
		clique_messages = np.ones((self.size**2, self.size**2, 2))
		beta = np.ones((self.size, self.size, 2))

		for i in range(self.iterations):
			for j in range(self.size**2):
				neighbours = self.get_neighbours(j)
				for n in neighbours:
					factor_messages[j,n,:] = self.compatibility_outer[:, self.grid[np.unravel_index(j, (self.size, self.size))]]
					adj_neighbours = np.setdiff1d(neighbours,n)
					for adj in adj_neighbours:
						factor_messages[j,n,:] *= clique_messages[adj,j,:]
					clique_messages[n,j,:] *= self.compatibility_inter.dot(factor_messages[n,j,:])
			factor_norm = np.sum(factor_messages, axis=2)
			factor_messages[:,:,0] /= factor_norm
			factor_messages[:,:,1] /= factor_norm
			clique_norm = np.sum(clique_messages, axis=2)
			clique_messages[:,:,0] /= clique_norm
			clique_messages[:,:,1] /= clique_norm

		for j in range(self.size**2):
			neighbours = self.get_neighbours(j)
			for n in neighbours:
				beta[np.unravel_index(j, (self.size, self.size))] *= factor_messages[j,n]

		if self.noiseless_image:

			accuracy = np.sum(self.noiseless_image == np.argmax(beta, axis = 2))/self.size**2
			
			return(np.argmax(beta, axis = 2), accuracy)

		else:
			return(np.argmax(beta, axis = 2))

	# Function to perform synchronous message passing
	def messages_sync(self):	

		clique_messages = np.ones((self.height, self.width, 2, 4))
		factor_messages = np.ones((self.height, self.width, 2, 4))
		factor_norm = np.ones((self.height, self.width, 1, 4))
		clique_norm = np.ones((self.height, self.width, 1, 4))
		beta = np.ones((self.height, self.width, 2))

		for j in range(self.height*self.width):
			beta[np.unravel_index(j, (self.height, self.width))] = self.compatibility_outer[:, int(self.image[np.unravel_index(j, 
				(self.height, self.width))])]

		for i in range(self.iterations):

			factor_messages[:,:,:,0] *= beta*clique_messages[:,:,:,0]*clique_messages[:,:,:,2]*clique_messages[:,:,:,3]
			factor_messages[:,:,:,1] *= beta*clique_messages[:,:,:,1]*clique_messages[:,:,:,2]*clique_messages[:,:,:,3]
			factor_messages[:,:,:,2] *= beta*clique_messages[:,:,:,2]*clique_messages[:,:,:,0]*clique_messages[:,:,:,1]
			factor_messages[:,:,:,3] *= beta*clique_messages[:,:,:,3]*clique_messages[:,:,:,0]*clique_messages[:,:,:,1]


			for j in range(clique_messages.shape[-1]):
				factor_norm[:,:,:,j] = np.sum(factor_messages[:,:,:,j], axis=2)[..., None]

			for j in range(clique_messages.shape[-1]):
				for ii in range(2):
					factor_messages[:, :, ii, j] /= np.squeeze(factor_norm[:,:,:,j])

			clique_messages[:,:,:,0] *= self.compatibility_inter.dot(np.append(np.ones((self.height, 1, 2)), 
										factor_messages[:,:-1,:, 0], axis=1).transpose((0,2,1))).transpose(1,2,0)
			clique_messages[:,:,:,1] *= self.compatibility_inter.dot(np.append(factor_messages[:, 1:, :, 1], 
										np.ones((self.height, 1, 2)), axis = 1).transpose((0,2,1))).transpose(1,2,0)
			clique_messages[:,:,:,2] *= self.compatibility_inter.dot(np.append(factor_messages[1:,:, :, 2], 
										np.ones((1, self.width, 2)), axis = 0).transpose((0,2,1))).transpose(1,2,0)
			clique_messages[:,:,:,3] *= self.compatibility_inter.dot(np.append(np.ones((1, self.width, 2)), 	
										factor_messages[:-1,:, :, 3], axis = 0).transpose((0,2,1))).transpose(1,2,0)

			for j in range(clique_messages.shape[-1]):
				clique_norm[:,:,:,j] = np.sum(clique_messages[:,:,:,j], axis=2)[..., None]

			for j in range(clique_messages.shape[-1]):
				for ii in range(2):
					clique_messages[:, :, ii, j] /= np.squeeze(clique_norm[:,:,:,j])

		for j in range(clique_messages.shape[-1]):
			beta *= np.append(np.ones((self.height, 1, 2)), factor_messages[:,:-1,:,0], axis=1)\
					*np.append(np.ones((1, self.width, 2)), factor_messages[:-1,:,:,3], axis = 0)\
					*np.append(factor_messages[1:,:,:,2], np.ones((1, self.width, 2)), axis = 0)\
					*np.append(factor_messages[:, 1:,:,1], np.ones((self.height, 1, 2)), axis = 1)

		if self.add_noise:
			accuracy = np.sum(self.noiseless_image == np.argmax(beta, axis = 2)) #/(self.height*self.width)
			return(self.image, np.argmax(beta, axis = 2), self.noiseless_image, accuracy)
		else:
			return(self.image, np.argmax(beta, axis = 2))

params = {'theta':0.9, 'gamma':0.9}
L = LBP(results.image_path, 20, params, results.add_noise)

if results.mode == 'seq':
	L.messages_sync()
else:
	try:
		image, denoised, noiseless, accuracy = L.messages_sync()
		plt.figure(figsize=(10, 30))
		plt.subplot(1,3,1)
		plt.axis('off')
		plt.title('Original Image')	
		plt.imshow(noiseless, cmap = 'Greys_r')
		plt.subplot(1,3,2)
		plt.axis('off')
		plt.title('Noisy Image, Noise Probability={}'.format(0.1))
		plt.imshow(image, cmap = 'Greys_r')
		plt.subplot(1,3,3)
		plt.axis('off')
		plt.title('Denoised Image \n Accuracy = {0:.2f}, Theta = {1:.2f}, Gamma = {2:.2f}'.format(accuracy, 
			 params['theta'], params['gamma']))
		plt.imshow(denoised, cmap = 'Greys_r')
		plt.show()
	except:
		image, denoised = L.messages_sync()
		plt.figure(figsize=(10, 20))
		plt.subplot(1,2,1)
		plt.axis('off')
		plt.title('Noisy Image')
		plt.imshow(image, cmap = 'Greys_r')
		plt.subplot(1,2,2)
		plt.axis('off')
		plt.title('Denoised Image \n Accuracy = NA, Theta = {}, Gamma = {}'.format(params['theta'], params['gamma']))
		plt.imshow(denoised, cmap = 'Greys_r')
		plt.show()


# Code for performing grid search on Theta and Gamma values

# scores = np.zeros((len(np.arange(0, 1, 0.1)), len(np.arange(0, 1, 0.1)), 10))

# for ii in range(10):
# 	for i, theta in enumerate(list(np.arange(0, 1, 0.1))):
# 		for j, gamma in enumerate(list(np.arange(0, 1, 0.1))):
# 			params = {'theta':theta, 'gamma':gamma}
# 			L = LBP(results.image_path, 20, params, results.add_noise)
# 			image, denoised, noiseless, accuracy = L.messages_sync()
# 			scores[i, j, ii] = accuracy

# print(np.max(np.mean(scores, axis=2)))
# plt.imshow(np.mean(scores, axis=2), cmap='Reds')
# plt.xlabel('Theta')
# plt.ylabel('Gamma')
# plt.colorbar()
# plt.show()