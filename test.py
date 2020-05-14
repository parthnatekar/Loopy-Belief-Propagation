import numpy as np
import pandas as pd
from scipy.stats import gamma
import matplotlib.pyplot as plt
import cv2
from sklearn.model_selection import ParameterGrid
from datetime import datetime
start=datetime.now()

class Loopy():

	def __init__(self, grid, iterations, params):
		self.grid = grid
		self.height, self.width = self.grid.shape
		self.iterations = iterations
		self.params = params
		self.compatibility_inter = np.array([[1.0, self.params['theta']], [self.params['theta'], 1.0]])
		self.compatibility_outer = np.array([[1.0, self.params['gamma']], [self.params['gamma'], 1.0]])

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

	def get_neighbours_indexed(self, idx):

		neighbours = []
		if idx-self.size > 0:
			neighbours.append(idx-self.size)
		else:
			neighbours.append(np.nan)
		if idx+self.size < self.size**2:
			neighbours.append(idx+self.size)
		else:
			neighbours.append(np.nan)
		try:
			if np.unravel_index(idx-1, (self.size,self.size))[0] ==  np.unravel_index(idx, (self.size,self.size))[0]:
				neighbours.append(idx-1)
			else:
				neighbours.append(np.nan)
		except:
			neighbours.append(np.nan)
		try:
			if np.unravel_index(idx+1, (self.size,self.size))[0] ==  np.unravel_index(idx, (self.size,self.size))[0]:
				neighbours.append(idx+1)
			else:
				neighbours.append(np.nan)
		except:
			neighbours.append(np.nan)

		return(neighbours)

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

		accuracy = np.sum(self.grid == np.argmax(beta, axis = 2))/self.size**2
		plt.imshow(np.argmax(beta, axis = 2))
		plt.title('Denoised Image, Accuracy = {0:.2f}, Theta = {1:.2f}, Gamma = {2:.2f}'.format(accuracy, self.params['theta'], self.params['gamma']))
		plt.show()

	def messages_parallel(self):

		clique_messages_r = np.ones((self.size, self.size, 2))
		clique_messages_l = np.ones((self.size, self.size, 2))
		clique_messages_t = np.ones((self.size, self.size, 2))
		clique_messages_b = np.ones((self.size, self.size, 2))

		factor_messages_r = np.ones((self.size, self.size, 2))
		factor_messages_l = np.ones((self.size, self.size, 2))
		factor_messages_t = np.ones((self.size, self.size, 2))
		factor_messages_b = np.ones((self.size, self.size, 2))

		beta = np.ones((self.size, self.size, 2))

		for j in range(self.size**2):
			beta[np.unravel_index(j, (self.size, self.size))] = self.compatibility_outer[:, self.grid[np.unravel_index(j, 
				(self.size, self.size))]]

		# print(np.isfinite(neighbours)*1)

		for i in range(self.iterations):
			factor_messages_r *= beta*clique_messages_r*clique_messages_b*clique_messages_t
			factor_messages_l *= beta*clique_messages_l*clique_messages_b*clique_messages_t
			factor_messages_t *= beta*clique_messages_r*clique_messages_l*clique_messages_t
			factor_messages_b *= beta*clique_messages_r*clique_messages_l*clique_messages_b

			factor_norm_r = np.sum(factor_messages_r, axis=2)
			factor_norm_l = np.sum(factor_messages_l, axis=2)
			factor_norm_b = np.sum(factor_messages_b, axis=2)
			factor_norm_t = np.sum(factor_messages_t, axis=2)

			for ii in range(2):
				factor_messages_r[:, :, ii] /= factor_norm_r
				factor_messages_l[:, :, ii] /= factor_norm_l
				factor_messages_t[:, :, ii] /= factor_norm_t
				factor_messages_b[:, :, ii] /= factor_norm_b

			# print(np.append(factor_messages_l[:, 1:], np.ones((self.size, 1, 2)), axis = 1)[:,-1,:])
			# print(np.append(factor_messages_b[1:,:], np.ones((1, self.size, 2)), axis = 0))
			clique_messages_r *= self.compatibility_inter.dot(np.append(np.ones((self.size, 1, 2)), factor_messages_r[:,:-1], axis=1).transpose((0,2,1))).transpose()
			clique_messages_l *= self.compatibility_inter.dot(np.append(factor_messages_l[:, 1:], np.ones((self.size, 1, 2)), axis = 1).transpose((0,2,1))).transpose()
			clique_messages_t *= self.compatibility_inter.dot(np.append(factor_messages_t[1:,:], np.ones((1, self.size, 2)), axis = 0).transpose((0,2,1))).transpose()
			clique_messages_b *= self.compatibility_inter.dot(np.append(np.ones((1, self.size, 2)), factor_messages_b[:-1,:], axis = 0).transpose((0,2,1))).transpose()

			clique_norm_r = np.sum(clique_messages_r, axis=2)
			clique_norm_l = np.sum(clique_messages_l, axis=2)
			clique_norm_b = np.sum(clique_messages_b, axis=2)
			clique_norm_t = np.sum(clique_messages_t, axis=2)

			for ii in range(2):
				clique_messages_r[:, :, ii] /= clique_norm_r
				clique_messages_l[:, :, ii] /= clique_norm_l
				clique_messages_t[:, :, ii] /= clique_norm_t
				clique_messages_b[:, :, ii] /= clique_norm_b

		beta *= np.append(np.ones((self.size, 1, 2)), factor_messages_r[:,:-1], axis=1)\
				*np.append(np.ones((1, self.size, 2)), factor_messages_b[:-1,:], axis = 0)\
				*np.append(factor_messages_t[1:,:], np.ones((1, self.size, 2)), axis = 0)\
				*np.append(factor_messages_l[:, 1:], np.ones((self.size, 1, 2)), axis = 1)

		plt.imshow(np.argmax(beta, axis = 2))
		# plt.title('Denoised Image, Accuracy = {0:.2f}, Theta = {1:.2f}, Gamma = {2:.2f}'.format(accuracy, self.params['theta'], self.params['gamma']))
		plt.show()

	def messages_sync(self):	

		clique_messages = np.ones((self.height, self.width, 2, 4))
		factor_messages = np.ones((self.height, self.width, 2, 4))
		factor_norm = np.ones((self.height, self.width, 1, 4))
		clique_norm = np.ones((self.height, self.width, 1, 4))
		beta = np.ones((self.height, self.width, 2))

		for j in range(self.height*self.width):
			beta[np.unravel_index(j, (self.height, self.width))] = self.compatibility_outer[:, int(self.grid[np.unravel_index(j, 
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

		accuracy = np.sum((self.grid == np.argmax(beta, axis = 2))*1)/(self.height*self.width)
		# plt.imshow(np.argmax(beta, axis = 2), cmap = 'Greys_r')
		# plt.title('Denoised Image, Accuracy = {0:.2f}, Theta = {1:.2f}, Gamma = {2:.2f}'.format(accuracy, self.params['theta'], self.params['gamma']))
		# plt.show()
		return(np.argmax(beta, axis = 2), accuracy)

if __name__ == '__main__':

	size = 100
	flip_prob = 0.2
	grid = np.zeros((size, size), dtype='int64')
	for j in range(size**2):
		idx = np.unravel_index(j, (size, size))
		if ((idx[0]-50)**2+(idx[1]-50)**2)**0.5 <= 25:
			grid[idx] = 1
		thresh = np.random.random_sample()
		if thresh < flip_prob:
			grid[idx] = 1-grid[idx]

	image = cv2.imread('binary.png', cv2.IMREAD_GRAYSCALE)/255
	image = cv2.threshold(image, 0.5, 1, cv2.THRESH_BINARY)[1]
	noiseless_image = image.copy()
	for j in range(image.shape[0]*image.shape[1]):
		idx = np.unravel_index(j, (image.shape[0], image.shape[1]))
		thresh = np.random.random_sample()
		if thresh < flip_prob:
			image[idx] = 1-image[idx]

	plt.imshow(image, cmap = 'Greys_r')
	plt.title('Noisy Image')
	plt.show()

	params = {'theta':0.7, 'gamma':0.3}
	# for i in list(np.arange(0, 1, 0.1)):
	# 	for j in list(np.arange(0, 1, 0.1)):
	# 		params = {'theta':0.5, 'gamma':0.2}
	L = Loopy(image, 20, params)
	denoised, _ = L.messages_sync()

	LBP_time = datetime.now()

	print('LBP done in {}'.format(LBP_time-start))
	dst = cv2.fastNlMeansDenoising(image.astype('uint8'),None,1,3,5)
	NLM_time = datetime.now()
	print('NLM done in {}'.format(NLM_time-LBP_time))
	plt.figure(figsize=(10, 30))
	plt.subplot(1,3,1)
	plt.axis('off')
	# plt.title('Noisy Image')
	plt.imshow(noiseless_image, cmap = 'Greys_r')
	plt.subplot(1,3,2)
	plt.axis('off')
	# plt.title('Noisy Image')
	plt.imshow(denoised, cmap = 'Greys_r')
	plt.subplot(1,3,3)
	plt.axis('off')
	# plt.title('Denoised Image \n Accuracy = NA, Theta = {}, Gamma = {}'.format(params['theta'], params['gamma']))
	plt.imshow(dst, cmap = 'Greys_r')
	plt.show()
		

	# im_csv = np.zeros((image.shape[0]*image.shape[1], 3))

	# for i in range(im_csv.shape[0]):
	# 	index = np.unravel_index(i, (image.shape[0], image.shape[1]))
	# 	im_csv[i,0] = index[0]
	# 	im_csv[i,1] = index[1]
	# 	im_csv[i,2] = image[index]

	# np.savetxt('circle_csv.csv', im_csv, delimiter = ',')
	# parameters = {'theta':list(np.arange(0,1,0.1)), 'gamma':list(np.arange(0,1,0.1))}
	# p_grid = ParameterGrid(parameters)	
	# acc_dict = {}
	# for params in p_grid:
	# 	L = Loopy(grid, 50, params)
	# 	acc_dict[tuple(params.values())] = L.message_dict_sync()

	# print(acc_dict)