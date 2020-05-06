import numpy as np
import pandas as pd
from scipy.stats import gamma
import matplotlib.pyplot as plt
import cv2

class Loopy():

	def __init__(self, grid, iterations, params):
		self.grid = grid
		self.size = self.grid.shape[0]
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

	def message_dict(self):

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

		accuracy = np.sum(self.grid == np.argmax(beta, axis = 2))*1/self.size**2
		plt.imshow(np.argmax(beta, axis = 2))
		plt.title('Denoised Image, Accuracy = {0:.2f}, Theta = {1:.2f}, Gamma = {2:.2f}'.format(accuracy, self.params['theta'], self.params['gamma']))
		plt.show()

	def message_dict_parallel(self):

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

	def message_dict_sync(self):

		clique_messages = np.ones((self.size, self.size, 2, 4))
		factor_messages = np.ones((self.size, self.size, 2, 4))
		factor_norm = np.ones((self.size, self.size, 1, 4))
		clique_norm = np.ones((self.size, self.size, 1, 4))
		beta = np.ones((self.size, self.size, 2))

		for j in range(self.size**2):
			beta[np.unravel_index(j, (self.size, self.size))] = self.compatibility_outer[:, self.grid[np.unravel_index(j, 
				(self.size, self.size))]]

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

			
			clique_messages[:,:,:,0] *= self.compatibility_inter.dot(np.append(np.ones((self.size, 1, 2)), 
										factor_messages[:,:-1,:, 0], axis=1).transpose((0,2,1))).transpose()
			clique_messages[:,:,:,1] *= self.compatibility_inter.dot(np.append(factor_messages[:, 1:, :, 1], 
										np.ones((self.size, 1, 2)), axis = 1).transpose((0,2,1))).transpose()
			clique_messages[:,:,:,2] *= self.compatibility_inter.dot(np.append(factor_messages[1:,:, :, 2], 
										np.ones((1, self.size, 2)), axis = 0).transpose((0,2,1))).transpose()
			clique_messages[:,:,:,3] *= self.compatibility_inter.dot(np.append(np.ones((1, self.size, 2)), 	
										factor_messages[:-1,:, :, 3], axis = 0).transpose((0,2,1))).transpose()

			for j in range(clique_messages.shape[-1]):
				clique_norm[:,:,:,j] = np.sum(clique_messages[:,:,:,j], axis=2)[..., None]

			for j in range(clique_messages.shape[-1]):
				for ii in range(2):
					clique_messages[:, :, ii, j] /= np.squeeze(clique_norm[:,:,:,j])

		for j in range(clique_messages.shape[-1]):
			beta *= np.append(np.ones((self.size, 1, 2)), factor_messages[:,:-1,:,0], axis=1)\
					*np.append(np.ones((1, self.size, 2)), factor_messages[:-1,:,:,3], axis = 0)\
					*np.append(factor_messages[1:,:,:,2], np.ones((1, self.size, 2)), axis = 0)\
					*np.append(factor_messages[:, 1:,:,1], np.ones((self.size, 1, 2)), axis = 1)

		plt.imshow(np.argmax(beta, axis = 2))
		# plt.title('Denoised Image, Accuracy = {0:.2f}, Theta = {1:.2f}, Gamma = {2:.2f}'.format(accuracy, self.params['theta'], self.params['gamma']))
		plt.show()

if __name__ == '__main__':

	size = 200
	flip_prob = 0.2
	grid = np.zeros((size, size), dtype='int64')
	for j in range(size**2):
		idx = np.unravel_index(j, (size, size))
		if ((idx[0]-50)**2+(idx[1]-50)**2)**0.5 <= 50:
			grid[idx] = 1
		thresh = np.random.random_sample()
		if thresh < flip_prob:
			grid[idx] = 1-grid[idx]

	plt.imshow(grid)
	plt.title('Noisy Image')
	plt.show()

	params = {'theta':0.4, 'gamma':0.2}
	L = Loopy(grid, 100, params)
	L.message_dict_sync()

