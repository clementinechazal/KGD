import numpy as np

#######GAUSSIAN KERNEL#######

def k_gauss(x,y,sigma): #Gram matrix for gaussian kernel
    X_norm = np.sum(x**2, axis=1)
    Y_norm = np.sum(y**2, axis=1)
    xy = np.dot(x, y.T)
    dist = X_norm[:,None] + Y_norm[None,:] - 2 * xy
    return np.exp(-dist/(2*sigma**2))

def dk_gauss(x,y,sigma): #Gram matrix of nabla_1 k(x,y) , dim = (n,m,d)
    x_mat = np.tile(x, (len(y), 1, 1))
    y_mat = np.tile(y, (len(x), 1, 1))
    x_mat = np.transpose(x_mat, axes=(1, 0, 2))
    diff = y_mat - x_mat
    return diff/(sigma**2) * k_gauss(x,y,sigma)[:,:,None]

def ddk_gauss(x,y,sigma): #Gram matrix of nabla_2 . nabla_1 k(x,y), dim = (n,m)
    d = len(x[0])
    X_norm = np.sum(x**2, axis=1)
    Y_norm = np.sum(y**2, axis=1)
    xy = np.dot(x, y.T)
    dist = X_norm[:,None] + Y_norm[None,:] - 2 * xy
    K = np.exp(-dist/(2*sigma**2))
    return 1/sigma**2 * K *(d - 1/sigma**2 * dist)


######## INVERSE MULTIQUADRIC KERNEL ########






######## LAPLACE KERNEL ########
def k_laplace(x,y,sigma): #Gram matrix for laplace kernel
    X_norm = np.sum(x**2, axis=1)
    Y_norm = np.sum(y**2, axis=1)
    xy = np.dot(x, y.T)
    dist = X_norm[:,None] + Y_norm[None,:] - 2 * xy
    return np.exp(-np.sqrt(dist)/sigma)

def dk_laplace(x, y, sigma): # Gram matrix of nabla_1 k(x,y), dim = (n,m,d)
    x_mat = np.tile(x, (len(y), 1, 1))
    y_mat = np.tile(y, (len(x), 1, 1))
    x_mat = np.transpose(x_mat, axes=(1, 0, 2))
    diff = y_mat - x_mat
    dist = np.linalg.norm(diff, axis=2) + np.ones(len(x))  # Compute the Euclidean distance
    grad = np.zeros_like(diff)  # Initialize gradient to zero
    #non_zero_mask = dist > 0  # Mask for non-zero distances
    #expanded_mask = np.concatenate((non_zero_mask[:,:,None],non_zero_mask[:,:,None]), axis=2)
    #print(grad[expanded_mask].shape)
    #grad[expanded_mask] = -diff[expanded_mask] / (sigma * dist[non_zero_mask][:, :, None]) * np.exp(-dist[non_zero_mask] / sigma)[:, :, None]
    grad = -diff / (sigma * dist[:, :, None]) * np.exp(-dist / sigma)[:, :, None]
    return grad

def ddk_laplace(x, y, sigma): # Gram matrix of nabla_2 . nabla_1 k(x,y), dim = (n,m)
    x_mat = np.tile(x, (len(y), 1, 1))
    y_mat = np.tile(y, (len(x), 1, 1))
    x_mat = np.transpose(x_mat, axes=(1, 0, 2))
    diff = y_mat - x_mat
    dist = np.linalg.norm(diff, axis=2)  # Compute the Euclidean distance
    K = np.exp(-dist / sigma)  # Laplace kernel values
    d = x.shape[1]  # Dimensionality of the input data

    # Initialize second derivative matrix to zero
    second_derivative = np.zeros((x.shape[0], y.shape[0]))

    # Mask for non-zero distances
    non_zero_mask = dist > 0

    # Compute second derivative for non-zero distances
    second_derivative[non_zero_mask] = K[non_zero_mask] * (
        (d - 1) / (sigma**2 * dist[non_zero_mask]) - 1 / (sigma**3)
    )

    return second_derivative

