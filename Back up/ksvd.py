import numpy as np
import cv2
from skimage.util import view_as_windows, view_as_blocks
from numpy.linalg import svd
from sklearn.linear_model import OrthogonalMatchingPursuit
import time

def initialize_dictionary(data, num_atoms):
    """ Randomly selects 'num_atoms' samples from the data to initialize the dictionary. """
    indices = np.random.choice(data.shape[1], num_atoms, replace=False)
    return data[:, indices]

def sparse_coding(dictionary, data, sparsity):
    """ Solves the sparse coding problem using Orthogonal Matching Pursuit (OMP). """
    omp = OrthogonalMatchingPursuit(n_nonzero_coefs=sparsity)
    omp.fit(dictionary, data)
    return omp.coef_.T

def update_dictionary(dictionary, data, sparse_codes, num_atoms):
    """ Updates the dictionary using the SVD approach. """
    for j in range(num_atoms):
        # Find the data points that use the j-th atom
        indices = np.nonzero(sparse_codes[j, :])[0]
        if len(indices) == 0:
            continue
        
        # Extract the relevant data
        data_subset = data[:, indices]
        sparse_subset = sparse_codes[:, indices]
        
        # Zero out the contribution of the j-th atom
        sparse_subset[j, :] = 0
        residual = data_subset - dictionary @ sparse_subset
        
        # Update the atom using SVD
        U, S, Vt = svd(residual, full_matrices=False)
        dictionary[:, j] = U[:, 0] / np.linalg.norm(U[:, 0])  # Normalize
        sparse_codes[j, indices] = S[0] * Vt[0, :]
    
    return dictionary, sparse_codes

def ksvd(data, num_atoms, sparsity, max_iter=20):
    """ Runs the K-SVD algorithm for dictionary learning. """
    dictionary = initialize_dictionary(data, num_atoms)
    
    for _ in range(max_iter):
        sparse_codes = sparse_coding(dictionary, data, sparsity)
        dictionary, sparse_codes = update_dictionary(dictionary, data, sparse_codes, num_atoms)
    
    return dictionary, sparse_codes

def extract_patches(image, patch_size, step=1):
    """ Extract overlapping patches from the image. """
    patches = view_as_windows(image, (patch_size, patch_size), step)
    return patches.reshape(-1, patch_size**2).T  # Reshape into (patch_size^2, num_patches)

def reconstruct_from_patches(patches, image_shape, patch_size, step=1):
    """ Reconstruct the image from overlapping patches. """
    h, w = image_shape
    output_image = np.zeros((h, w))
    weight_matrix = np.zeros((h, w))
    
    patch_index = 0
    for i in range(0, h - patch_size + 1, step):
        for j in range(0, w - patch_size + 1, step):
            output_image[i:i+patch_size, j:j+patch_size] += patches[:, patch_index].reshape(patch_size, patch_size)
            weight_matrix[i:i+patch_size, j:j+patch_size] += 1
            patch_index += 1

    return output_image / np.maximum(weight_matrix, 1)  # Normalize

def ksvd_for_image(image, patch_size=8, num_atoms=64, sparsity=3, max_iter=1):
    """ Apply K-SVD on image patches and reconstruct the image. """
    # Extract patches
    patches = extract_patches(image, patch_size)

    # Run K-SVD on patches
    s =time.perf_counter()
    dictionary, sparse_codes = ksvd(patches, num_atoms, sparsity, max_iter)
    e = time.perf_counter()
    print(e-s)
    # Reconstruct patches
    s = time.perf_counter()
    reconstructed_patches = dictionary @ sparse_codes
    e = time.perf_counter()
    print(e -s )
    # Reconstruct image
    reconstructed_image = reconstruct_from_patches(reconstructed_patches, image.shape, patch_size)
    return reconstructed_image

# Load and preprocess image

image = cv2.imread('cathedral .pgm', cv2.IMREAD_UNCHANGED)  # Read in grayscale)
print(image.shape)
image = image.astype(np.float32) / 255.0  # Normalize
time_start = time.perf_counter()
# Apply K-SVD
reconstructed_image = ksvd_for_image(image, patch_size=32, num_atoms=64, sparsity=3, max_iter=1)
time_end = time.perf_counter()
print((time_end-time_start))
# Display original and processed images
cv2.imshow("Original Image 2", image)
cv2.imshow("Reconstructed Image 2", reconstructed_image)
cv2.waitKey(0)
cv2.destroyAllWindows()

# Save the output image
cv2.imwrite("ksvd_output.jpg", (reconstructed_image * 255).astype(np.uint8))