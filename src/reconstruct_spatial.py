import numpy as np
import matplotlib.pyplot as plt
import cvxpy as cp
import os
from skimage.metrics import structural_similarity as ssim

def skip_header(file_path):
    with open(file_path, 'r', encoding='latin1') as f:
        for line in f:
            if not line.startswith('#'):
                yield line

def load_afm_txt(file_path):
    data = []
    for line in skip_header(file_path):
        data = data + [float(val) for val in line.split()]
    n = np.sqrt(len(data))
    if not n.is_integer():
        raise ValueError(f"Expected a square-shaped image from {file_path}")
    return np.array(data).reshape((int(n), int(n)))

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")

#full_image = load_afm_txt(os.path.join(DATA_DIR, "HA 2uM DOPC NTA 10 0.5mM_Sln_201202_153921.txt"))
#partial_image = load_afm_txt(os.path.join(DATA_DIR, "HA 2uM DOPC NTA 10 0.5mM_Sln_201202_153921_corrupt_quarter.txt"))

full_image = load_afm_txt(os.path.join(DATA_DIR, "1_TMV_0.1_Au_TSGs_RH10__amp 2V_150701_114145.txt"))
partial_image = load_afm_txt(os.path.join(DATA_DIR, "1_TMV_0.1_Au_TSGs_RH10__amp 2V_150701_114145_corrupt_sqspiral_cov30.txt"))

#full_image = load_afm_txt(os.path.join(DATA_DIR, "c.txt"))
#partial_image = load_afm_txt(os.path.join(DATA_DIR, "c4.txt"))

measured_mask = (partial_image != 0)
partial_image = partial_image * measured_mask


# Example: Create a synthetic partially measured image
n = len(full_image)

X = cp.Variable((n, n))
objective = cp.Minimize(cp.tv(X))

# X should match the measured pixels
constraints = [X[measured_mask] == partial_image[measured_mask]]
problem = cp.Problem(objective, constraints)
problem.solve(solver=cp.CLARABEL, verbose=True)
reconstructed_image = X.value


max_amp = 100
mse = np.mean((full_image - reconstructed_image)**2)
psnr = 20*np.log10(max_amp) - 10*np.log10(mse)
print("\n" + f"The PSNR between the original and reconstructed images is {"%.2f" % psnr} dB." + "\n")

ssim1 = ssim(full_image, reconstructed_image, data_range = max_amp)
print("\n" + f"The SSIM between the original and reconstructed images is {ssim1}." + "\n")


# Visualize the results
plt.figure(figsize=(18, 6))
plt.subplot(1, 3, 1)
plt.title('Partially Measured Image')
plt.imshow(partial_image, cmap='hot')

plt.subplot(1, 3, 2)
plt.title('Original Image')
plt.imshow(full_image, cmap='hot')

plt.subplot(1, 3, 3)
plt.title('Reconstructed Image')
plt.imshow(reconstructed_image, cmap='hot')

plt.show()


