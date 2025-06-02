import numpy as np
import matplotlib.pyplot as plt
from scipy.fftpack import dct, idct
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



def dct2(x): # 2D Discrete Cosine Transform
    return dct(dct(x.T, norm='ortho').T, norm='ortho')


def idct2(x): # 2D Inverse DCT
    return idct(idct(x.T, norm='ortho').T, norm='ortho')



def reconstruct_tv(corrupt):
    """Reconstruct an AFM image from compressed measurements using Total Variation minimization in the DCT domain."""
    if corrupt.max() == 300:
        mask = (corrupt != 300)
    else:
        mask = (corrupt != 0)  # Tip lifted means no data
    n = len(corrupt)
    y = corrupt[mask]  # Measured data

    # 2D DCT basis flattened
    Psi = np.kron(dct(np.eye(n, dtype=np.float32), norm='ortho'), dct(np.eye(n, dtype=np.float32), norm='ortho'))
    Phi = Psi[mask.flatten(), :]  # Select only rows corresponding to available data
    
    x = cp.Variable(n**2)
    objective = cp.Minimize(cp.tv(x.reshape((n,n), order='C')))
    constraints = [Phi @ x == y]
    prob = cp.Problem(objective, constraints)
    prob.solve()

    x_rec = x.value.reshape((n, n))
    return idct2(x_rec)  # Reconstructed image in spatial domain


def reconstruct_l1(corrupt):
    """Reconstruct an AFM image from compressed measurements using L1 minimization in the DCT domain."""
    if corrupt.max() == 300:
        mask = (corrupt != 300)
    else:
        mask = (corrupt != 0)  # Tip lifted means no data
    n = len(corrupt)
    y = corrupt[mask]  # Measured data


    # 2D DCT basis flattened
    Psi = np.kron(dct(np.eye(n), norm='ortho'), dct(np.eye(n), norm='ortho'))
    Phi = Psi[mask.flatten(), :]  # Select only rows corresponding to available data

    # Solve L1 minimization problem
    x = cp.Variable(n**2)
    objective = cp.Minimize(cp.norm1(x))
    constraints = [Phi @ x == y]
    prob = cp.Problem(objective, constraints)
    prob.solve()

    x_rec = x.value.reshape((n, n))
    return idct2(x_rec)  # Reconstructed image in spatial domain


# Total variation minimization of an image separated in blocks
def block_tv(corrupt, b):
    n = len(corrupt)
    rec_img = np.zeros((n,n))
    n_box = (n//b)**2
    k=0
    for i in range(0,n,b):
        for j in range(0,n,b):
            k += 1
            print(f"Analysis block {k}/{n_box}...")
            block = reconstruct_tv(corrupt[i:i+b,j:j+b])
            rec_img[i:i+b,j:j+b] = block
    return rec_img

# Total variation minimization of an image separated in blocks with an overlap of 25% between 2 consecutive blocks
def block_tv_25pc(corrupt, b):
    n = len(corrupt)
    # Count the number of block containing each pixel
    weight = np.ones((n,n)) # Number of time a pixel appears in a block (1, 2 or 4)
    m = (n-0.25*b)/(0.75*b) # Number of blocks in a row
    # Test to see if all the last block is at the end of the row.
    # If not, we will add another block that will cover the last bit but also part of the block before.
    if not m.is_integer():
        add_block = True
    else:
        add_block = False
    m = int(m)
    tf_b = 3*b//4 # 3/4th of a block: periodicity of the cube overlap
    for i in range(1,m):
        for k in range(b//4): # Assume the overlap of the blocks is 25 %
            weight[i*tf_b+k,:] = weight[i*tf_b+k,:] * 2
            weight[:,i*tf_b+k] = weight[:,i*tf_b+k] * 2

    if add_block:
        for i in range(b,b//4,-1):
            weight[n-i,:] = weight[n-i,:] * 2
            weight[:,n-i] = weight[:,n-i] * 2

    rec_img = np.zeros((n,n))
    k = 0
    l_blocks_row = 3*m*b//4 # Space taken by the blocks in a row
    t_block = m**2 # Total number of blocks 
    if add_block:
        t_block = t_block + 2*m + 1
    for i in range(0,l_blocks_row,3*b//4):
        for j in range(0,l_blocks_row,3*b//4):
            k += 1
            print(f"Analysis block {k}/{t_block}...")
            block = reconstruct_tv(corrupt[i:i+b,j:j+b])
            rec_img[i:i+b,j:j+b] = rec_img[i:i+b,j:j+b] + block
    if add_block:
        for i in range(0,l_blocks_row,3*b//4):
            k += 1
            print(f"Analysis block {k}/{t_block}...")
            block_right = reconstruct_tv(corrupt[n-b:n,i:i+b])
            rec_img[n-b:n,i:i+b] = rec_img[n-b:n,i:i+b] + block_right
            k += 1
            print(f"Analysis block {k}/{t_block}...")
            block_down = reconstruct_tv(corrupt[i:i+b,n-b:n])
            rec_img[i:i+b,n-b:n] = rec_img[i:i+b,n-b:n] + block_down
        k += 1
        print(f"Analysis block {k}/{t_block}...")
        block  = reconstruct_tv(corrupt[n-b:n,n-b:n])
        rec_img[n-b:n,n-b:n] = rec_img[n-b:n,n-b:n] + block
    rec_img = rec_img/weight # Get the average value for each pixel
    
    if rec_img.max() > 500:
        for i in range(n):
            for j in range(n):
                if rec_img[i,j] > 130 or rec_img[i,j] < -30:
                    rec_img[i,j] = 0
    
    return rec_img 


if __name__ == "__main__":

    b = 32 # Size of the blocks

    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    DATA_DIR = os.path.join(BASE_DIR, "data")
    IMG_DIR = os.path.join(BASE_DIR, "image")

    file_origin = "1_TMV_0.1_Au_TSGs_RH10__amp 2V_150701_114145.txt"
    file_corrupt = "1_TMV_0.1_Au_TSGs_RH10__amp 2V_150701_114145_corrupt50_1.txt"
    l_og = len(file_origin) - 4
    suffix = file_corrupt[l_og:-4]

    image = load_afm_txt(os.path.join(DATA_DIR, file_origin))
    corrupt_image = load_afm_txt(os.path.join(DATA_DIR, file_corrupt))

#    image = load_afm_txt(os.path.join(DATA_DIR, "c.txt"))
#    corrupt_image = load_afm_txt(os.path.join(DATA_DIR, "c4.txt"))

    # Visualize corrupted input
    plt.imshow(corrupt_image, cmap='hot')
    plt.colorbar(label='Height (z)')
    plt.title('Corrupted scan')

    plt.show()



    # Reconstruct
    rec_img = block_tv_25pc(corrupt_image, b)

    # Calculation of PSNR
    n = len(rec_img)
    max_amp = rec_img.max() - rec_img.min()
    mse = np.mean((image-rec_img)**2)
    psnr = 20*np.log10(max_amp) - 10*np.log10(mse)
    print("\n" + f"The PSNR between the original and reconstructed images is {'%0.2f' % psnr} dB." + "\n")

    # Calculation of SSIM
    # This function is excluding the edges for the calculation of ssim. Default is 3 rows and columns on each side ((7-1)/2)
    ssim1 = ssim(image, rec_img, data_range = max_amp)
    print("\n" + f"The SSIM between the original and reconstructed images is {'%0.3f' %ssim1}." + "\n")

    rec_img = (rec_img - rec_img.min()) / (rec_img.max() - rec_img.min()) * 100



    # Visualize
    plt.figure(figsize=(18, 6), layout='constrained')
    plt.subplot(1, 3, 1)
    plt.title('Partially Measured Image')
    plt.imshow(corrupt_image, cmap='hot')

    plt.subplot(1, 3, 2)
    plt.title('Original Image')
    plt.imshow(image, cmap='hot')

    plt.subplot(1, 3, 3)
    plt.title('Reconstructed Image')
    plt.imshow(rec_img, cmap='hot')
    plt.colorbar()

    plt.savefig(os.path.join(IMG_DIR, f"{n}px_{b}px-box_tv_overlap{suffix}.png"))
    plt.show()
