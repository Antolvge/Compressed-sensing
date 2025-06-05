import numpy as np
from random import randrange
import matplotlib.pyplot as plt
import os

class MethodError(Exception):
    pass

#Create a corrupt image with only some of data points with a micro-paths pattern
def create_mask_micro(image, coverage, k):
    if coverage > 1 or coverage < 0:
        raise ValueError("The coverage must be between 0 and 1")
    if not isinstance(k, int):
        raise ValueError("k must be an integer")
    
    l_row = len(image[0])
    n_pix = len(image) * len(image[0]) # In the case of rectangle images (not for AFM but SEM?)
    pix_cov = n_pix * coverage # Number of pixel with non-zero measurement
    n_paths = int(pix_cov // k) # Number of micro-paths

    corrupt_img = np.zeros_like(image)
    for i in range(n_paths):
        row_p = randrange(len(image)) # Select a starting point for the micro-path
        col_p = randrange(l_row)
        t = 0
        # Verify that it isn't at the en of a row, that it doesn't overlap with another path and that there is enough space between 2 paths
        while (col_p + k > l_row) or (corrupt_img[row_p, col_p] != 0) or (corrupt_img[row_p, col_p+k-1] != 0) or (col_p >= s and corrupt_img[row_p, col_p-s] != 0)  or (col_p < l_row- k -s and corrupt_img[row_p, col_p+k-1+s] != 0): 
            row_p = randrange(len(image))
            col_p = randrange(l_row)
            t += 1
            if t > 50:
                raise MethodError("The coverage is probably too high and the micropaths are not the most appropriate technique")
        for j in range(k):
            corrupt_img[row_p][col_p+j] = image[row_p][col_p+j]
    return corrupt_img


# Simulate a measurement where only 1 every 2 pixels is measured in every direction. In that case, only a quarter of the image is measured.
def create_mask_quarter(image):
    if (not begin.is_integer()) or begin < 1 or begin > 4:
        raise ValueError("'begin' must be equal to 1, 2, 3 or 4")
    i0 = (begin-1)%2
    j0 = (begin-1)//2
    n = len(image)
    corrupt_img = np.zeros((n,n))
    for i in range(i0,n,2):
        for j in range(j0,n,2):
            corrupt_img[i,j] = image[i,j] # Only take 1 every 2 pixels 
    return corrupt_img



if __name__ == "__main__":

    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    DATA_DIR = os.path.join(BASE_DIR, "data")
    file = "1_TMV_0.1_Au_TSGs_RH10__amp 2V_150701_114145.txt"
    im_file = os.path.join(DATA_DIR, file)

    with open(im_file, 'r', encoding='latin1') as f:
        lines = f.readlines()
    
    data = [float(val) for line in lines for val in line.strip().split()] # This part doesn't work for rectangle images. One should find the length of the rows first.
    n = np.sqrt(len(data))
    if not n.is_integer():
        raise ValueError(f"Expected a square-shaped image")
    image = np.array(data).reshape((int(n), int(n)))


    # Type of corrupt image. For now u-paths or quarter
    c_type = 'u-paths' 

    if c_type == 'quarter':
        begin = 1 # Possible values: 1, 2, 3 or 4 to select which first point to select in the top left 2x2 square of the image
        extension = '_quarter_' + str(begin)
        corrupt_img = create_mask_quarter(image)
    elif c_type == 'u-paths':
        coverage = 0.4 # Fraction of the surface measured
        k = 15 # Length of the micro-paths
        s = 6 # Space between 2 micro-paths
        extension = str(int(coverage*100)) + "_k" + str(k) + "_2"
        corrupt_img = create_mask_micro(image, coverage, k)
    else:
        raise ValueError(f"the algorithm {c_type} is not recognized. Try 'quarter' or 'u-paths' instead.")


    # Comparison plot between original and corrupted images
    fig, (ax1, ax2) = plt.subplots(1, 2, layout='constrained')
    norm = ax1.imshow(image, cmap="hot")
    corrupt = ax2.imshow(corrupt_img, cmap='hot')
    fig.colorbar(corrupt, label='Height (%)', location="right")
    fig.suptitle('Comparaison')
    plt.show()


    # File saving
    output_file = im_file[:-4] + "_corrupt" + extension + ".txt"
    np.savetxt(output_file, corrupt_img, fmt='%.4f', delimiter="\t")
    print("File saved", output_file)