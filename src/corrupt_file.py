import numpy as np
import math
from random import randrange
import matplotlib.pyplot as plt
import os

class MethodError(Exception):
    pass

#Create a corrupt image with only some of data points with a micro-paths pattern
def create_mask_micro(image, coverage, k, s):

    """ Create a corrupted image with micro-paths measurements. The rest of the pixels are put to 0.
     
    Parameters:
    -----------
    image (array): Image to corrupt
    coverage (float): Fraction of the image measured (between 0 and 1)
    k (int): length of the micro-paths (number of pixels)
    s (int): minimal space between two micro-paths (number of pixels)
    
    Return:
    -------
    corrupt_img (array): Corrupted image composed of micro-paths measurements of length k with a minimal distance between them of s pixels and total coverage of coverage.
    
    To be noted: the k and s parameters cannot be any value and depend on the AFM.
    In our case, k must be greater than 15 pixels and s greater than 6 pixels (4 + the 2 pixels at the ends of the micro-paths that should be removed).
    It is limked to the time taken for the tip to go up and down. """

    if coverage > 1 or coverage < 0:
        raise ValueError("The coverage must be between 0 and 1")
    if not isinstance(k, int):
        raise ValueError("k must be an integer")
    if not isinstance(s, int):
        raise ValueError("s must be an integer")
    
    l_row = len(image[0])
    n_pix = len(image) * len(image[0]) # In the case of rectangle images (not for AFM but SEM?)
    pix_cov = n_pix * coverage # Number of pixels with non-zero measurement
    n_paths = int(pix_cov // k) # Number of micro-paths

    corrupt_img = np.zeros_like(image)
    for i in range(n_paths):
        row_p = randrange(len(image)) # Select a starting point for the micro-path
        col_p = randrange(l_row)
        t = 0
        # Verify that it isn't at the end of a row, that it doesn't overlap with another path and that there is enough space between 2 paths
        while   (col_p + k > l_row) or \
                (corrupt_img[row_p, col_p] != 0) or \
                (corrupt_img[row_p, col_p+k-1] != 0) or \
                (col_p >= s and corrupt_img[row_p, col_p-s] != 0)  or \
                (col_p < l_row- k -s and corrupt_img[row_p, col_p+k-1+s] != 0): 
            row_p = randrange(len(image))
            col_p = randrange(l_row)
            t += 1
            if t > 1000:
                raise MethodError("The coverage is probably too high and the micropaths are not the most appropriate technique")
        for j in range(k):
            corrupt_img[row_p][col_p+j] = image[row_p][col_p+j]
    return corrupt_img


# Simulate a measurement where only 1 every 2 pixels is measured in every direction. In that case, only a quarter of the image is measured.
def create_mask_quarter(image, begin):
    if (not begin.is_integer()) or begin < 1 or begin > 4:
        raise ValueError("'begin' must be equal to 1, 2, 3 or 4")
    i0 = (begin-1)%2
    j0 = (begin-1)//2
    n = len(image)
    corrupt_img = np.zeros_like(image)
    for i in range(i0,n,2):
        for j in range(j0,n,2):
            corrupt_img[i,j] = image[i,j] # Only take 1 every 2 pixels 
    return corrupt_img


def create_mask_micro_unif(image, k, period, begin):
    """Create an image with micro-paths of length k seperated by k pixels. The coverage is 50 %"""

    if not math.log2(k).is_integer():
        raise ValueError("k must be a power of 2")
    if not isinstance(k//period, int):
        raise ValueError("period must be an integer that divides k")
    if not begin.is_integer() or begin <= 0 or begin > k:
        raise ValueError("begin must be a integer between 1 and period")
    n = len(image)
    corrupt_img = np.zeros_like(image)
    # For each line, the pattern is the same: k pixels measured followed by k pixels not measured and so on.
    # On the next line, the pattern is horizontally translated by 2*k//period pixels.
    # When the path reaches the end of the line, it stops with a length lesser than k
    for i in range(0, n, period):
        for j in range(0, n, 2*k//period):
            l = (j*period//(2*k) + begin - 1) % period
            corrupt_img[i+l ,j:j+k] = image[i+l, j:j+k]
    # Add the partial paths at the beginning of each line in the case period > 2
    if period > 2:
        for i in range(n):
            # End of the partial path at the beginning of each line. We add period//2 because, vertically, we start by no path
            end = ((i+(begin-1)+period//2)%period) * 2*k//period
            corrupt_img[i,max(0,end-k):end] = image[i,max(0,end-k):end]
    return corrupt_img


def create_mask_row_scan(image, coverage):
    if coverage > 100 or coverage < 0:
        raise ValueError("The coverage must be between 0 and 1")
    
    n = len(image)
    corrupt_img = np.zeros_like(image)
    rand_row = np.random.random(n)
    per = np.percentile(rand_row, coverage)
    for i in range(n):
        if rand_row[i] <= per:
            corrupt_img[i] = image[i]
    return corrupt_img


def create_mask_square_spiral(image, coverage):
    if coverage > 0.5 or coverage < 0:
        raise ValueError("The coverage only works for coverage between 0 and 0.5. Other values may be added in the future.")
    
    n = len(image)
    corrupt_img = np.zeros_like(image)
    period = 1
    while not (2*coverage*period).is_integer(): # Frist multiple of 0.25 reached (possibility to change the value if needed)
        period += 1
        if period == 100:
            raise ValueError(f"{coverage} is too complicated for the current algorithm. Try simpler numbers.")

    n_steps = int(2 * coverage * period)
    steps = np.zeros(n_steps, dtype=int)
    for i in range(period):
        steps[i%n_steps] += 1

    corrupt_img[0] = image[0]
    s = n
    row = 0
    col = 0
    orientation = 0 # Top, right, bottom, left, 1, 2, 3 and 4 respectively 
    while s >= 0:
        if orientation%4 == 0:
            corrupt_img[row, col:col+s] = image[row, col:col+s]
            col = col + s - 1
        elif orientation%4 == 1:
            corrupt_img[row:row+s, col] = image[row:row+s, col]
            row = row + s - 1
        elif orientation%4 == 2:
            corrupt_img[row, col-s+1:col+1] = image[row, col-s+1:col+1]
            col = col - s + 1
        else:
            corrupt_img[row-s+1:row+1, col] = image[row-s+1:row+1, col]
            row = row - s + 1
        s = s - steps[orientation%n_steps]
        orientation += 1
    num = np.count_nonzero(corrupt_img)
    
    exact_cov = np.round(100*num/n**2,1)
    error = '%.2g' % (abs(num/n**2 - coverage)/coverage * 100)
    print(f"\nThe real coverage is {exact_cov} %. \nThe error from the expected coverage is {error} %.")
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
    c_type = 'row scan' 

    if c_type == 'quarter':
        begin = 1 # Possible values: 1, 2, 3 or 4 to select which first point to select in the top left 2x2 square of the image
        extension = '_quarter_' + str(begin)
        corrupt_img = create_mask_quarter(image, begin)
    elif c_type == 'u-paths':
        coverage = 0.5 # Fraction of the surface measured
        k = 15 # Length of the micro-paths
        s = 6 # Space between 2 micro-paths
        extension = str(int(coverage*100)) + "_k" + str(k) + "_s" + str(s) + "_5"
        corrupt_img = create_mask_micro(image, coverage, k, s)
    elif c_type == 'unif':
        k = 16 # Length of the micro-paths
        period = 8 # Number of lines of the pattern. Must be a divider of k
        begin = 1 # Integer between 1 and 'period'. Line of the pattern from which to start.
        extension = f"_unif_k{k}_per{period}_{begin}"
        corrupt_img = create_mask_micro_unif(image, k, period, begin)
    elif c_type == 'row scan':
        coverage = 0.5 # Fraction of the rows measured
        extension = f"_row_cov{str(int(coverage*100))}_1"
        corrupt_img = create_mask_row_scan(image, int(coverage*100))
    elif c_type == 'square spiral':
        coverage = 0.3 # Should be between 0 and 0.5. It currently works for 0.1, 0.2, 0.25, 0.3, 0.4 and 0.5 for sure. Other values may give strange results.
        extension = f"_sqspiral_cov{str(int(coverage*100))}"
        corrupt_img = create_mask_square_spiral(image, coverage)
    else:
        raise ValueError(f"the algorithm {c_type} is not recognized. Try 'quarter', 'unif', 'row scan', 'square spiral' or 'u-paths' instead.")


    # Comparison plot between original and corrupted images
    fig, (ax1, ax2) = plt.subplots(1, 2, layout='constrained')
    norm = ax1.imshow(image, cmap="hot")
    corrupt = ax2.imshow(corrupt_img, cmap='hot')
    fig.colorbar(corrupt, label='Height (%)', location="right")
    fig.suptitle('Comparison')
    plt.show()


    # File saving
    output_file = im_file[:-4] + "_corrupt" + extension + ".txt"
    np.savetxt(output_file, corrupt_img, fmt='%.4f', delimiter="\t")
    print(f"\nFile saved: {output_file}\n")