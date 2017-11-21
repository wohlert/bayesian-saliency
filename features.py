from sklearn.feature_extraction.image import extract_patches_2d
from sklearn.decomposition import FastICA


def compute_components(images, size=11):
    """
    Computes the ICA components based on the input images
    by extracting `size x size x channels` image patches.

    :param images: (np.array) 4-dimensional array of natural images
    :param size (optional): (int) component height and width
    :return (np.array) ICA components
    -------------
    References:

    "Fast and Robust Fixed-Point Algorithms for Independent Component Analysis"
    (Hyvärinen 1999).
    """
    n, h, w, channels = images.shape

    patches = extract_patches_2d(np.vstack(images), (size, size), max_patches=50000)

    ica = FastICA()
    ica.fit(patches.reshape(-1, size*size*channels))

    components = (ica.components_ + ica.mean_).reshape(-1, size, size, channels)
    return components


def extract_patches(image, size=11):
    """
    Extracts image patches of size `size x size x channels` from a single colour
    image using Fortran array ordering.

    :param image: (np.array) 3-dimensional array of an image
    :param size (optional): (int) patch height and width
    :return (np.array) image patch array
    """
    n, m, c = image.shape
    n = n - size + 1
    m = m - size + 1

    dimensions = size**2 * c
    patches = np.zeros((dimensions, n*m))

    for i in range(n*m):
        col = i  % m
        row = i // m
        patches[:, i] = np.reshape(img[row:row+size, col:col+size, :], dimensions, "F")

    return patches


def compute_response(image, filters):
    """
    Computes the filter responses.

    :param image: (np.array) 3-dimensional array of an image
    :param filters: (np.array) 2-dimensional (n_filters, size**2 * channels)
        filter bank of ICA components.
    :return activation (np.array): activation after applying the filters to the image
    :return dimensions (np.array): dimensions of the image after activation
    """
    n_filters, filter_size = filters.shape
    size = int(np.sqrt(filter_size/3))

    n, m, _ = image.shape
    n = n - size + 1
    m = m - size + 1

    image_dimensions = (n, m)

    patches = extract_patches(image, size)

    activation = filters @ patches

    return activation, image_dimensions


def compute_saliency(image, filters, sigma=None, theta=None):
    """
    Computes the saliency of an image by basically convoluting an image
    with an ICA component filter bank.

    :param image: (np.array) 3-dimensional array of an image
    :param filters: (np.array) 2-dimensional (n_filters, size**2 * channels)
        filter bank of ICA components.
    :param sigma (optional): (np.array) array of shape parameters
    :param theta (optional): (np.array) array of scale parameters
    :return (np.array) 2-dimensional saliency map
    -------------
    References:

    "SUN: A Bayesian framework for saliency using natural statistics." (Zhang 2008).
    """
    n_filters, filter_size = filters.shape
    size = int(np.sqrt(filter_size/3))

    # If no weights, use uniform weighting
    if sigma is None:
        sigma = np.ones((n_filters))

    if theta is None:
        theta = np.ones((n_filters))

    activation, dimensions = compute_response(image, filters)
    saliency_map = sum([(np.abs(activation[i].T)/sigma[i])**theta[i] for i in range(n_filters)])

    # Output 2d grayscale saliency map
    return saliency_map.reshape(dimensions)


def fit_gaussian(images, filters):
    """
    Computes the shape and scale parameters of a generalized
    Gaussian distribution from a set of images.

    :param images (list): List of 3d image arrays
    :param filters (np.array): ICA filters
    :return sigmas: shapes of fitted Gaussian
    :return thetas: scales of fitted Gaussian
    -------------
    References:

    "A globally convergent and consistent method for estimating the shape
    parameter of a generalized Gaussian distribution" (Song 2006).
    """
    from scipy.stats import gennorm

    n_filters, _ = filters.shape

    responses = [compute_saliency(image, filters) for image in images]

    # Fit a Generalized Gaussian to the data while
    # keeping the mean fixed.
    # Fit distribution to each filter.
    (thetas, _, sigmas) = zip(*[gennorm.fit(filt, floc=0) for filt in responses)

    return sigmas, thetas
