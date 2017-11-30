import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat
from scipy.stats import gennorm
from features import extract_patches, compute_components, compute_saliency, compute_response
from PIL import Image
from multiprocessing import Pool, cpu_count
from functools import partial


show_example = True

if __name__ == '__main__':

    tokyo_path = "./eizaburo-doi-kyoto_natim-c2015ff"
    base_path = "{}/osLMS0103-{}.mat"

    def load_tokyo(path):
        """
        Load the Tokyo dataset each matrix at a time and
        combine the color changes such that (L, M, S) <=> (R, G, B)
        and all images have dimensions (250, 320).
        """
        channels = ("OL", "OM", "OS")
        image = np.dstack([loadmat(path)[c] for c in channels])
        if image.shape != (500, 640, 3):
            image = image.transpose(1, 0, 2)
        image = Image.fromarray((image*255).astype(np.uint8))
        image.thumbnail((320, 320))
        image = np.array(image)/255
        return image

    try:
        filters = np.load("filters.npy")

    except FileNotFoundError:

        print("1.1: Loading images.")
        images = np.array([load_tokyo(base_path.format(tokyo_path, i)) for i in range(1, 20)])

        print("1.2: Extracting patches.")
        patches = extract_patches(np.vstack(images), size=11).T

        # Only use the 50000 random patches
        np.random.shuffle(patches)
        patches = patches[:50000]

        # Compute ICA components
        print("1.3: Computing ICA components (this might take a while).")
        ica_components = compute_components(patches)

        # Save centered components as filters
        filters = ica_components - np.mean(ica_components, 0)
        np.save("filters", filters)

    if show_example:
        print("Loading image for display.")
        img = Image.open("../CAT2000_train/Stimuli/OutdoorNatural/031.jpg")
        img.thumbnail((512, 512))
        img = np.array(img)/255

        print("Computing responses and saving file `response.png`.")
        saliency_sun = compute_saliency(img, filters)
        plt.imsave("response.png", saliency_sun)

    try:
        theta = np.load("theta.npy")
        sigma = np.load("sigma.npy")

    except FileNotFoundError:
        # Use the rest of the natural images to estimate parameters
        print("2.1: Loading rest of dataset.")
        images = np.array([load_tokyo(base_path.format(tokyo_path, i)) for i in range(21, 35)])

        print("2.2: Computing total responses over dataset.")
        total_response, _ = compute_response(np.vstack(images), filters)

        # Use multiprocessing to speed up computation
        # as each filter is independent.
        print("2.3: Estimating parameters (this might take a while).")
        pool = Pool(cpu_count())
        model_fitting = partial(gennorm.fit, floc=0)
        filter_parameters = pool.map(model_fitting, total_response)

        theta, _, sigma = zip(*filter_parameters)
        theta = np.array(theta)
        sigma = np.array(sigma)

        np.save("theta", theta)
        np.save("sigma", sigma)

    if show_example:
        print("Computing weighted responses and saving file `response-weighted.png`.")
        saliency_sun = compute_saliency(img, filters, sigma=sigma, theta=theta)
        plt.imsave("response-weighted.png", saliency_sun)
