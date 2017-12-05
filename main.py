import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat
from scipy.stats import gennorm, norm, multivariate_normal
from scipy.integrate import simps
from features import extract_patches, compute_components, compute_saliency, compute_response
from PIL import Image
from multiprocessing import Pool, cpu_count
from functools import partial
import argparse
plt.rcParams["image.cmap"] = "binary_r"


H, W = 228, 512

parser = argparse.ArgumentParser()
parser.add_argument("image_id")
parser.add_argument("prior")

args = parser.parse_args()
image_id = int(args.image_id)
include_prior = True if args.prior is "True" else False


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

    def load_fixation(image_id):
        """
        Load the fixation map for a given image and return
        the downscaled points of fixation
        """
        ratio = (1980/512)

        fixations = loadmat("./CAT2000_train/FIXATIONLOCS/OutdoorNatural/{0:03d}.mat".format(image_id))["fixLocs"]
        y, x = np.where(fixations == 1)
        x, y = x//ratio, y//ratio
        return x, y

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

    print("Loading image for display.")
    img = Image.open("./CAT2000_train/Stimuli/OutdoorNatural/{0:03d}.jpg".format(image_id))
    fmap = Image.open("./CAT2000_train/FIXATIONMAPS/OutdoorNatural/{0:03d}.jpg".format(image_id))
    x, y = load_fixation(31)

    # Half image size
    img.thumbnail((512, 512))
    img = np.array(img)/255

    fmap.thumbnail((512, 512))
    fmap = np.array(fmap)/255

    print("Computing responses and saving file `response.png`.")
    saliency = compute_saliency(img, filters)

    f, axarr = plt.subplots(2, 2, figsize=(10, 10))
    axarr[0, 0].imshow(img)
    axarr[0, 1].imshow(saliency)
    axarr[1, 0].imshow(fmap)
    axarr[1, 1].imshow(img)
    axarr[1, 1].scatter(x, y, s=1, c="r")

    plt.imsave("response.png", saliency)

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
        pool = Pool(cpu_count()*2)
        model_fitting = partial(gennorm.fit, floc=0)
        filter_parameters = pool.map(model_fitting, total_response)

        theta, _, sigma = zip(*filter_parameters)
        theta = np.array(theta)
        sigma = np.array(sigma)

        np.save("theta", theta)
        np.save("sigma", sigma)

    print("Computing weighted responses and saving file `response-weighted.png`.")
    saliency = compute_saliency(img, filters, sigma=sigma, theta=theta)

    mat = loadmat("./stats.mat")

    k_filters = mat["B1"]
    k_sigma = mat["sigmas"].reshape(-1)
    k_theta = mat["thetas"].reshape(-1)

    saliency_kanan = compute_saliency(img, k_filters, sigma=k_sigma, theta=k_theta)
    plt.imsave("response-weighted.png", saliency)

    try:
        prior = np.load("prior.npy")
    except:
        print("3.1: Estimate variance for prior.")
        x, y = zip(*[load_fixation(2*i+1) for i in range(100)])
        x, y = np.hstack(x), np.hstack(y)

        # Estimate the variance over each dimension
        # individually by fitting a normal distribution.
        (_, var_x) = norm.fit(x, floc=W//2)
        (_, var_y) = norm.fit(y, floc=H//2)

        var = max(var_x, var_y)

        h, w = saliency.shape
        x, y = np.mgrid[0:h:1, 0:w:1]
        grid = np.empty(x.shape + (2,))
        grid[:, :, 0], grid[:, :, 1] = x, y

        gaussian = multivariate_normal(mean=(h//2, w//2), cov=var*np.eye(2))
        prior = np.log(gaussian.pdf(grid) + 1e-8)
        np.save("prior", prior)

    # Calculate ratio between self-information and prior
    ratio = -(saliency.max() / prior.max())/2

    print("Plotting maps and saving as `comparison.pdf`")
    f, ((ax0, ax1, ax2), (ax3, ax4, ax5)) = plt.subplots(2, 3, figsize=(12, 6))

    x, y = load_fixation(image_id)

    ax0.imshow(img)
    ax0.scatter(x, y, s=2, c="r")
    ax0.set_title("Input image")
    ax0.axis("off")

    ax1.imshow(fmap)
    ax1.set_title("Attention map")
    ax1.axis("off")

    ax2.imshow(saliency_kanan)
    ax2.set_title("Kanan, Cottrell (2010)")
    ax2.axis("off")

    ax3.imshow(saliency)
    ax3.set_title("Ours")
    ax3.axis("off")

    ax4.imshow(prior)
    ax4.set_title("Prior")
    ax4.axis("off")

    ax5.imshow(saliency + ratio*prior)
    ax5.set_title("Ours with prior")
    ax5.axis("off")

    plt.tight_layout()

    plt.savefig("comparison.pdf")

    def roc_auc(id, n_points=20, include_prior=True):
        """
        Calculates the Reciever-Operating-Characteristic (ROC) area under
        the curve (AUC) by numerical integration.
        """
        target = Image.open("./CAT2000_train/FIXATIONMAPS/OutdoorNatural/{0:03d}.jpg".format(id))
        target.thumbnail((512, 512))
        target = np.array(target)/255
        target = target[5:-5, 5:-5]

        img = Image.open("./CAT2000_train/Stimuli/OutdoorNatural/{0:03d}.jpg".format(id))
        img.thumbnail((512, 512))
        img = np.array(img)/255

        saliency = compute_saliency(img, filters, sigma=sigma, theta=theta)

        if include_prior:
            ratio = -(saliency.max() / prior.max())/2
            generated = saliency + ratio*prior
        else:
            generated = saliency

        # min max normalisation
        generated = (generated - generated.min())/(generated.max() - generated.min())

        def roc(p=0.1):
            x = generated.reshape(-1) > p
            t = target.reshape(-1) > p

            return np.sum(x==t)/len(t)

        calculate_roc = np.vectorize(roc)

        x = np.linspace(0, 1, n_points)
        auc = simps(calculate_roc(x))/n_points

        return auc

    print("4.1: Calculating ROC AUCs (this might take a while).")
    pool = Pool(cpu_count()*2)
    auc_score = partial(roc_auc, include_prior=include_prior)
    aucs = pool.map(auc_score, [2*i+1 for i in range(100)])

    #np.save("auc-prior-{}".format(include_prior), aucs)
    np.save("auc-prior-{}".format(include_prior), aucs)
