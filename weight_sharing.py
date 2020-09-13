import torch
import numpy as np
from sklearn.cluster import KMeans
from scipy.sparse import csc_matrix, csr_matrix


def applyWeightSharing(model, bits=4):
    """
        Applyig Weight Sharing to the Model, uses KMeans Clustering
    """


    for module in model.children():

        device = module.weight.device
        weight = module.weight.data.cpu().numpy()

        shape = weight.shape
        mat = csr_matrix(weight) if shape[0] < shape[1] else csc_matrix(weight)
        min_ = min(mat.data)
        max_ = max(mat.data)

        # Intital Centres
        space = np.linspace(min_, max_, num=2**bits)

        # Applying KMeans Clustering on Weight
        kmeans = KMeans(n_clusters=len(space), init=space.reshape(-1, 1), n_init=1, precompute_distances=True, algorithm="full")
        kmeans.fit(mat.data.reshape(-1, -1))

        newWeight = kmeans.cluster_centers_[kmeans.labels_].reshape(-1)
        mat.data = newWeight
        module.weight.data = torch.from_numpy(mat.toarray()).to(device)
    



