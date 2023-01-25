# python Kmean.py --config=configs/chair/cat.txt

import pdb
import numpy as np
import torch
import torch.utils.data as data
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

from parser import *
from often_use import *
from train_pl import *

torch.pi = torch.acos(torch.zeros(1)).item() * 2 # which is 3.1415927410125732

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
np.random.seed(0)
DEBUG = False





if __name__=='__main__':
    # Get args
    args = get_args()
    args.gpu_num = torch.cuda.device_count() # log used gpu num.

    # Get ddf.
    ddf = DDF(args)
    ddf = ddf.load_from_checkpoint(
        checkpoint_path='/home/yyoshitake/works/DeepSDF/project/DDF/lightning_logs/chair/cat_depth_mae_normal_mae_seed0_normal001_lr00001/checkpoints/0000010000.ckpt', 
        args=args)
    ddf.eval()
    
    lat_vecs = []
    for lat_id in range(ddf.lat_vecs.num_embeddings):
        lat_vecs.append(ddf.lat_vecs(torch.tensor([lat_id], dtype=torch.long).to(ddf.device)))
    lat_vecs = torch.cat(lat_vecs, dim=0)
    lat_vecs = lat_vecs.to('cpu').detach().numpy().copy()

    # #########################
    # instance_list = pickle_load('instance_list.pickle')
    # outliers_list = pickle_load('outliers_list.pickle')
    # refined_instance_mask = np.array([not ins in outliers_list for ins in instance_list])
    # lat_vecs = lat_vecs[refined_instance_mask]
    # instance_list = instance_list[refined_instance_mask]
    # #########################
    import pdb; pdb.set_trace()

    n_clusters = 5
    random_state = None
    pca = PCA()
    pca.fit(lat_vecs)
    feature = pca.transform(lat_vecs)

    # scaler = StandardScaler()
    # data = scaler.fit_transform(feature)[:, :2]
    # kmeans = KMeans(n_clusters=n_clusters, init='k-means++', n_init=10, max_iter=300,
    #                 tol=0.0001, verbose=0,
    #                 random_state=random_state, copy_x=True).fit(data)
    
    import pdb; pdb.set_trace()

    # fig = plt.figure()
    # cmap = plt.get_cmap("hsv")
    # point = feature
    # for label in range(n_clusters):
    #     cls = np.where(kmeans.labels_==label)
    #     plt.scatter(point[cls, 0], point[cls, 1], marker='o', color=cmap(float(label)/n_clusters), label='class_' + str(label))
    #     instance_list[cls]
    #     ########################################
    #     path = f'kmeans_list_{label}.txt'
    #     with open(path, 'a') as f:
    #         for instance_id in instance_list[cls]:
    #             f.write(instance_id+'\n')
    #     ########################################
    # plt.legend()
    # fig.savefig("img.png")
    # import pdb; pdb.set_trace()