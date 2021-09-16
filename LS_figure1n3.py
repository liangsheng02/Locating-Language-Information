import scipy.cluster.hierarchy as hac
from sklearn import metrics
import pandas as pd
import torch
import numpy as np
from matplotlib import pyplot as plt
from scipy.cluster.hierarchy import dendrogram, linkage
import numpy as np
import random
from scipy.cluster.hierarchy import cophenet
from scipy.spatial.distance import pdist
import joblib
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
import argparse, os


def get_hcv(emb, y, families):
    tree = linkage(emb, method='single', metric='cosine')
    clustered = hac.fcluster(tree, len(families), criterion='maxclust')
    homogenity = metrics.homogeneity_score(y, clustered)
    completeness = metrics.completeness_score(y, clustered)
    v_measure = metrics.v_measure_score(y, clustered)
    #c, coph_dists = cophenet(tree, pdist(emb))
    #return tree, homogenity, completeness,
    return v_measure


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--mode', type=str, default='token')
    parser.add_argument('--lda', action="store_true", default=False)
    parser.add_argument('--xlmr', action="store_true", default=False)
    args = parser.parse_args()
    # process
    f_word = 'genus'
    wiki2wals = pd.read_csv('/mounts/work/language_subspace/language_subspace/wiki2wals_xlm.csv') if args.xlmr \
        else pd.read_csv('/mounts/work/language_subspace/language_subspace/wiki2wals.csv')
    wiki2wals = wiki2wals[~wiki2wals[f_word].isnull()]
    gunus = wiki2wals.iloc[:, [0, 1, 2, 3, 4, 5]]
    s = gunus[f_word].value_counts()
    gunus = gunus[gunus.isin(s.index[s >= 2]).values]
    families = list(set(gunus[f_word]))
    # setup
    random.seed(args.seed)
    n_layers = 13
    n_samples = 5000
    emb_path = 'cc-100_emb_2' if args.xlmr else 'mwiki_emb_2'

    n_langs = 100 if args.xlmr else 104
    acc_layer=[]
    for layer in range(n_layers):
        # load projection matrix
        if not args.lda:
            densray_path = '/mounts/work/language_subspace/' + emb_path + '/' + args.mode + '/' + str(layer) + '/Q_' + str(
                n_langs) + '_new2.pt'
            Q = torch.load(densray_path)
        else:
            lda_pth = '/mounts/work/language_subspace/' + emb_path + '/' + args.mode + '/' + str(layer) + '/lda_' + str(
                n_langs) + '.model'
            lda = joblib.load(lda_pth)
        # load data
        emb, y = torch.Tensor(()), []
        for lang in gunus['wiki'].tolist():
            label = families.index(list(gunus[f_word][gunus['wiki'] == lang])[0])
            y.extend([label])
            e = torch.load('/mounts/work/language_subspace/'+emb_path+'/' + args.mode + '/'+str(layer)+'/' + lang + '.pt')[-10000:]
            eid = random.sample(list(range(len(e))), n_samples)
            emb = torch.cat((emb, e[eid].mean(dim=0).unsqueeze(0)))  # get language centroids
        if not args.lda:
            emb = torch.mm(emb, Q).numpy()
        else:
            emb = emb.numpy()
            emb = lda.transform(emb)
        y = np.array(y)
        acc_layer.append(get_hcv(emb[:, :10], y, families))
    print(acc_layer)
