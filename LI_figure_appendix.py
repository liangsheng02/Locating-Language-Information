import random
import os
import numpy as np
import argparse
import pickle
import itertools
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from densray import *
import warnings
warnings.filterwarnings('ignore')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def go(dim=10, cross=False):
    #first x dims
    model = LogisticRegression(random_state=0)
    model.fit(x_train[:,:dim], y_train)
    a = model.score(x_test[:,:dim], y_test)

    #[x:2x]
    model.fit(x_train[:,dim:2*dim], y_train)
    b = model.score(x_test[:,dim:2*dim], y_test)

    #randomly choose x dims from [x:]
    if cross:
        idx = random.sample(range(dim,768-dim), 5)
        score = 0
        for i in range(5):
            model.fit(x_train[:,idx[i]:idx[i]+dim], y_train)
            score += model.score(x_test[:,idx[i]:idx[i]+dim], y_test)
        c = score/5
    else:
        idx = random.sample(range(dim, 768 - dim), 1)
        score = 0
        model.fit(x_train[:, idx[0]:idx[0] + dim], y_train)
        score += model.score(x_test[:, idx[0]:idx[0] + dim], y_test)
        c = score
    return a,b,c


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', default=0, type=int)
    args = parser.parse_args()

    all_langs = 'af, sq, ar, an, hy, ast, az, ba, eu, bar, be, bn, bpy, bs, br, bg, my, ca, ceb, ce, zh, zh_classical, ' \
            'cv, hr, cs, da, nl, en, et, fi, fr, gl, ka, de, el, gu, ht, he, hi, hu, is, io, id, ga, it, ja, jv, kn, ' \
            'kk, ky, ko, la, lv, lt, lmo, nds, lb, mk, mg, ms, ml, mr, min, ne, new, no, nn, oc, fa, pms, pl, pt, ' \
            'pa, ro, ru, sco, sr, sh, scn, sk, sl, azb, es, su, sw, sv, tl, tg, ta, tt, te, tr, uk, ur, uz, vi, vo, ' \
            'war, cy, fy, pnb, yo, th, mn'.split(', ')
    Austronesian = 'ceb,id,jv,mg,ms,min,su,tl,vi,war'.split(',')
    Italic = 'an,ca,fr,gl,ht,it,la,pms,scn,es'.split(',')
    Germanic = 'af,bar,nl,en,de,is,nds,lb,sco,fy'.split(',')
    cls_model = LogisticRegression(random_state=args.seed)

    random.seed(args.seed)
    # Q for all langs
    idx = random.sample(range(12, 768 - 2), 5)
    for family in [Austronesian,Italic,Germanic,random.sample(all_langs, 10)]:
        embs = [torch.load('/mounts/work/language_subspace/mwiki_emb_2/token/12/' + i + '.pt')[:10000] for i in family]
        dsr = DensRay(embs)
        dsr.fit()
        #print("Q")

        # CLS pairwise
        dims = list(range(0, 15+1, 1))
        acc_a, acc_b, acc_c = np.empty((0, len(dims))), np.empty((0, len(dims))), np.empty((0, len(dims)))
        for pair in list(itertools.combinations(family, 2)):
            # X
            emb = torch.Tensor(()).cpu()
            for i in pair:
                e = torch.load('/mounts/work/language_subspace/mwiki_emb_2/token/12/' + i + '.pt')[-10000:]
                eid = random.sample(list(range(len(e))), 10000)
                emb = torch.cat((emb, e[eid]))
            emb = torch.mm(emb, dsr.eigvecs)
            emb = emb.cpu().detach().numpy()
            #print("X")
            # Y
            y = []
            for i in range(2):
                y.extend([i] * 10000)
            y = np.array(y)
            # split
            x_train, x_test, y_train, y_test = train_test_split(emb, y, random_state=0, train_size=0.8)
            # train
            a, b, c = np.array([]), np.array([]), np.array([])
            #print("Y")
            for dim in dims:
                cls_model.fit(x_train[:, dim:dim+2], y_train)
                aa = cls_model.score(x_test[:, dim:dim+2], y_test)
                a = np.concatenate((a, [aa]))
            # random baseline
            score = 0
            for i in range(5):
                cls_model.fit(x_train[:, idx[i]:(idx[i] + 2)], y_train)
                score += cls_model.score(x_test[:, idx[i]:(idx[i] + 2)], y_test)
            cc = score / 5
            # pairwise summary: diff
            acc_a = np.vstack((acc_a, a))
    #summary = [(acc.mean(axis=0),acc.std(axis=0)) for acc in [acc_a,acc_b,acc_c]]
        print([round(x-cc, 4) for x in acc_a.mean(axis=0)], sep="=")

