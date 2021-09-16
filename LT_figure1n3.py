import pandas as pd
import io
import requests
import random
import os, joblib
import numpy as np
import argparse
import pickle
import torch
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings('ignore')

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--mode', type=str, default='token')
    parser.add_argument('--lda', action="store_true", default=False)
    parser.add_argument('--xlmr', action="store_true", default=False)
    parser.add_argument('--svc', action="store_true", default=False)
    args = parser.parse_args()

    all_langs = 'af,am,ar,as,az,be,bg,bn,bn_rom,br,bs,ca,cs,cy,da,de,el,en,eo,es,et,eu,fa,fi,fr,fy,ga,gd,gl,gu,ha,he,' \
                'hi,hi_rom,hr,hu,hy,id,is,it,ja,jv,ka,kk,km,kn,ko,ku,ky,la,lo,lt,lv,mg,mk,ml,mn,mr,ms,my,my_zaw,ne,' \
                'nl,no,om,or,pa,pl,ps,pt,ro,ru,sa,sd,si,sk,sl,so,sq,sr,su,sv,sw,ta,ta_rom,te,te_rom,th,tl,tr,ug,uk,' \
                'ur,ur_rom,uz,vi,xh,yi,zh,zh_classical'.split(',') if args.xlmr \
        else 'af,sq,ar,an,hy,ast,az,ba,eu,bar,be,bn,bpy,bs,br,bg,my,ca,ceb,ce,zh,zh_classical,cv,hr,cs,da,nl,en,et,' \
             'fi,fr,gl,ka,de,el,gu,ht,he,hi,hu,is,io,id,ga,it,ja,jv,kn,kk,ky,ko,la,lv,lt,lmo,nds,lb,mk,mg,ms,ml,mr,' \
             'min,ne,new,no,nn,oc,fa,pms,pl,pt,pa,ro,ru,sco,sr,sh,scn,sk,sl,azb,es,su,sw,sv,tl,tg,ta,tt,te,tr,uk,ur,' \
             'uz,vi,vo,war,cy,fy,pnb,yo,th,mn'.split(',')
    random.seed(args.seed)
    n_langs = len(all_langs)
    n_samples = 1000
    ids = random.sample(range(0, 10000), n_samples)
    cls_model = LogisticRegression(random_state=args.seed)
    emb_path = 'cc-100_emb_2' if args.xlmr else 'mwiki_emb_2'
    nl = 100 if args.xlmr else 104
    n_layers = 12

    language = pd.read_csv('/mounts/work/language_subspace/language.csv')
    wiki2wals = pd.read_csv('/mounts/work/language_subspace/language_subspace/wiki2wals_xlm.csv') if args.xlmr \
        else pd.read_csv('/mounts/work/language_subspace/language_subspace/wiki2wals.csv')
    wiki2wals = wiki2wals[~wiki2wals['wals'].isnull()]
    language.columns = language.columns.values.tolist()[:10] \
                       + [i.split()[0] for i in language.columns.values.tolist()[10:]]
    for i in range(10, language.shape[1]):
        language.iloc[:, i] = language.iloc[:, i].str[0]
    language.iloc[:, 10:] = language.iloc[:, 10:].replace(np.nan, -1).astype(int)
    maxs = language.iloc[:, 10:].max().tolist()
    wals = pd.merge(wiki2wals, language, left_on='wals', right_on='wals_code').iloc[:, [0] + list(range(16, 208))]
    # for layer
    acc_layer = []
    for layer in range(1,13):
        # load projection matrix
        if not args.lda:
            densray_path = '/mounts/work/language_subspace/' + emb_path + '/' + args.mode + '/' + str(layer) + '/Q_'+str(nl)+'_new2.pt'
            Q = torch.load(densray_path)
        else:
            lda_pth = '/mounts/work/language_subspace/' + emb_path + '/' + args.mode + '/' + str(layer) + '/lda_'+str(nl)+'.model'
            lda = joblib.load(lda_pth)
        # load data
        acc_col = []
        for col in random.sample(list(range(1, wals.shape[1])), 5):#range(1, wals.shape[1]):#wals.shape[1]):  # 90, wals.shape[1]
            wal = wals[wals.iloc[:, col] > 0].iloc[:, [0, col]]
            langs = wal.iloc[:, 0].tolist()
            n_class = len(wal.iloc[:, 1].unique())
            if len(wal.iloc[:, 1].unique()) > 1:
                # X, Y
                emb_tr = torch.Tensor(()).cpu()
                y_tr = []
                emb_te = torch.Tensor(()).cpu()
                y_te = []
                for l in langs:
                    e = torch.load('/mounts/work/language_subspace/'+emb_path+'/' + args.mode + '/'+str(layer)+'/' + l + '.pt')[-n_samples:]
                    # tr
                    ee = random.sample(list(range(len(e))), n_samples)
                    eid = ee[:int(n_samples * 0.8)]
                    emb_tr = torch.cat((emb_tr, e[eid]))
                    label = wal[wal['wiki'] == l].iloc[:, 1].max()
                    y_tr.extend([label] * len(eid))

                    # te
                    eid = ee[int(n_samples * 0.8)+1:]
                    emb_te = torch.cat((emb_te, e[eid]))
                    y_te.extend([label] * len(eid))
                    # print(emb_te.shape,emb_tr.shape,len(y_te),len(y_tr))
                if not args.lda:
                    x_train = torch.mm(emb_tr, Q).numpy()
                    x_test = torch.mm(emb_te, Q).numpy()
                else:
                    x_train = emb_tr.numpy()
                    x_train = lda.transform(x_train)
                    x_test = emb_te.numpy()
                    x_test = lda.transform(x_test)
                y_train = np.array(y_tr)
                y_test = np.array(y_te)
                #a = go(x_train, y_train, x_test, y_test, dim=20, cls_model=cls_model)
                cls_model.fit(x_train[:, :20], y_train)
                acc_col.append(cls_model.score(x_test[:, :20], y_test))
        acc_layer.append(round(np.mean(acc_col),4))
    print(acc_layer)
