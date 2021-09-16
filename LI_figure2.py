import random
import os
import numpy as np
import argparse
import pickle
import itertools
import torch
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression
import joblib
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import train_test_split
from densray import *
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
    n_samples = 5000
    ids = random.sample(range(0, 10000), n_samples)
    cls_model = LinearSVC(random_state=args.seed) if args.svc else LogisticRegression(random_state=args.seed)
    emb_path = 'cc-100_emb_2' if args.xlmr else 'mwiki_emb_2'

    for layer in range(1,13):
        if not args.lda:
            densray_path = '/mounts/work/language_subspace/' + emb_path + '/' + args.mode + '/' + str(layer) + '/Q_' + str(n_langs) + '.pt'
            if not os.path.exists(densray_path):
                embs = [torch.load('/mounts/work/language_subspace/'+emb_path+'/'+args.mode+'/'+str(layer)+'/'+all_langs[i]+'.pt')[ids,:] for i in range(len(all_langs))]
                dsr = DensRay(embs)
                dsr.fit()
                Q = dsr.eigvecs
                torch.save(Q, densray_path, _use_new_zipfile_serialization=False)
                torch.save(dsr.eigvals, '/mounts/work/language_subspace/'+emb_path+'/' + args.mode + '/'+str(layer)+'/Eigvals_'+str(n_langs)+'.pt',
                           _use_new_zipfile_serialization=False)
            else:
                Q = torch.load(densray_path)
        else:
            lda_pth = '/mounts/work/language_subspace/' + emb_path + '/' + args.mode + '/' + str(layer) + '/lda_' + str(n_langs) + '.model'
            if not os.path.exists(lda_pth):
                embs = torch.tensor(())
                labels = []
                for i in range(len(all_langs)):
                    embs = torch.cat((embs, torch.load('/mounts/work/language_subspace/'+emb_path+'/'+args.mode+'/'+str(layer)+'/'+all_langs[i]+'.pt')[ids, :]))
                    labels.extend([i] * n_samples)
                lda = LinearDiscriminantAnalysis()
                lda.fit(embs.numpy(), labels)
                joblib.dump(lda, lda_pth)
            else:
                lda = joblib.load(lda_pth)

        # pairwise langid classification
        dims = []
        dims.extend(list(range(0, 200+1, 10))[1:])
        acc_a = np.empty((0, len(dims)))
        for pair in random.sample(list(itertools.combinations(all_langs, 2)), 10):
            # X
            emb = torch.Tensor(()).cpu()
            for i in pair:
                e = torch.load('/mounts/work/language_subspace/'+emb_path+'/' + args.mode + '/'+str(layer)+'/' + i + '.pt')[-10000:]
                eid = random.sample(list(range(len(e))), n_samples)
                emb = torch.cat((emb, e[eid]))
            if not args.lda:
                emb = torch.mm(emb, Q)
                emb = emb.cpu().detach().numpy()
            else:
                emb = emb.numpy()
                emb2 = lda.transform(emb)
                emb = np.hstack((emb2[:, :103], emb[:, :]))
            # Y
            y = []
            for i in range(2):
                y.extend([i] * n_samples)
            y = np.array(y)
            # split
            x_train, x_test, y_train, y_test = train_test_split(emb, y, random_state=args.seed, train_size=0.8)
            # train
            a = np.array([])
            # every 10 dims
            for dim in dims:
                cls_model.fit(x_train[:, (dim-10):dim], y_train)
                aa = cls_model.score(x_test[:, (dim-10):dim], y_test)
                a = np.concatenate((a, [aa]))
            # random baseline
            idx = random.sample(range(105, 768 - 10), 5)
            score = 0
            for i in range(5):
                cls_model.fit(x_train[:, idx[i]:(idx[i] + 10)], y_train)
                score += cls_model.score(x_test[:, idx[i]:(idx[i] + 10)], y_test)
            cc = score / 5
            # pairwise summary: diff
            acc_a = np.vstack((acc_a, a))
        print('layer_'+str(layer), [round(x-cc, 4) for x in acc_a.mean(axis=0)], sep="=")

