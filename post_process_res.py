import torch
import glob
import os
from tqdm import tqdm


def post_process(folder):
    res_per_HP = {}
    # path = '../cross-mistake-learning/' + folder + '/stage_1_avg/'
    path = '../cross-mistake-learning/' + folder + '/stage_1_2/'
    os.makedirs(path, exist_ok=True)
    # for file in tqdm(glob.glob('../cross-mistake-learning/' + folder + '/stage_1/*.pt')):
    for file in tqdm(glob.glob('../cross-mistake-learning/' + folder + '/stage_1/*seed_2*.pt')):
        key = file.split('/')[-1]
        # for s in ['seed_0', 'seed_1', 'seed_2', 'seed_3', 'seed_4']:
        #     key = key.replace(s, 'seed_avg')
        if key in res_per_HP.keys():
            res_per_HP[key]['tr'] += [torch.load(file)['tr']['outs']]
            res_per_HP[key]['va'] += [torch.load(file)['va']['outs']]
        else:
            res_per_HP[key] = {
                'tr': [torch.load(file)['tr']['outs']],
                'va': [torch.load(file)['va']['outs']]}

    ys_tr = torch.load(file)['tr']['ys']
    ys_va = torch.load(file)['va']['ys']

    for key in res_per_HP.keys():
        outs_tr = sum(res_per_HP[key]['tr']) / len(res_per_HP[key]['tr'])
        outs_va = sum(res_per_HP[key]['va']) / len(res_per_HP[key]['va'])
        to_save = {'tr': {}, 'va': {}}
        to_save['tr']['outs'] = outs_tr
        to_save['tr']['ys'] = ys_tr
        to_save['tr']['m_hat'] = outs_tr.argmax(1, keepdim=True).eq(ys_tr).long()
        to_save['va']['outs'] = outs_va
        to_save['va']['ys'] = ys_va
        to_save['va']['m_hat'] = outs_va.argmax(1, keepdim=True).eq(ys_va).long()

        print(key)
        torch.save(to_save, path + key)

# post_process('res23')
# post_process('res24')
# post_process('res25')

post_process('res27')
post_process('res28')
post_process('res29')
