import os
import numpy as np
import torch
import json
import glob
from subpopbench import hparams_registry
from subpopbench.utils import misc
from matplotlib import pyplot as plt
import re
from torchmetrics import CalibrationError
from subpopbench.utils import eval_helper
from tqdm import tqdm


def best_te_acc(path):
    va_worst_accs = []
    te_worst_accs = []
    with open(path, "r") as file:
        for line in file:
            res = json.loads(line)
            va_worst_accs += [res['va_worst_acc']]
            te_worst_accs += [res['te_worst_acc']]
    return te_worst_accs[np.argmax(va_worst_accs)]


def plot_perf(input_dir, te_accs, crs, cr_names, sorted=False):

    fig, ax = plt.subplots(1 + crs.shape[1], 1, figsize=(8, 5 * (1 + crs.shape[1])))
    ax[0].axhline(y=te_accs[-1], alpha=0.4, c='tab:blue', ls='--')
    if sorted:
        inds = np.argsort(te_accs[:-1])[::-1]
        ax[0].plot(np.array(te_accs[:-1])[inds])
    else:
        ax[0].plot(te_accs[:-1])
        ax[0].axvline(x=np.argmax(te_accs[:-1]), alpha=0.4, c='tab:red', ls='--')
    for i in np.arange(1, 1 + crs.shape[1]):
        if sorted:
            ax[i].plot(crs[:, i - 1][inds], label=cr_names[i - 1])
        else:
            ax[i].plot(crs[:, i - 1], label=cr_names[i - 1])
            ax[i].axvline(x=np.argmax(te_accs[:-1]), alpha=0.4, c='tab:red', ls='--')
        ax[i].grid(ls='--')
        ax[i].legend()
        if 'nll' in cr_names[i - 1]:
            ax[i].set_ylim([0.0, 2.0])
    ax[0].grid(ls='--')
    plt.tight_layout()
    plt.savefig('plots/' + input_dir + '.png')


def plot_margin(input_dir, best_model, pf=''):
    dataset = best_model.split('INFERRED_')[-1].split('_')[0]
    real = torch.load(os.path.join('/checkpoint/dlp/cross_mistakes_features/', dataset + '.pt'))
    infr = torch.load(json.load(open(best_model.replace('results', 'args')))['train_attr'])
    fig, ax = plt.subplots(3, 2, figsize=(20, 24))
    y = real['tr']['y'][:, 0]
    m_orig = real['tr']['m'][:, 0]
    m_soft = (infr['tr']['o_s'][:, 1] -
              infr['tr']['o_s'][:, 0]) * (2 * real['tr']['y'][:, 0] - 1)
    # import ipdb; ipdb.set_trace()
    ym = len(m_orig.unique()) * y + m_orig
    for j, yi in enumerate(y.unique()):
        for mj in m_orig.unique():
            ymi = len(m_orig.unique()) * yi + mj
            ax[j, 0].hist(
                m_soft.view(-1)[ym.view(-1) == ymi].cpu().numpy(),
                alpha=0.5,
                bins=20,
                label="group (y={}, m={}, n={})".format(
                    int(ymi.item() // len(m_orig.unique())),
                    int(ymi % len(m_orig.unique())),
                    int((ym == ymi).sum())))
            ax[j, 0].set_yscale("log")
            ax[j, 0].axvline(0, color="gray", ls=":", lw=2)
            ax[j, 0].legend()
    plt.tight_layout()
    plt.savefig('plots/' + input_dir + "_margin" + str(pf) + ".png")
    plt.close("all")


def stage_1_res(stage_2_path):
    return torch.load(json.load(open(
        stage_2_path.replace('results', 'args')))['train_attr'])


def get_cr(input_dir, path):
    res = stage_1_res(path)
    nll_va = torch.nn.functional.cross_entropy(
        res['va']['o_s'].float(), res['va']['y_s'][:, 0]).item()
    if np.isnan(nll_va):
        nll_va = -1

    # nll_tr = torch.nn.functional.cross_entropy(
    #     res['tr']['o_s'], res['tr']['y_s'][:, 0]).item()
    # if np.isnan(nll_tr):
    #     nll_tr = -1

    acc_va = res['va']['o_s'].argmax(1).eq(res['va']['y_s'][:, 0]).float().mean().item()
    # acc_tr = res['tr']['o_s'].argmax(1).eq(res['tr']['y_s'][:, 0]).float().mean().item()

    calibration_error = CalibrationError(
        num_bins=20, task='multiclass', num_classes=res['va']['o_s'].size(1), norm='max')
    if not np.isnan(res['va']['o_s'].sum().numpy()):
        va_ce = calibration_error(
            torch.softmax(res['va']['o_s'].float(), dim=1),
            res['va']['y_s'][:, 0]).item()
    else:
        va_ce = -1

    # top_10 = torch.nn.functional.cross_entropy(
    #     res['va']['o_s'], res['va']['y_s'][:, 0],
    #     reduction='none').sort().values[-int(0.01 * res['va']['y_s'].size(0)):].mean().item()

    per_cls_acc = []
    for c in range(res['va']['o_s'].size(1)):
        per_cls_acc += [res['va']['o_s'].argmax(1)[res['va']['y_s'][:, 0] == c].eq(c).float().mean()]
    # std_va = np.std(per_cls_acc)
    # dist_va = np.abs(per_cls_acc[0] - per_cls_acc[1])
    sum_cls = np.sum(per_cls_acc)

    mets = eval_helper.binary_metrics(res['va']['y_s'][:, 0], res['va']['o_s'].argmax(1),
                                      label_set=res['va']['y_s'][:, 0].unique().tolist())
    cr0 = mets['accuracy'] if 'accuracy' in mets.keys() else -1
    cr1 = mets['TPR'] if 'TPR' in mets.keys() else -1
    cr2 = mets['FNR'] if 'FNR' in mets.keys() else -1
    cr3 = mets['FPR'] if 'FPR' in mets.keys() else -1
    cr4 = mets['TNR'] if 'TNR' in mets.keys() else -1
    cr5 = mets['pred_prevalence'] if 'pred_prevalence' in mets.keys() else -1
    cr6 = mets['prevalence'] if 'prevalence' in mets.keys() else -1
    cr7 = mets['balanced_acc'] if 'balanced_acc' in mets.keys() else -1
    label_set = res['va']['y_s'][:, 0].unique().tolist()
    if len(label_set) > 2:
        p = res['va']['o_s'].float().softmax(1).numpy()
    else:
        p = res['va']['o_s']
        p = torch.sigmoid((p[:, 1] - p[:, 0])).numpy()
    try:
        mets = eval_helper.prob_metrics(res['va']['y_s'][:, 0].numpy(), p,
                                        label_set=label_set)
    except:
        mets = {}
    cr8 = mets['AUROC_ovo'] if 'AUROC_ovo' in mets.keys() else -1
    cr9 = mets['BCE'] if 'BCE' in mets.keys() else -1
    cr10 = mets['ECE'] if 'ECE' in mets.keys() else -1
    cr11 = mets['AUROC'] if 'AUROC' in mets.keys() else -1
    cr12 = mets['AUPRC'] if 'AUPRC' in mets.keys() else -1
    cr13 = mets['brier'] if 'brier' in mets.keys() else -1

    nll_mistakes = torch.nn.functional.cross_entropy(
        res['va']['o_s'].float(),
        res['va']['y_s'][:, 0], reduction='none')[
            torch.logical_not(res['va']['o_s'].argmax(1).eq(
                res['va']['y_s'][:, 0]))].mean().item()

    crs = np.array([nll_va, cr0, va_ce, sum_cls, nll_mistakes,
                    cr1, cr2, cr3, cr4, cr5, cr6, cr7, cr8,
                    cr9, cr10, cr11, cr12, cr13])
    names = ['nll_va', 'acc_va', 'ce_va', 'sum_cls_acc', 'nll_mistakes',
             'TPR', 'FNR', 'FPR', 'TNR', 'pred_prevalence', 'prevalence',
             'balanced_acc', 'AUROC_ovo', 'BCE', 'ECE', 'AUROC',
             'AUPRC', 'brier']
    return crs, names


def all_paths_sorted(match):
    paths = glob.glob(match)
    lr = []
    wd = []
    bs = []
    seed = []
    epoch = []
    for path in paths:
        if '_lr_' in path:
            lr += [float(path.split('_lr_')[1].split('_wd')[0])]
            wd += [float(path.split('_wd_')[1].split('_bs')[0])]
            bs += [int(path.split('_bs_')[1].split('_seed')[0])]
            seed += [int(path.split('_seed_')[1].split('_epoch_')[0])]
            epoch += [int(path.split('_epoch_')[1].split('/')[0])]
            sample_path = (path.split('_lr_')[0] +
                           f'_lr_llll_wd_wwww_bs_bbbb_seed_ssss_epoch_eeee/' +
                           '/'.join(path.split('_epoch_')[1].split('/')[1:]))
        elif 'attrYes' in path:
            oracle_path = path + ''
        else:
            pass
    lr = sorted(set(lr))
    wd = sorted(set(wd))
    bs = sorted(set(bs))
    seed = sorted(set(seed))
    epoch = sorted(set(epoch))

    paths_ = []
    for lr_ in lr:
        for wd_ in wd:
            for bs_ in bs:
                for seed_ in seed:
                    for epoch_ in epoch:
                        path_ = sample_path + ''
                        path_ = path_.replace('llll', str(lr_))
                        path_ = path_.replace('wwww', str(wd_))
                        path_ = path_.replace('bbbb', str(bs_))
                        path_ = path_.replace('ssss', str(seed_))
                        path_ = path_.replace('eeee', str(epoch_))
                        if path_ in paths:
                            paths_ += [path_]
    return paths_, oracle_path


def show_all(folder, input_dir, method):
    match = f'./{folder}/{input_dir}*/*{method}*/results.json'
    paths, oracle_path = all_paths_sorted(match)
    print('number of jobs:', len(paths))
    te_accs = []
    crs = []
    lin = np.load('../cross-mistake-learning/lin.npy')
    # paths = np.array(paths)[np.argsort(lin)[::-1][:5]]           
    for pf, path in enumerate(tqdm(paths)):
        acc = best_te_acc(path)
        te_accs += [acc]
        # te_accs += [lin[pf]]     
        cr, cr_names = get_cr(input_dir, path)
        crs += [cr]
        # plot_margin(input_dir, path, pf)       

    # import ipdb; ipdb.set_trace()

    crs = np.nan_to_num(np.array(crs), nan=-1)
    best_model = paths[np.argmax(te_accs)]
    print(best_model)
    plot_margin(input_dir, best_model)
    te_accs += [best_te_acc(oracle_path)]
    plot_perf(input_dir, te_accs, crs, cr_names, sorted=True)

show_all(folder='output_m', input_dir='WB_L1_2fold', method='GroupDRO')
# show_all(folder='output_r', input_dir='WB_MSE_2fold', method='GroupDRO')
# show_all(folder='output_n', input_dir='WB_ERM_2fold', method='GroupDRO')
# show_all(folder='output_p', input_dir='WB_L1_5fold', method='GroupDRO')

# show_all(folder='output_m', input_dir='CA_L1_2fold', method='GroupDRO')
# show_all(folder='output_r', input_dir='CA_MSE_2fold', method='GroupDRO')
# show_all(folder='output_n', input_dir='CA_ERM_2fold', method='GroupDRO')
# show_all(folder='output_p', input_dir='CA_L1_5fold', method='GroupDRO')

# show_all(folder='output_m', input_dir='MNLI_L1_2fold', method='GroupDRO')
# show_all(folder='output_r', input_dir='MNLI_MSE_2fold', method='GroupDRO')
# show_all(folder='output_n', input_dir='MNLI_ERM_2fold', method='GroupDRO')
# show_all(folder='output_p', input_dir='MNLI_L1_5fold', method='GroupDRO')
