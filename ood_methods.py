import copy

import numpy as np
import torch
from numpy import linalg as LA
from numpy.linalg import norm
from scipy.special import logsumexp
from sklearn.covariance import EmpiricalCovariance
from sklearn.decomposition import PCA
from sklearn.metrics import pairwise_distances_argmin_min
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm
import utils_ood as utils
from sklearn.decomposition import TruncatedSVD

recall = 0.95


def get_energy_score(logits):
    scores = torch.logsumexp(logits.data.cpu(), dim=1).numpy()
    return scores


def ash_b_base(abc, percentile=65):
    assert abc.dim() == 4
    assert 0 <= percentile <= 100
    x = copy.deepcopy(abc)
    b, c, h, w = x.shape

    # calculate the sum of the input per sample
    s1 = x.sum(dim=[1, 2, 3])

    n = x.shape[1:].numel()
    k = n - int(np.round(n * percentile / 100.0))
    t = x.view((b, c * h * w))
    v, i = torch.topk(t, k, dim=1)
    fill = s1 / k
    fill = fill.unsqueeze(dim=1).expand(v.shape)
    t.zero_().scatter_(dim=1, index=i, src=fill)
    return x


def ash_p_base(abc, percentile=65):
    assert abc.dim() == 4
    assert 0 <= percentile <= 100
    x = copy.deepcopy(abc)

    b, c, h, w = x.shape

    n = x.shape[1:].numel()
    k = n - int(np.round(n * percentile / 100.0))
    t = x.view((b, c * h * w))
    v, i = torch.topk(t, k, dim=1)
    t.zero_().scatter_(dim=1, index=i, src=v)

    return x


def ash_s_base(abc, percentile=65):
    assert abc.dim() == 4
    assert 0 <= percentile <= 100
    b, c, h, w = abc.shape
    x = copy.deepcopy(abc)
    # calculate the sum of the input per sample
    s1 = x.sum(dim=[1, 2, 3])
    n = x.shape[1:].numel()
    k = n - int(np.round(n * percentile / 100.0))
    t = x.view((b, c * h * w))
    v, i = torch.topk(t, k, dim=1)
    t.zero_().scatter_(dim=1, index=i, src=v)

    # calculate new sum of the input per sample after pruning
    s2 = x.sum(dim=[1, 2, 3])

    # apply sharpening
    scale = s1 / s2
    x = x * torch.exp(scale[:, None, None, None])

    return x


def ash_b(feature_id_val_ash, feature_ood_ash, thresh, weight, bias, name):
    method = 'ASH_B'
    w, b = weight, bias
    score_id = ash_b_base(feature_id_val_ash, percentile=thresh)
    score_ood = ash_b_base(feature_ood_ash, percentile=thresh)

    score_ood = np.squeeze(score_ood)
    score_id = np.squeeze(score_id)
    score_id = score_id @ w.T + b
    score_ood = score_ood @ w.T + b
    score_id = get_energy_score(score_id)
    score_ood = get_energy_score(score_ood)
    score_id = score_id.tolist()
    score_ood = score_ood.tolist()
    print(f'\n{method}')
    auc_ood = utils.auc(score_id, score_ood)[0]
    fpr_ood, _ = utils.fpr_recall(score_id, score_ood, recall)
    print(f'{method}: {name} auroc {auc_ood:.2%}, fpr {fpr_ood:.2%}')


def ash_p(feature_id_val_ash, feature_ood_ash, thresh, weight, bias, name):
    method = 'ASH_P'
    w, b = weight, bias
    score_id = ash_p_base(feature_id_val_ash, percentile=thresh)
    score_ood = ash_p_base(feature_ood_ash, percentile=thresh)
    score_ood = np.squeeze(score_ood)
    score_id = np.squeeze(score_id)
    score_id = score_id @ w.T + b
    score_ood = score_ood @ w.T + b
    score_id = get_energy_score(score_id)
    score_ood = get_energy_score(score_ood)
    score_id = score_id.tolist()
    score_ood = score_ood.tolist()

    print(f'\n{method}')
    auc_ood = utils.auc(score_id, score_ood)[0]
    fpr_ood, _ = utils.fpr_recall(score_id, score_ood, recall)
    print(f'{method}: {name} auroc {auc_ood:.2%}, fpr {fpr_ood:.2%}')


def ash_s(feature_id_val_ash, feature_ood_ash, thresh, weight, bias, name):
    method = 'ASH_S'
    w, b = weight, bias
    score_id = ash_s_base(feature_id_val_ash, percentile=thresh)
    score_ood = ash_s_base(feature_ood_ash, percentile=thresh)
    score_ood = np.squeeze(score_ood)
    score_id = np.squeeze(score_id)
    score_id = score_id @ w.T + b
    score_ood = score_ood @ w.T + b
    score_id = get_energy_score(score_id)
    score_ood = get_energy_score(score_ood)
    score_id = score_id.tolist()
    score_ood = score_ood.tolist()

    print(f'\n{method}')
    auc_ood = utils.auc(score_id, score_ood)[0]
    fpr_ood, _ = utils.fpr_recall(score_id, score_ood, recall)
    print(f'{method}: {name} auroc {auc_ood:.2%}, fpr {fpr_ood:.2%}')


def msp(softmax_id_val, softmax_ood, ood_name):
    method = 'MSP'
    score_id = softmax_id_val.max(axis=-1)
    score_ood = softmax_ood.max(axis=-1)
    auc_ood = utils.auc(score_id, score_ood)[0]
    fpr_ood, _ = utils.fpr_recall(score_id, score_ood, recall)
    print(f'{method}: {ood_name} auroc {auc_ood:.2%}, fpr {fpr_ood:.2%}')


def maxLogit(logit_id_val, logit_ood, ood_name):
    method = 'Max_Logit'
    score_id = logit_id_val.max(axis=-1)
    score_ood = logit_ood.max(axis=-1)
    auc_ood = utils.auc(score_id, score_ood)[0]
    fpr_ood, _ = utils.fpr_recall(score_id, score_ood, recall)
    print(f'{method}: {ood_name} auroc {auc_ood:.2%}, fpr {fpr_ood:.2%}')


def energy(logit_id_val, logit_ood, ood_name):
    method = 'Energy'
    score_id = logsumexp(logit_id_val, axis=-1)
    score_ood = logsumexp(logit_ood, axis=-1)
    auc_ood = utils.auc(score_id, score_ood)[0]
    fpr_ood, _ = utils.fpr_recall(score_id, score_ood, recall)
    print(f'{method}: {ood_name} auroc {auc_ood:.2%}, fpr {fpr_ood:.2%}')


def react(feature_id_val, feature_ood, clip, w, b, name):
    method = 'ReAct'
    logit_id_val_clip = np.clip(
        feature_id_val, a_min=None, a_max=clip) @ w.T + b
    score_id = logsumexp(logit_id_val_clip, axis=-1)
    logit_ood_clip = np.clip(feature_ood, a_min=None, a_max=clip) @ w.T + b
    score_ood = logsumexp(logit_ood_clip, axis=-1)
    auc_ood = utils.auc(score_id, score_ood)[0]
    fpr_ood, _ = utils.fpr_recall(score_id, score_ood, recall)
    print(f'{method}: {name} auroc {auc_ood:.2%}, fpr {fpr_ood:.2%}')


def vim(feature_id_train, feature_id_val, feature_ood, logit_id_train, logit_id_val, logit_ood, name, model_architecture_type, model_name, u):
    method = 'ViM'

    DIM = 1000 if feature_id_val.shape[-1] >= 2048 else 512
    if model_architecture_type == "resnet" and (model_name == "resnet34" or model_name == 'resnet18'):
        DIM = 300
    print(f'{DIM=}')
    ec = EmpiricalCovariance(assume_centered=True)
    ec.fit(feature_id_train - u)
    eig_vals, eigen_vectors = np.linalg.eig(ec.covariance_)
    NS = np.ascontiguousarray(
        (eigen_vectors.T[np.argsort(eig_vals * -1)[DIM:]]).T)
    print('computing alpha...')
    vlogit_id_train = norm(np.matmul(feature_id_train - u, NS), axis=-1)
    alpha = logit_id_train.max(axis=-1).mean() / vlogit_id_train.mean()
    print(f'{alpha=:.4f}')
    vlogit_id_val = norm(np.matmul(feature_id_val - u, NS), axis=-1) * alpha
    energy_id_val = logsumexp(logit_id_val, axis=-1)
    score_id = -vlogit_id_val + energy_id_val
    energy_ood = logsumexp(logit_ood, axis=-1)
    vlogit_ood = norm(np.matmul(feature_ood - u, NS), axis=-1) * alpha
    score_ood = -vlogit_ood + energy_ood
    auc_ood = utils.auc(score_id, score_ood)[0]
    fpr_ood, _ = utils.fpr_recall(score_id, score_ood, recall)
    print(f'{method}: {name} auroc {auc_ood:.2%}, fpr {fpr_ood:.2%}')


def residual(feature_id_train, feature_id_val, feature_ood, model_architecture_type, u, name):
    method = 'Residual'
    DIM = 1000 if feature_id_val.shape[-1] >= 2048 else 512
    if model_architecture_type == "resnet":
        DIM = 300
    ec = EmpiricalCovariance(assume_centered=True)
    ec.fit(feature_id_train - u)
    eig_vals, eigen_vectors = np.linalg.eig(ec.covariance_)
    NS = np.ascontiguousarray(
        (eigen_vectors.T[np.argsort(eig_vals * -1)[DIM:]]).T)
    score_id = -norm(np.matmul(feature_id_val - u, NS), axis=-1)
    score_ood = -norm(np.matmul(feature_ood - u, NS), axis=-1)
    auc_ood = utils.auc(score_id, score_ood)[0]
    fpr_ood, _ = utils.fpr_recall(score_id, score_ood, recall)
    print(f'{method}: {name} auroc {auc_ood:.2%}, fpr {fpr_ood:.2%}')


def neco(feature_id_train, feature_id_val, feature_ood, logit_id_val, logit_ood, model_architecture_type, neco_dim):
    '''
    Prints the auc/fpr result for NECO method.

            Parameters:
                    feature_id_train (array): An array of training samples features
                    feature_id_val (array): An array of evaluation samples features
                    feature_ood (array): An array of OOD samples features
                    logit_id_val (array): An array of evaluation samples logits
                    logit_ood (array): An array of OOD samples logits
                    model_architecture_type (string): Module architecture used
                    neco_dim (int): ETF approximative dimmention for the tested case

            Returns:
                    None
    '''
    method = 'NECO'
    ss = StandardScaler()  # if NC1 is well verified, i.e a well seperated class clusters (case of cifar using ViT, its better not to use the scaler)
    complete_vectors_train = ss.fit_transform(feature_id_train)
    complete_vectors_test = ss.transform(feature_id_val)
    complete_vectors_ood = ss.transform(feature_ood)

    pca_estimator = PCA(feature_id_train.shape[1])
    _ = pca_estimator.fit_transform(complete_vectors_train)
    cls_test_reduced_all = pca_estimator.transform(complete_vectors_test)
    cls_ood_reduced_all = pca_estimator.transform(complete_vectors_ood)

    score_id_maxlogit = logit_id_val.max(axis=-1)
    score_ood_maxlogit = logit_ood.max(axis=-1)
    if model_architecture_type in ['deit', 'swin']:
        complete_vectors_train = feature_id_train
        complete_vectors_test = feature_id_val
        complete_vectors_ood = feature_ood

    cls_test_reduced = cls_test_reduced_all[:, :neco_dim]
    cls_ood_reduced = cls_ood_reduced_all[:, :neco_dim]
    l_ID = []
    l_OOD = []

    for i in range(cls_test_reduced.shape[0]):
        sc_complet = LA.norm((complete_vectors_test[i, :]))
        sc = LA.norm(cls_test_reduced[i, :])
        sc_finale = sc/sc_complet
        l_ID.append(sc_finale)
    for i in range(cls_ood_reduced.shape[0]):
        sc_complet = LA.norm((complete_vectors_ood[i, :]))
        sc = LA.norm(cls_ood_reduced[i, :])
        sc_finale = sc/sc_complet
        l_OOD.append(sc_finale)
    l_OOD = np.array(l_OOD)
    l_ID = np.array(l_ID)
    #############################################################
    score_id = l_ID
    score_ood = l_OOD
    if model_architecture_type != 'resnet':
        score_id *= score_id_maxlogit
        score_ood *= score_ood_maxlogit
        auc_ood = utils.auc(score_id, score_ood)[0]
    auc_ood = utils.auc(score_id, score_ood)[0]
    recall = 0.95
    fpr_ood, _ = utils.fpr_recall(score_id, score_ood, recall)
    print(f' \n {method}: auroc {auc_ood:.2%}, fpr {fpr_ood:.2%}')


def gradNorm(feature_id_val, feature_ood, name, num_classes, w, b):
    method = 'GradNorm'
    score_id = gradnorm(feature_id_val, w, b, num_classes=num_classes)
    score_ood = gradnorm(feature_ood, w, b, num_classes=num_classes)
    auc_ood = utils.auc(score_id, score_ood)[0]
    fpr_ood, _ = utils.fpr_recall(score_id, score_ood, recall)
    result = dict(method=method, oodset=name, auroc=auc_ood, fpr=fpr_ood)

    print(f'{method}:  auroc {auc_ood:.2%}, fpr {fpr_ood:.2%}')  # {name}

def kl(p, q):
    return np.sum(np.where(p != 0, p * np.log(p / q), 0))

def kl_matching(softmax_id_train, softmax_id_val, softmax_ood, name, num_classes):
    method = 'KL-Matching'
    print(f'\n{method}')

    print('computing classwise mean softmax...')
    pred_labels_train = np.argmax(softmax_id_train, axis=-1)
    mean_softmax_train = [softmax_id_train[pred_labels_train == i].mean(
        axis=0) for i in tqdm(range(num_classes))]
    score_id = -pairwise_distances_argmin_min(
        softmax_id_val, np.array(mean_softmax_train), metric=kl)[1]

    score_ood = -pairwise_distances_argmin_min(
        softmax_ood, np.array(mean_softmax_train), metric=kl)[1]
    auc_ood = utils.auc(score_id, score_ood)[0]
    fpr_ood, _ = utils.fpr_recall(score_id, score_ood, recall)
    result = dict(method=method, oodset=name, auroc=auc_ood, fpr=fpr_ood)
    print(f'{method}: {name} auroc {auc_ood:.2%}, fpr {fpr_ood:.2%}')


def mahalanobis(feature_id_train, train_labels, feature_id_val, feature_ood, name, num_classes):
    method = 'Mahalanobis'
    print('computing classwise mean feature...')
    train_means = []
    train_feat_centered = []
    for i in tqdm(range(num_classes)):
        fs = feature_id_train[train_labels == i]
        _m = fs.mean(axis=0)
        train_means.append(_m)
        train_feat_centered.extend(fs - _m)
    print(f" len of train_feat_centered {len(train_feat_centered)}")
    print('computing precision matrix...')
    ec = EmpiricalCovariance(assume_centered=True)
    ec.fit(np.array(train_feat_centered).astype(np.float64))
    print('go to gpu...')
    mean = torch.from_numpy(np.array(train_means)).cuda().float()
    prec = torch.from_numpy(ec.precision_).cuda().float()
    score_id = -np.array([(((f - mean)@prec) * (f - mean)).sum(axis=-1).min().cpu().item()
                         for f in tqdm(torch.from_numpy(feature_id_val).cuda().float())])
    score_ood = -np.array([(((f - mean)@prec) * (f - mean)).sum(axis=-1).min().cpu().item()
                          for f in tqdm(torch.from_numpy(feature_ood).cuda().float())])
    auc_ood = utils.auc(score_id, score_ood)[0]
    fpr_ood, _ = utils.fpr_recall(score_id, score_ood, recall)
    result = dict(method=method, oodset=name, auroc=auc_ood, fpr=fpr_ood)
    print(f'{method}: auroc {auc_ood:.2%}, fpr {fpr_ood:.2%}')


def gradnorm(x, w, b, num_classes=1000):
    fc = torch.nn.Linear(*w.shape[::-1])
    fc.weight.data[...] = torch.from_numpy(w)
    fc.bias.data[...] = torch.from_numpy(b)
    fc.cuda()

    x = torch.from_numpy(x).float().cuda()
    logsoftmax = torch.nn.LogSoftmax(dim=-1).cuda()

    confs = []

    for i in tqdm(x):
        targets = torch.ones((1, num_classes)).cuda()
        fc.zero_grad()
        loss = torch.mean(
            torch.sum(-targets * logsoftmax(fc(i[None])), dim=-1))
        loss.backward()
        layer_grad_norm = torch.sum(
            torch.abs(fc.weight.grad.data)).cpu().numpy()
        confs.append(layer_grad_norm)

    return np.array(confs)

def multiple_proj( id_features, id_labels, ood_features,  name,model, device, n_components=128):
    method = 'Multi_proj'
    """
    Perform the entire pipeline of feature extraction, subspace creation, 
    projection, and reconstruction error calculation in one function.
    """
    # Step 1: Extract Features
    def extract_features(loader):
        features, labels = [], []
        with torch.no_grad():
            for inputs, targets in loader:
                inputs, targets = inputs.to(device), targets.to(device)
                _, _, penultimate_features, _ = model(inputs)
                features.append(penultimate_features.cpu())
                labels.append(targets.cpu())
        return torch.cat(features), torch.cat(labels)
    
    print("Extracting ID and OOD features...")
    # id_features, id_labels = extract_features(id_loader)
    # ood_features, _ = extract_features(ood_loader)
    
    # Step 2: Create Subspaces
    def create_subspaces(features, labels):
        subspaces = {}
        for i in np.unique(labels):  # Iterate over unique classes
            class_features = features[labels == i]
            svd = TruncatedSVD(n_components=n_components)
            svd.fit(class_features)
            subspaces[int(i)] = svd.components_  # Basis vectors for class i
        return subspaces
    
    print("Creating subspaces for ID classes...")
    subspaces = create_subspaces(id_features, id_labels)
    
    # Step 3: Project and Analyze Samples
    def project_and_analyze(sample, subspaces):
        alignments = {}
        for cls, basis in subspaces.items():
            projection = sample @ basis.T @ basis  # Project onto subspace
            error = np.linalg.norm(sample - projection)  # Reconstruction error
            alignments[cls] = error
        sorted_alignments = sorted(alignments.items(), key=lambda x: x[1])
        return sorted_alignments
    
    # Step 4: Calculate Best Reconstruction Errors
    def calculate_best_errors(features, label):
        scores = []
        for sample in features:
            alignment_errors = project_and_analyze(sample, subspaces)
            best_error = alignment_errors[0][1]  # Smallest error
            scores.append(best_error)
        print(f"Calculated best errors for {label} data.")
        return scores
    
    print("Analyzing features...")
    score_id = calculate_best_errors(id_features, "ID")
    score_ood = calculate_best_errors(ood_features, "OOD")
    recall = 0.95
    auc_ood = utils.auc(score_ood, score_id)[0]
    fpr_ood, _ = utils.fpr_recall(score_ood, score_id, recall)
    result = dict(method=method, oodset=name, auroc=auc_ood, fpr=fpr_ood)

    print(f'{method}:  auroc {auc_ood:.2%}, fpr {fpr_ood:.2%}')  # {name}