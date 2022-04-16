import os.path as osp
import random
import torch
import torch.nn.functional as F
from nets import *
from data_utils import *
from args import parse_args
from models import *
from losses import *
from sklearn.metrics import balanced_accuracy_score, f1_score
import statistics
import numpy as np

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

## Arg Parser ##
args = parse_args()

## Handling exception from arguments ##
assert not (args.warmup < 1 and args.ens)
assert args.imb_ratio == 100

## Load Dataset ##
dataset = args.dataset
path = osp.join(osp.dirname(osp.realpath(__file__)), 'data', dataset)
dataset = get_dataset(dataset, path, split_type='full')
data = dataset[0]
n_cls = data.y.max().item() + 1
data = data.to(device)


def backward_hook(module, grad_input, grad_output):
    global saliency
    saliency = grad_input[0].data


def train():
    global class_num_list, idx_info, prev_out, aggregator
    global data_train_mask, data_val_mask, data_test_mask, train_node_mask, train_edge_mask

    model.train()
    optimizer.zero_grad()

    if args.ens:
        # Hook saliency map of input features
        model.conv1[0].temp_weight.register_backward_hook(backward_hook)

        # Sampling source and destination nodes
        sampling_src_idx, sampling_dst_idx = sampling_idx_individual_dst(class_num_list, idx_info, device)
        beta = torch.distributions.beta.Beta(2, 2)
        lam = beta.sample((len(sampling_src_idx),) ).unsqueeze(1)
        ori_saliency = saliency[:data.x.shape[0]] if (saliency != None) else None

        # Augment nodes
        if epoch > args.warmup:
            with torch.no_grad():
                prev_out = aggregator(prev_out, data.edge_index[:,train_edge_mask])
                prev_out = F.softmax(prev_out / args.pred_temp, dim=1).detach().clone()
            new_edge_index, dist_kl = neighbor_sampling(data.x.size(0), data.edge_index[:,train_edge_mask], sampling_src_idx, sampling_dst_idx,
                                        neighbor_dist_list, prev_out, train_node_mask)
            new_x = saliency_mixup(data.x, sampling_src_idx, sampling_dst_idx, lam, ori_saliency, dist_kl = dist_kl, keep_prob=args.keep_prob)
        else:
            new_edge_index = duplicate_neighbor(data.x.size(0), data.edge_index[:,train_edge_mask], sampling_src_idx)
            dist_kl, ori_saliency = None, None
            new_x = saliency_mixup(data.x, sampling_src_idx, sampling_dst_idx, lam, ori_saliency, dist_kl = dist_kl)
        new_x.requires_grad = True

        # Get predictions
        output = model(new_x, new_edge_index, None)
        prev_out = (output[:data.x.size(0)]).detach().clone() # logit propagation

        ## Train_mask modification ##
        add_num = output.shape[0] - data_train_mask.shape[0]
        new_train_mask = torch.ones(add_num, dtype=torch.bool, device= data.x.device)
        new_train_mask = torch.cat((data_train_mask, new_train_mask), dim =0)

        ## Label modification ##
        new_y = data.y[sampling_src_idx].clone()
        new_y = torch.cat((data.y[data_train_mask], new_y),dim =0)

        ## Compute Loss ##
        criterion(output[new_train_mask], new_y).backward()

    else: ## Vanilla Train ##
        output = model(data.x, data.edge_index[:,train_edge_mask], None)
        criterion(output[data_train_mask], data.y[data_train_mask]).backward()

    with torch.no_grad():
        model.eval()
        output = model(data.x, data.edge_index[:,train_edge_mask], None)
        val_loss= F.cross_entropy(output[data_val_mask], data.y[data_val_mask])

    optimizer.step()
    scheduler.step(val_loss)


@torch.no_grad()
def test():
    model.eval()
    logits = model(data.x, data.edge_index[:,train_edge_mask], None,)
    accs, baccs, f1s = [], [], []

    for i, mask in enumerate([data_train_mask, data_val_mask, data_test_mask]):
        pred = logits[mask].max(1)[1]
        y_pred = pred.cpu().numpy()
        y_true = data.y[mask].cpu().numpy()
        acc = pred.eq(data.y[mask]).sum().item() / mask.sum().item()
        bacc = balanced_accuracy_score(y_true, y_pred)
        f1 = f1_score(y_true, y_pred, average='macro')

        accs.append(acc)
        baccs.append(bacc)
        f1s.append(f1)

    return accs, baccs, f1s


## Log for Experiment Setting ##
setting_log = "Dataset: {}, ratio: {}, net: {}, n_layer: {}, feat_dim: {}, ens: {}".format(
    args.dataset, str(args.imb_ratio), args.net, str(args.n_layer), str(args.feat_dim), str(args.ens))

repeatition = 5
seed = 100
avg_test_acc, avg_val_acc, avg_val_f1, avg_test_bacc, avg_test_f1 = [], [], [], [], []
for r in range(repeatition):

    ## Fix seed ##
    torch.cuda.empty_cache()
    seed += 1
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    random.seed(seed)
    np.random.seed(seed)

    data_train_mask, data_val_mask, data_test_mask = data.train_mask.clone(), data.val_mask.clone(), data.test_mask.clone()

    ## Data statistic ##
    stats = data.y[data_train_mask]
    n_data = []
    for i in range(n_cls):
        data_num = (stats == i).sum()
        n_data.append(int(data_num.item()))
    idx_info = get_idx_info(data.y, n_cls, data_train_mask)
    class_num_list = n_data

    ## Construct a long-tailed graph ##
    class_num_list, data_train_mask, idx_info, train_node_mask, train_edge_mask = make_longtailed_data_remove(data.edge_index, \
                                                                data.y, n_data, n_cls, args.imb_ratio, data_train_mask.clone())

    ## Adjacent node distribution ##
    if args.ens:
        neighbor_dist_list = get_ins_neighbor_dist(data.y.size(0), data.edge_index[:,train_edge_mask], data_train_mask, device)
    else:
        neighbor_dist_list = None

    ## Model Selection ##
    if args.net == 'GCN':
        model = GCN(args.n_layer, dataset.num_features, args.feat_dim, n_cls, normalize=True, is_add_self_loops=True)
    elif args.net == 'GAT':
        model = GAT(args.n_layer, dataset.num_features, args.feat_dim, n_cls, args.n_head, is_add_self_loops=True)
    elif args.net == "SAGE":
        model = SAGE(args.n_layer, dataset.num_features, args.feat_dim, n_cls)
    else:
        raise NotImplementedError("Not Implemented Architecture!")

    ## Criterion Selection ##
    if args.loss_type == 'ce': # CE
        criterion = CrossEntropy()
    else:
        raise NotImplementedError("Not Implemented Loss!")

    model = model.to(device)
    criterion = criterion.to(device)

    # Set optimizer
    optimizer = torch.optim.Adam([
        dict(params=model.reg_params, weight_decay=5e-4),
        dict(params=model.non_reg_params, weight_decay=0),], lr=0.01)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min',
                                                           factor = 0.5,
                                                           patience = 100,
                                                           verbose=False)

    # Train models
    best_val_acc = test_acc = best_val_f1 = 0
    saliency = None
    prev_out = None
    aggregator = MeanAggregation()
    for epoch in range(1, 2001):

        train()
        accs, bacc, f1s = test()
        train_acc, val_acc, tmp_test_acc = accs
        train_f1, tmp_val_f1, tmp_test_f1 = f1s
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            val_f1 = tmp_val_f1
            test_acc = tmp_test_acc
            test_bacc = bacc[2]
            test_f1 = f1s[2]

    avg_val_acc.append(best_val_acc)
    avg_val_f1.append(val_f1)
    avg_test_acc.append(test_acc)
    avg_test_bacc.append(test_bacc)
    avg_test_f1.append(test_f1)

## Calculate statistics ##
acc_CI =  (statistics.stdev(avg_test_acc) / (repeatition ** (1/2)))
bacc_CI =  (statistics.stdev(avg_test_bacc) / (repeatition ** (1/2)))
f1_CI =  (statistics.stdev(avg_test_f1) / (repeatition ** (1/2)))
avg_acc = statistics.mean(avg_test_acc)
avg_val_acc = statistics.mean(avg_val_acc)
avg_val_f1 = statistics.mean(avg_val_f1)
avg_bacc = statistics.mean(avg_test_bacc)
avg_f1 = statistics.mean(avg_test_f1)

avg_log = 'Test Acc: {:.4f} +- {:.4f}, BAcc: {:.4f} +- {:.4f}, F1: {:.4f} +- {:.4f}, Val Acc: {:.4f}, Val F1: {:.4f}'
avg_log = avg_log.format(avg_acc ,acc_CI ,avg_bacc, bacc_CI, avg_f1, f1_CI, avg_val_acc, avg_val_f1)
log = "{}\n{}".format(setting_log, avg_log)
print(log)
