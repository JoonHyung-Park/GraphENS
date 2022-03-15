import argparse

def parse_args():
    parser = argparse.ArgumentParser()

    # Dataset
    parser.add_argument('--dataset', type=str, default='Cora',
                        help='Dataset Name')
    parser.add_argument('--imb_ratio', type=float, default=10,
                        help='Imbalance Ratio')
    # Architecture
    parser.add_argument('--net', type=str, default='GCN',
                        help='Architecture name')
    parser.add_argument('--n_layer', type=int, default=2,
                        help='the number of layers')
    parser.add_argument('--feat_dim', type=int, default=256,
                        help='Feature dimension')
    # GAT
    parser.add_argument('--n_head', type=int, default=8,
                        help='the number of heads in GAT')
    # Imbalance Loss
    parser.add_argument('--loss_type', type=str, default='ce',
                        help='Loss type')
    # Method
    parser.add_argument('--ens', action='store_true',
                        help='Mixing node')
    # Hyperparameter for our approach
    parser.add_argument('--keep_prob', type=float, default=0.01,
                        help='Keeping Probability')
    parser.add_argument('--pred_temp', type=float, default=2,
                        help='Prediction temperature')
    parser.add_argument('--warmup', type=int, default=1,
                        help='warmup')
    args = parser.parse_args()

    return args