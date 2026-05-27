import argparse
import datetime
import os
import random
import warnings

import numpy as np
import torch

try:
    from Downstream.finetune_trainer import Trainer
    from Downstream.standard_finetune import available_downstream_datasets, build_loader_and_model
except ImportError:
    from finetune_trainer import Trainer
    from standard_finetune import available_downstream_datasets, build_loader_and_model

warnings.filterwarnings('ignore')


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description='CodeBrain Downstream')
    parser.add_argument('--seed', type=int, default=42, help='random seed (default: 0)')
    parser.add_argument('--cuda', type=int, default=4, help='cuda number (default: 1)')
    parser.add_argument('--epochs', type=int, default=50, help='number of epochs (default: 5)')
    parser.add_argument('--batch_size', type=int, default=64, help='batch size for training (default: 32)')
    parser.add_argument('--lr', type=float, default=5e-4, help='learning rate (default: 1e-3)')
    parser.add_argument('--weight_decay', type=float, default=5e-4, help='weight decay (default: 1e-2)')
    parser.add_argument('--optimizer', type=str, default='AdamW', help='optimizer (AdamW, SGD)')
    parser.add_argument('--clip_value', type=float, default=5, help='clip_value')
    parser.add_argument('--dropout', type=float, default=0.1, help='dropout')
    parser.add_argument('--n_layer', type=int, default=8, help='n_layer')
    parser.add_argument('--classifier', type=str, default='all_patch_reps',
                        help='classifier head type')
    parser.add_argument('--select_best_by', type=str, default='',
                        choices=['', 'val', 'val_kappa', 'val_f1', 'val_roc_auc', 'val_pr_auc'],
                        help='selection metric for best epoch')
    parser.add_argument('--multi_lr', action=argparse.BooleanOptionalAction, default=True,
                        help='set different lrs for backbone/head')

    parser.add_argument('--downstream_dataset', type=str, default='SEED-V',
                        choices=available_downstream_datasets(),
                        help='downstream dataset')
    parser.add_argument('--datasets_dir', type=str,
                        default='',
                        help='datasets_dir')
    parser.add_argument('--num_of_classes', type=int, default=0, help='number of classes (auto-filled by registry)')
    parser.add_argument('--model_dir', type=str,
                        default='',
                        help='model_dir')
    parser.add_argument('--log_dir', type=str,
                        default='',
                        help='log_dir')

    parser.add_argument('--num_workers', type=int, default=16, help='num_workers')
    parser.add_argument('--label_smoothing', type=float, default=0.1, help='label_smoothing')
    parser.add_argument('--frozen', action=argparse.BooleanOptionalAction,
                        default=False, help='frozen')
    parser.add_argument('--use_pretrained_weights', action=argparse.BooleanOptionalAction,
                        default=True, help='use_pretrained_weights')
    parser.add_argument('--foundation_dir', type=str,
                        default='',
                        help='foundation_dir')

    parser.add_argument('--codebook_size_t', default=4096, type=int,
                        help='number of temporal codebook (default: 4096)')
    parser.add_argument('--codebook_size_f', default=4096, type=int,
                        help='number of frequency codebook (default: 4096)')
    parser.add_argument('--codebook_dim', default=32, type=int,
                        help='dimention of codebook (default: 32)')
    parser.add_argument('--write_summary_json', action=argparse.BooleanOptionalAction,
                        default=False, help='write summary.json into model_dir')
    return parser


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


def run_from_namespace(params: argparse.Namespace) -> None:
    params.model_dir = params.model_dir + params.downstream_dataset + '/'
    params.log_dir = params.log_dir + params.downstream_dataset + '/'
    os.makedirs(params.log_dir, exist_ok=True)
    current_time = datetime.datetime.now().strftime('%Y_%m_%d_%H_%M_%S')
    params.file_name = str(params.log_dir) + str(current_time) + '_' + str(params.cuda) + '.txt'

    print(params)
    with open(params.file_name, 'a') as file:
        file.write(str(params) + '\n')

    setup_seed(params.seed)
    torch.cuda.set_device(params.cuda)
    print('The downstream dataset is {}'.format(params.downstream_dataset))
    with open(params.file_name, 'a') as file:
        file.write('The downstream dataset is {}'.format(params.downstream_dataset))
        file.write('\n')

    data_loader, model, task = build_loader_and_model(params)
    params.task_kind = task
    trainer = Trainer(params, data_loader, model)

    if task == 'multiclass':
        trainer.train_for_multiclass()
    elif task == 'binary':
        trainer.train_for_binaryclass()
    else:
        raise ValueError(f'Unsupported task kind: {task}')


def run_from_dict(overrides=None) -> None:
    parser = build_parser()
    params = parser.parse_args([])
    if overrides:
        for key, value in overrides.items():
            if not hasattr(params, key):
                raise ValueError(f'Unknown config key: {key}')
            setattr(params, key, value)
    run_from_namespace(params)


def run_cli(argv=None) -> None:
    parser = build_parser()
    params = parser.parse_args(argv)
    run_from_namespace(params)


def main() -> None:
    run_cli()


if __name__ == '__main__':
    main()
