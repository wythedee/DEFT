import argparse
import random

import numpy as np
import torch

from datasets import faced_dataset, seedv_dataset, physio_dataset, shu_dataset, isruc_dataset, isruc_s3_dataset, chb_dataset, \
    speech_dataset, mumtaz_dataset, seedvig_dataset, stress_dataset, tuev_dataset, tuab_dataset, bciciv2a_dataset, sch2017_dataset, hebin2021_dataset, seediv_dataset, seed_dataset, cs_bcic_track4_dataset, atten_dataset, mi_koreau_dataset, mi_track1_fewshot_dataset, mi_cho2017_dataset, mi_shin2017a_dataset, mi_weibo2014_dataset
from finetune_trainer import Trainer
from models import model_for_faced, model_for_seedv, model_for_physio, model_for_shu, model_for_isruc, model_for_chb, \
    model_for_speech, model_for_mumtaz, model_for_seedvig, model_for_stress, model_for_tuev, model_for_tuab, \
    model_for_bciciv2a, model_for_sch2017, model_for_hebin, model_for_seediv, model_for_seed, model_for_cs_bcic_track4, model_for_atten, model_for_mi_koreau, model_for_mi_track1_fewshot, model_for_mi_cho2017, model_for_mi_shin2017a, model_for_mi_weibo2014


def main():
    parser = argparse.ArgumentParser(description='Big model downstream')
    parser.add_argument('--seed', type=int, default=3407, help='random seed (default: 0)')
    parser.add_argument('--cuda', type=int, default=1, help='cuda number (default: 1)')
    parser.add_argument('--epochs', type=int, default=50, help='number of epochs (default: 5)')
    parser.add_argument('--batch_size', type=int, default=64, help='batch size for training (default: 32)')
    parser.add_argument('--lr', type=float, default=1e-4, help='learning rate (default: 1e-3)')
    parser.add_argument('--weight_decay', type=float, default=5e-2, help='weight decay (default: 1e-2)')
    parser.add_argument('--optimizer', type=str, default='AdamW', help='optimizer (AdamW, SGD)')
    parser.add_argument('--clip_value', type=float, default=1, help='clip_value')
    parser.add_argument('--dropout', type=float, default=0.1, help='dropout')
    parser.add_argument('--classifier', type=str, default='all_patch_reps',
                        help='[all_patch_reps, all_patch_reps_twolayer, '
                             'all_patch_reps_onelayer, avgpooling_patch_reps]')
    # all_patch_reps: use all patch features with a three-layer classifier;
    # all_patch_reps_twolayer: use all patch features with a two-layer classifier;
    # all_patch_reps_onelayer: use all patch features with a one-layer classifier;
    # avgpooling_patch_reps: use average pooling for patch features;

    """############ Downstream dataset settings ############"""
    parser.add_argument('--downstream_dataset', type=str, default='FACED',
                    help='[FACED, SEED-V, PhysioNet-MI, SHU-MI, ISRUC, ISRUC-S3, CHB-MIT, BCIC2020-3, CS-BCIC-Track4, Mumtaz2016, '
                        'SEED-VIG, MentalArithmetic, TUEV, TUAB, BCIC-IV-2a, Schirrmeister2017, HeBin2021-LR, HeBin2021-UD, '
                        'SEED-IV, SEED, ATTEN, MI-KoreaU, MI-Track1-FewShot, MI-Cho2017, MI-Shin2017A, MI-Weibo2014]')
    parser.add_argument('--datasets_dir', type=str,
                        default='/path/to/eeg/Faced/processed',
                        help='datasets_dir')
    parser.add_argument('--num_of_classes', type=int, default=9, help='number of classes')
    parser.add_argument('--model_dir', type=str, default='/path/to/outputs/Big/BigFaced', help='model_dir')
    """############ Downstream dataset settings ############"""

    parser.add_argument('--num_workers', type=int, default=16, help='num_workers')
    parser.add_argument('--label_smoothing', type=float, default=0.1, help='label_smoothing')
    parser.add_argument('--multi_lr', type=bool, default=True,
                        help='multi_lr')  # set different learning rates for different modules
    parser.add_argument('--frozen', type=bool,
                        default=False, help='frozen')
    parser.add_argument('--use_pretrained_weights', type=bool,
                        default=True, help='use_pretrained_weights')
    parser.add_argument('--foundation_dir', type=str,
                        default='pretrained_weights/pretrained_weights.pth',
                        help='foundation_dir')
    parser.add_argument('--enable_progress_bar', type=bool, default=False, help='enable_progress_bar')
    parser.add_argument('--write_summary_json', action='store_true',
                        help='Write summary.json under model_dir after training.')
    parser.add_argument('--test_channel_permutation_mode', type=str, default='identity',
                        choices=['identity', 'reverse', 'shuffle', 'custom'],
                        help='How to permute channel order during evaluation. "identity" keeps the original order.')
    parser.add_argument('--test_channel_permutation_custom', type=str, default='',
                        help='Comma/whitespace/JSON list (or file path) with 0-based indices for custom permutations.')
    parser.add_argument('--test_channel_permutation_seed', type=int, default=3407,
                        help='Random seed for shuffle mode channel permutations.')
    parser.add_argument('--test_channel_axis', type=int, default=1,
                        help='Tensor axis that corresponds to channels (defaults to 1).')
    parser.add_argument('--test_channel_permutation_apply_to_val', action='store_true',
                        help='Apply the same channel permutation to validation as well as test.')

    params = parser.parse_args()
    print(params)

    setup_seed(params.seed)
    torch.cuda.set_device(params.cuda)
    print('The downstream dataset is {}'.format(params.downstream_dataset))
    if params.downstream_dataset == 'FACED':
        load_dataset = faced_dataset.LoadDataset(params)
        data_loader = load_dataset.get_data_loader()
        model = model_for_faced.Model(params)
        t = Trainer(params, data_loader, model)
        t.train_for_multiclass()
    elif params.downstream_dataset == 'SEED-V':
        load_dataset = seedv_dataset.LoadDataset(params)
        data_loader = load_dataset.get_data_loader()
        model = model_for_seedv.Model(params)
        t = Trainer(params, data_loader, model)
        t.train_for_multiclass()
    elif params.downstream_dataset == 'PhysioNet-MI':
        load_dataset = physio_dataset.LoadDataset(params)
        data_loader = load_dataset.get_data_loader()
        model = model_for_physio.Model(params)
        t = Trainer(params, data_loader, model)
        t.train_for_multiclass()
    elif params.downstream_dataset == 'SHU-MI':
        load_dataset = shu_dataset.LoadDataset(params)
        data_loader = load_dataset.get_data_loader()
        model = model_for_shu.Model(params)
        t = Trainer(params, data_loader, model)
        t.train_for_binaryclass()
    elif params.downstream_dataset == 'ISRUC':
        load_dataset = isruc_dataset.LoadDataset(params)
        data_loader = load_dataset.get_data_loader()
        model = model_for_isruc.Model(params)
        t = Trainer(params, data_loader, model)
        t.train_for_multiclass()
    elif params.downstream_dataset == 'ISRUC-S3':
        load_dataset = isruc_s3_dataset.LoadDataset(params)
        data_loader = load_dataset.get_data_loader()
        model = model_for_isruc.Model(params)
        t = Trainer(params, data_loader, model)
        t.train_for_multiclass()
    elif params.downstream_dataset == 'CHB-MIT':
        load_dataset = chb_dataset.LoadDataset(params)
        data_loader = load_dataset.get_data_loader()
        model = model_for_chb.Model(params)
        t = Trainer(params, data_loader, model)
        t.train_for_binaryclass()
    elif params.downstream_dataset == 'BCIC2020-3':
        load_dataset = speech_dataset.LoadDataset(params)
        data_loader = load_dataset.get_data_loader()
        model = model_for_speech.Model(params)
        t = Trainer(params, data_loader, model)
        t.train_for_multiclass()
    elif params.downstream_dataset == 'CS-BCIC-Track4':
        load_dataset = cs_bcic_track4_dataset.LoadDataset(params)
        data_loader = load_dataset.get_data_loader()
        model = model_for_cs_bcic_track4.Model(params)
        t = Trainer(params, data_loader, model)
        t.train_for_multiclass()
    elif params.downstream_dataset == 'Mumtaz2016':
        load_dataset = mumtaz_dataset.LoadDataset(params)
        data_loader = load_dataset.get_data_loader()
        model = model_for_mumtaz.Model(params)
        t = Trainer(params, data_loader, model)
        t.train_for_binaryclass()
    elif params.downstream_dataset == 'SEED-VIG':
        load_dataset = seedvig_dataset.LoadDataset(params)
        data_loader = load_dataset.get_data_loader()
        model = model_for_seedvig.Model(params)
        t = Trainer(params, data_loader, model)
        t.train_for_regression()
    elif params.downstream_dataset == 'MentalArithmetic':
        load_dataset = stress_dataset.LoadDataset(params)
        data_loader = load_dataset.get_data_loader()
        model = model_for_stress.Model(params)
        t = Trainer(params, data_loader, model)
        t.train_for_binaryclass()
    elif params.downstream_dataset == 'TUEV':
        load_dataset = tuev_dataset.LoadDataset(params)
        data_loader = load_dataset.get_data_loader()
        model = model_for_tuev.Model(params)
        t = Trainer(params, data_loader, model)
        t.train_for_multiclass()
    elif params.downstream_dataset == 'TUAB':
        load_dataset = tuab_dataset.LoadDataset(params)
        data_loader = load_dataset.get_data_loader()
        model = model_for_tuab.Model(params)
        t = Trainer(params, data_loader, model)
        t.train_for_binaryclass()
    elif params.downstream_dataset == 'BCIC-IV-2a':
        load_dataset = bciciv2a_dataset.LoadDataset(params)
        data_loader = load_dataset.get_data_loader()
        model = model_for_bciciv2a.Model(params)
        t = Trainer(params, data_loader, model)
        t.train_for_multiclass()
    elif params.downstream_dataset == 'Schirrmeister2017':
        load_dataset = sch2017_dataset.LoadDataset(params)
        data_loader = load_dataset.get_data_loader()
        model = model_for_sch2017.Model(params)
        t = Trainer(params, data_loader, model)
        t.train_for_binaryclass()
    elif params.downstream_dataset == 'HeBin2021-LR':
        load_dataset = hebin2021_dataset.LoadDataset(params)
        data_loader = load_dataset.get_data_loader()
        model = model_for_hebin.Model(params)
        t = Trainer(params, data_loader, model)
        t.train_for_binaryclass()
    elif params.downstream_dataset == 'HeBin2021-UD':
        load_dataset = hebin2021_dataset.LoadDataset(params)
        data_loader = load_dataset.get_data_loader()
        model = model_for_hebin.Model(params)
        t = Trainer(params, data_loader, model)
        t.train_for_binaryclass()
    elif params.downstream_dataset == 'SEED-IV':
        load_dataset = seediv_dataset.LoadDataset(params)
        data_loader = load_dataset.get_data_loader()
        model = model_for_seediv.Model(params)
        t = Trainer(params, data_loader, model)
        t.train_for_multiclass()
    elif params.downstream_dataset == 'SEED':
        load_dataset = seed_dataset.LoadDataset(params)
        data_loader = load_dataset.get_data_loader()
        model = model_for_seed.Model(params)
        t = Trainer(params, data_loader, model)
        t.train_for_binaryclass()
    elif params.downstream_dataset == 'ATTEN':
        load_dataset = atten_dataset.LoadDataset(params)
        data_loader = load_dataset.get_data_loader()
        model = model_for_atten.Model(params)
        t = Trainer(params, data_loader, model)
        t.train_for_binaryclass()
    elif params.downstream_dataset == 'MI-KoreaU':
        load_dataset = mi_koreau_dataset.LoadDataset(params)
        data_loader = load_dataset.get_data_loader()
        model = model_for_mi_koreau.Model(params)
        t = Trainer(params, data_loader, model)
        t.train_for_binaryclass()
    elif params.downstream_dataset == 'MI-Track1-FewShot':
        load_dataset = mi_track1_fewshot_dataset.LoadDataset(params)
        data_loader = load_dataset.get_data_loader()
        model = model_for_mi_track1_fewshot.Model(params)
        t = Trainer(params, data_loader, model)
        t.train_for_binaryclass()
    elif params.downstream_dataset == 'MI-Cho2017':
        load_dataset = mi_cho2017_dataset.LoadDataset(params)
        data_loader = load_dataset.get_data_loader()
        model = model_for_mi_cho2017.Model(params)
        t = Trainer(params, data_loader, model)
        t.train_for_binaryclass()
    elif params.downstream_dataset == 'MI-Shin2017A':
        load_dataset = mi_shin2017a_dataset.LoadDataset(params)
        data_loader = load_dataset.get_data_loader()
        model = model_for_mi_shin2017a.Model(params)
        t = Trainer(params, data_loader, model)
        t.train_for_binaryclass()
    elif params.downstream_dataset == 'MI-Weibo2014':
        load_dataset = mi_weibo2014_dataset.LoadDataset(params)
        data_loader = load_dataset.get_data_loader()
        model = model_for_mi_weibo2014.Model(params)
        t = Trainer(params, data_loader, model)
        t.train_for_binaryclass()
    print('Done!!!!!')


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


if __name__ == '__main__':
    main()
