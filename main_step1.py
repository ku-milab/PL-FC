import os
import numpy as np
import random

import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter

from settings import setting as setting
import dataset.dataload_TDASTS as ld

from models.model_relation_step1 import FCTransformer_Cls
import utils.load as lo
import utils.make as mk
from utils.make import remove_file_train, save_args
from experiments.exp_step1 import train, evaluate


def main(args, paths_all, f):

    gpu_id = args.gpu
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    torch.cuda.empty_cache()

    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

    # Load the model
    model = FCTransformer_Cls(args).to(device)
    optimizer, scheduler = lo.load_optimizer_step1(args, model)
    train_data, valid_data, test_data = ld.dataloader(args, f)
    main_path, tpath, mpath, vpath = paths_all

    # Loss
    criterion_rec = nn.L1Loss(reduction="sum").to(device)

    writer = SummaryWriter(tpath + 'f%d' % f)
    best_loss, best_auc, best_acc = float("inf"), float(0), float(0)
    for epoch in range(1, args.epoch + 1):
        trn_loss, trn_rloss = train(args, device, epoch,  model, optimizer, train_data, criterion_rec, paths_all)
        val_loss, val_rloss = evaluate(args, device, epoch, model, valid_data, vpath, criterion_rec, 'valid')
        tst_loss, tst_rloss = evaluate(args, device, epoch, model, test_data, vpath, criterion_rec, 'test')

        if val_loss < best_loss:
            best_loss = val_loss
            best_epoch_los = epoch

        scheduler.step()
        writer.add_scalars('LOS', {"train": trn_loss, "valid": val_loss, "test": tst_loss}, epoch)
        writer.add_scalars('REC', {"train": trn_rloss, "valid": val_rloss, "test": tst_rloss}, epoch)

    best_epoch = np.unique(np.asarray([20, 40, 60, 80, 100, best_epoch_los]))
    remove_file_train(mpath, best_epoch, args.epoch + 1)

    final_vlos = [ best_epoch_los]

    del best_epoch_los
    return final_vlos


if __name__ == "__main__":

    """ Step 1 Relational learning by Reconstruction """
    args = setting.get_args()
    log_path = mk.mk_paths_base(args, 0, 'set the path')[0]
    save_args(log_path, args)
    summary = open(log_path + 'summary.txt', 'w', encoding='utf-8')
    summary.write("Criterion\tAUC\tACC\tSEN\tSPC\n")

    arr_los = []
    for f in range(1, 6):
        paths = mk.mk_paths_base(args, f, 'set the path')
        final_vlos = main(args, paths, f)
        arr_los.append(final_vlos)
        print("Fold Results", final_vlos)
    np.savez(log_path + 'results_fold', VAL_LOS=np.array(arr_los))
