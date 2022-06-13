import os
import numpy as np
import random

import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter

from settings import setting as setting
import dataset.dataload_TDASTS as ld
from models.model_prototype import Prototype_Classifier

import utils.load as lo
import utils.make as mk
import utils.loss as ls
from utils.make import remove_file_train, save_args, bestsummary, writelog
from experiments.exp_step2 import train, evaluate
from models.model_relation import FCTransformer_Cls


def main(args, paths_all, f):

    gpu_id = args.gpu
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    torch.cuda.empty_cache()
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

    # Load the model
    model_recon = FCTransformer_Cls(args).to(device)
    model_prototype = Prototype_Classifier(args, 'input_size').to(device)
    train_data, valid_data, test_data = ld.dataloader(args, f)
    main_path, tpath, mpath, vpath = paths_all

    criterion_cls = nn.NLLLoss(reduction="mean").to(device)
    criterion_rec = nn.L1Loss(reduction="sum").to(device)
    criterion_proto_reg = ls.Prototype_margin_order(args, 'input_size').to(device)

    model, optimizer, scheduler = lo.load_optimizer_step2(args, model_recon, model_prototype,
                                                          criterion_proto_reg.proto_margin_loss,
                                                          f, args.best_epoch)

    writer = SummaryWriter(tpath + 'f%d' % f)
    best_loss = float("inf")
    for epoch in range(1, args.epoch + 1):
        trn_loss, trn_closs, trn_mloss, trn_rloss, result_tr = train(args, device, epoch, [model_recon, model_prototype], optimizer, train_data, [criterion_cls, criterion_rec, criterion_proto_reg], paths_all)
        val_loss, val_closs, val_mloss, val_rloss, result_v = evaluate(args, device, epoch, [model_recon, model_prototype], valid_data, vpath, [criterion_cls, criterion_rec, criterion_proto_reg], 'valid')
        tst_loss, tst_closs, tst_mloss, tst_rloss, result_t = evaluate(args, device, epoch, [model_recon, model_prototype], test_data, vpath, [criterion_cls, criterion_rec, criterion_proto_reg], 'test')

        if val_loss < best_loss:
            best_loss = val_loss
            best_epoch_los = epoch
            result_los = result_t.copy()

        scheduler.step()
        writer.add_scalars('ACC', {"train": result_tr[0], "valid": result_v[0], "test": result_t[0]}, epoch)
        writer.add_scalars('LOS', {"train": trn_loss, "valid": val_loss, "test": tst_loss}, epoch)
        writer.add_scalars('CLS', {"train": trn_closs, "valid": val_closs, "test": tst_closs}, epoch)
        writer.add_scalars('REC', {"train": trn_rloss, "valid": val_rloss, "test": tst_rloss}, epoch)
        writer.add_scalars('PREG', {"train": trn_mloss, "valid": val_mloss, "test": tst_mloss}, epoch)

    best_epoch = np.unique(np.asarray([best_epoch_los]))
    remove_file_train(mpath, best_epoch, args.epoch + 1)

    final_vlos = [best_epoch_los, result_los[1], result_los[0], result_los[2], result_los[3]]
    del best_epoch_los

    return final_vlos


if __name__ == "__main__":
    """ Step 2 Relational learning + Prototype learning """

    args = setting.get_args()
    log_path = mk.mk_paths_base(args, 0, 'set the main path')[0]
    save_args(log_path, args)
    summary = open(log_path + 'summary.txt', 'w', encoding='utf-8')
    summary.write("Criterion\tAUC\tACC\tSEN\tSPC\n")

    arr_los = []
    for f in range(1, 6):
        print("Fold", f)
        paths = mk.mk_paths_base(args, f, 'set the main path')
        final_vauc, final_vacc, final_vlos = main(args, paths, f)
        arr_los.append(final_vlos)
        print("Fold Results", final_vlos)
    los_fmean = np.array(arr_los).mean(0)
    np.savez(log_path + 'results_fold', VAL_LOS=np.array(arr_los))
    writelog(summary, 'VAL_LOS\t{:.4f}\t{:.4f}\t{:.4f}\t{:.4f}'.format(los_fmean[1], los_fmean[2], los_fmean[3], los_fmean[4]))
