import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import utils.loss as ls
import utils.metric as mt
import utils.mask as ms

import matplotlib


def train(args, device, epoch, models, optimizer, datalader, criterions, paths):
    model_tf, model_proto = models[0], models[1]
    criterion_cls, criterion_rec, criterion_reg = criterions[0], criterions[1], criterions[2]
    savepath, vispath = paths[-2], paths[-1]

    model_tf.train()
    model_proto.train()
    criterion_reg.proto_margin_loss.train()

    total_aloss = 0.
    total_closs = 0.
    total_rloss = 0.
    total_mloss = 0.
    logits, labels = [], []
    for i, batch in enumerate(zip(datalader)):
        optimizer.zero_grad()
        data, targets = batch[0]

        output, cls_token = model_tf(data.to(device))
        logit = model_proto(cls_token)
        logits.append(logit.detach().cpu().numpy()), labels.append(targets.detach().cpu().numpy())

        "Loss"
        loss_cls = criterion_cls(torch.log(logit), targets.to(device))
        total_closs += loss_cls.item()

        loss_l1_tf = args.l1_tf * ls.compute_l1_loss(model_tf)
        loss_l1_proto = args.l1_proto * ls.compute_l1_loss(model_proto)

        loss_rec = 0.0001*criterion_rec(output, data.to(device)) / (data.shape[0])
        total_rloss += loss_rec.item()

        proto_margin_reg = 0.1*criterion_reg(model_proto.prototypes)
        total_mloss += proto_margin_reg.item()

        loss = loss_cls + loss_rec + proto_margin_reg + loss_l1_tf + loss_l1_proto
        total_aloss += loss.item()
        loss.backward()
        optimizer.step()

    result = mt.cal_all_metric(logits, labels)

    # if epoch % 10 == 0 and epoch == 1:
    #     visualization_cls(output[:, 0, :], vispath, epoch, 'tr')

    print('-' * 89)
    print('| Epoch {:3d} | train cls/rec/reg {:5.4f}/{:5.4f}/{:5.4f} = {:5.4f} | AUC {:5.4f} | ACC {:5.4f} | SEN {:5.4f} | SPC {:5.4f}'.format(
        epoch, total_closs/(i+1), total_rloss/(i+1), total_mloss/(i+1), total_aloss/(i+1), result[1], result[0], result[2], result[3]))
    torch.save({"model_tf": model_tf.state_dict(), "model_proto": model_proto.state_dict(),
                "model_d_cos": criterion_reg.proto_margin_loss}, savepath + "E%d.pt" % (epoch))

    return total_aloss/(i+1), total_closs/(i+1), total_mloss/(i+1), total_rloss/(i+1), result

def evaluate(args, device, epoch, models, data_loader, vispath, criterions, phase):
    model_tf, model_proto = models[0], models[1]
    criterion_cls, criterion_rec, criterion_reg = criterions[0], criterions[1], criterions[2]

    model_tf.eval()
    model_proto.eval()
    criterion_reg.proto_margin_loss.eval()
    with torch.no_grad():
        total_aloss = 0.
        total_closs = 0.
        total_rloss = 0.
        total_mloss = 0.
        logits, labels = [], []
        for i, batch in enumerate(zip(data_loader)):
            data, targets = batch[0]

            output, cls_token = model_tf(data.to(device))
            logit = model_proto(cls_token)
            logits.append(logit.detach().cpu().numpy()), labels.append(targets.detach().cpu().numpy())

            loss_cls = criterion_cls(torch.log(logit), targets.to(device))
            total_closs += loss_cls.item()

            loss_l1_tf = args.l1_tf * ls.compute_l1_loss(model_tf)
            loss_l1_proto = args.l1_proto * ls.compute_l1_loss(model_proto)

            loss_rec = 0.0001 * criterion_rec(output, data.to(device)) / (data.shape[0])
            total_rloss += loss_rec.item()

            proto_margin_reg = 0.1*criterion_reg(model_proto.prototypes)
            total_mloss += proto_margin_reg.item()

            loss = loss_cls + loss_rec + proto_margin_reg + loss_l1_tf + loss_l1_proto
            total_aloss += loss.item()

    result = mt.cal_all_metric(logits, labels)

    if phase == 'valid':
        print('| Epoch {:3d} | valid cls/rec/reg {:5.4f}/{:5.4f}/{:5.4f} = {:5.4f} | AUC {:5.4f} | ACC {:5.4f} | SEN {:5.4f} | SPC {:5.4f}'.format(
                epoch, total_closs / (i + 1), total_rloss / (i + 1), total_mloss / (i + 1), total_aloss / (i + 1), result[1], result[0],
                result[2], result[3]))

    elif phase == 'test':
        print(
            '| Epoch {:3d} | test cls/rec/reg {:5.4f}/{:5.4f}/{:5.4f} = {:5.4f} | AUC {:5.4f} | ACC {:5.4f} | SEN {:5.4f} | SPC {:5.4f}'.format(
                epoch, total_closs / (i + 1), total_rloss / (i + 1), total_mloss / (i + 1), total_aloss / (i + 1),
                result[1], result[0], result[2], result[3]))

    return total_aloss/(i+1), total_closs/(i+1),  total_mloss/(i+1), total_rloss/(i+1), result
