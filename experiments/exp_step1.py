import numpy as np
import torch

import utils.loss as ls
import utils.mask as ms

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.switch_backend('agg')


def train(args, device, epoch, model, optimizer, datalader, criterion_rec, paths):
    savepath, vispath = paths[-2], paths[-1]

    model.train()
    total_rloss = 0.
    total_aloss = 0.
    for i, batch in enumerate(zip(datalader)):
        optimizer.zero_grad()

        data, targets = batch[0]
        output, cls_token = model(data.to(device))

        "Loss"
        loss_rec = criterion_rec(output, data.to(device)) / (data.shape[0])
        total_rloss += loss_rec.item()
        loss_l1 = args.l1 * ls.compute_l1_loss(model)
        loss = loss_rec + loss_l1
        total_aloss += loss.item()

        loss.backward()
        optimizer.step()

    print('-' * 89)
    print('| Epoch {:3d} | train rec {:5.4f} |'.format(epoch, total_rloss/(i+1)))
    torch.save({"model": model.state_dict()}, savepath + "E%d.pt" % (epoch))

    return (total_aloss)/(i+1), total_rloss/(i+1)

def evaluate(args, device, epoch, model, data_loader, vispath, criterion_rec, phase):

    model.eval()
    with torch.no_grad():
        total_rloss = 0.
        total_aloss = 0.
        for i, batch in enumerate(zip(data_loader)):
            data, targets = batch[0]
            output, cls_token = model(data.to(device))

            "Loss"
            loss_rec = criterion_rec(output, data.to(device)) / (data.shape[0])
            total_rloss += loss_rec.item()
            total_aloss += loss_rec.item()

    if phase == 'valid':
        print('| Epoch {:3d} | valid {:5.4f}|'.format(epoch, total_rloss/(i+1)))

    elif phase == 'test':
        print('| Epoch {:3d} | test {:5.4f}|'.format(epoch, total_rloss/(i+1)))

    return (total_aloss)/(i+1), total_rloss/(i+1)
