import numpy as np
import torch
import pandas as pd


def load_optimizer_step1(args, model):
    optimizer = torch.optim.AdamW([{"params": model.parameters(), "lr": args.lr_tf}],
                                    lr=args.lr_tf, weight_decay=args.l2)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, args.schedule_step, gamma=0.95)
    return optimizer, scheduler

def load_optimizer_step2(args, model, model2, model3, f, best=None):
    main_path = 'Should be set'

    # Load the best epoch of step 1
    # result = np.load(main_path + 'results_fold.npz')['VAL_LOS']
    # model_path = main_path + 'ckpt/cv{}/E{}.pt'.format(f, int(result[f-1, 0]))
    # best = int(result[f-1, 0]))
    # print("BEST TFFMR EPOCH {}".format(best, result[f-1, :])

    # Load the best epoch manually of step 1 - to load the last epoch
    model_path = main_path + 'ckpt/cv{}/E{}.pt'.format(f, best)
    print("BEST TFFMR EPOCH {}".format(best))

    model_dict = model.state_dict()
    saved_dict = torch.load(model_path)['model']
    # 1. filter out unnecessary keys
    saved_dict = {k: v for k, v in saved_dict.items() if k in model_dict}
    # 2. overwrite entries in the existing state dict
    model_dict.update(saved_dict)
    # 3. load the new state dict
    model.load_state_dict(model_dict)


    optimizer = torch.optim.AdamW([{"params": model.parameters()},
                                   {"params": model2.parameters(), "lr": args.lr_proto},
                                   {"params": model3.parameters(), "lr": args.lr_proto}],
                                  lr=args.lr_tf, weight_decay=args.l2)

    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, args.schedule_step, gamma=0.95)
    return model, optimizer, scheduler


