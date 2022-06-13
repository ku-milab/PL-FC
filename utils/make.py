import os
import json
import csv


def create_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)
    return path


def save_args(savepath, ARGS):
    with open(savepath + '/settings.txt', 'w') as f:
        json.dump(ARGS.__dict__, f, indent=2)
    f.close()


def bestsummary(savepath, fi, case):
    if fi == 1:
        train_summary = open(savepath + '/summary_%s.txt' % case, 'w', encoding='utf-8')
        train_summary.writelines(["Fold\t", "Epoch\t", "AUC\t", "ACC\t", "SEN\t", "SPC\n"])
    else:
        train_summary = open(savepath + '/summary_%s.txt' % case, 'a')
    return train_summary

def unseensummary(savepath, site):
    if site == 'Caltech':
        train_summary = open(savepath + '/summary_unseen.csv', 'w', encoding='utf-8')
        wr_summary = csv.writer(train_summary)
        wr_summary.writerow([ "Site", "Fold", "AUC", "ACC", "SEN", "SPC"])
    else:
        train_summary = open(savepath + '/summary_unseen.csv', 'a')
        wr_summary = csv.writer(train_summary)
    return wr_summary, train_summary

def writesummary(fold, best_ep, summary, auc, acc, sen, spc):

    for b_idx, b in enumerate(best_ep):
        summary.writerow([fold, b, str(auc[b_idx]), str(acc[b_idx]), str(sen[b_idx]), str(spc[b_idx])])

def writeunseensummary(fold, site, summary, auc, acc, sen, spc):
    summary.writerow([site, fold, str(auc), str(acc), str(sen), str(spc)])


def mk_paths_base(args, f, case, next=None):

    base_path = '/%s/' % case
    main_path = base_path + '/lr_tf_{}_lr_proto_{}/TFEncL{}_Head{}_FFH{}_l2_{}/'.format(
                              args.lr_tf, args.lr_proto, args.num_stack, args.num_heads, args.d_ff, args.l2)

    if next:
        main_path = main_path + '{}/'.format(next)

    model_path = main_path + 'ckpt/cv{}/'.format(f)
    vis_path = main_path + 'vis/cv{}/'.format(f)
    tb_path = main_path + 'tb/'

    if f == 0:
        create_dir(main_path)
    else:
        create_dir(tb_path)
        create_dir(model_path)
        create_dir(vis_path)

    paths = [main_path, tb_path, model_path, vis_path]

    return paths




def mk_paths_base_test(args, f, case, next=None):

    base_path = '/DataCommon2/eskang/Transformer/CPAC2022/%s/' % case

    if args.site == 1:
        site_path = base_path + '{}site/{}_W{}S{}/Clf_{}/Prototype{}_Similarity_{}/'.format(args.site, args.site_name, args.window, args.stride,
                                                                  args.clf_type, args.num_proto, args.proto_dist)
    else:
        site_path = base_path + '{}site/W{}S{}/Clf_{}/Prototype{}_Similarity_{}/'.format(args.site, args.window, args.stride,
                                                               args.clf_type, args.num_proto, args.proto_dist)

    main_path1 = site_path + '{}_case{}_reg/Pos_{}_Mask_{}_ratio{}/'.format(args.class_prob, args.case_proto_class, args.pos_type, args.mask_type, args.mask_ratio)
    main_path = main_path1 + '/lr_tf_{}_lr_proto_{}/TFEncL{}_Head{}_FFH{}_l2_{}/'.format(
                              args.lr_tf, args.lr_proto, args.num_stack, args.num_heads, args.d_ff, args.l2)

    if next:
        main_path = main_path + '{}/'.format(next)

    model_path = main_path + 'ckpt/cv{}/'.format(f)
    vis_path = main_path + 'vis/cv{}/'.format(f)
    tb_path = main_path + 'tb/'

    paths = [main_path, tb_path, model_path, vis_path]

    return paths


def mk_paths_base_wows(args, f, case, next=None):

    base_path = '/DataCommon2/eskang/Transformer/CPAC2022/%s/' % case

    if args.site == 1:
        site_path = base_path + '{}site/{}/Clf_{}/Prototype{}_Similarity_{}/'.format(args.site, args.site_name, args.window, args.stride,
                                                                  args.clf_type, args.num_proto, args.proto_dist)
    else:
        site_path = base_path + '{}site/Clf_{}/Prototype{}_Similarity_{}/'.format(args.site,
                                                               args.clf_type, args.num_proto, args.proto_dist)

    main_path1 = site_path + '{}_case{}_reg/Pos_{}_Mask_{}_ratio{}/'.format(args.class_prob, args.case_proto_class, args.pos_type, args.mask_type, args.mask_ratio)
    main_path = main_path1 + '/lr_tf_{}_lr_proto_{}/TFEncL{}_Head{}_FFH{}/'.format(
                              args.lr_tf, args.lr_proto, args.num_stack, args.num_heads, args.d_ff)

    if next:
        main_path = main_path + '{}/'.format(next)

    model_path = main_path + 'ckpt/cv{}/'.format(f)
    vis_path = main_path + 'vis/cv{}/'.format(f)
    tb_path = main_path + 'tb/'

    if f == 0:
        create_dir(main_path)
    else:
        create_dir(tb_path)
        create_dir(model_path)
        create_dir(vis_path)

    paths = [main_path, tb_path, model_path, vis_path]

    return paths



def writelog(file, line):
    """Define the function to print and write log"""
    file.write(line + '\n')
    print(line)

def remove_file_train(path, bestep, epoch):

    for i in range(epoch):
        if i in bestep:
            pass
        elif i == (epoch-1):
            pass
        else:
            bestfile = path + 'E%d.pt' % i
            if os.path.isfile(bestfile):
                os.remove(bestfile)
