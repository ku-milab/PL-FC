import argparse

def get_args():
    parser = argparse.ArgumentParser(description="Transformer")
    parser.add_argument("--gpu", type=str, default=0)

    # Pretraining
    parser.add_argument("--atlas", type=str, default='HO_110', choices=['HO_110'])
    parser.add_argument("--site", type=int, default=17, choices=[1, 4, 7, 17])
    parser.add_argument("--epoch", type=int, default=150)
    parser.add_argument("--dynamics", type=bool, default=True)
    parser.add_argument("--bs", type=int, default=32)

    parser.add_argument("--best_epoch", type=int, default= 100, help="Best epoch for the reconstruction")
    parser.add_argument("--input_size", type=int, default= 110, help="The number of ROIs")
    parser.add_argument("--input_emb", type=int, default= 50)
    parser.add_argument("--num_stack", type=int, default=2)
    parser.add_argument("--num_heads", type=int, default=10)
    parser.add_argument("--d_ff", type=int, default=50)
    parser.add_argument("--num_proto", type=int, default=2)
    parser.add_argument("--case_proto_class", type=int, default=4)

    parser.add_argument("--rec_case", type=str, default='roiwise_cls_concat_nonlinear')
    parser.add_argument("--proto_dist", type=str, default='cosine')
    parser.add_argument("--class_prob", type=str, default='sum')

    parser.add_argument("--pos_type", type=str, default='sincos')
    parser.add_argument("--clf_type", type=str, default='NN_NLL')

    parser.add_argument("--schedule_step", type=int, default=25)
    parser.add_argument("--lr_tf", type=float, default=0.00001)
    parser.add_argument("--lr_proto", type=float, default=0.00005)
    parser.add_argument("--l1_tf", type=float, default=0.0)
    parser.add_argument("--l1_proto", type=float, default=0.0)
    parser.add_argument("--l2", type=float, default=0.0001)

    ARGS = parser.parse_args()

    return ARGS