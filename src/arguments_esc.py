import argparse
import os


def get_parser():
    parser = argparse.ArgumentParser()

    # device number
    parser.add_argument('--device_num', type=int, default=0)
    parser.add_argument('--random_seed', type=int, default=6688)

    # model dir
    parser.add_argument('--save_model', type=str, default='esc_00')
    parser.add_argument('--save_model_dir', type=str, default='save_model')
    parser.add_argument('--output_dir', type=str, default='output')

    # task setting
    parser.add_argument('--model', type=str, default='all')
    parser.add_argument('--num_labels', type=int, default=2)
    parser.add_argument("--only_test", type=int, default=0, help="only load stat from dir and directly test")
    parser.add_argument("--fold", type=int, default=0)

    parser.add_argument('--epoch_shuffle', type=bool, default=False)
    parser.add_argument("--negative_sample", type=bool, default=False)
    parser.add_argument("--sample_ratio", type=float, default=0.1)

    # pretrain language model path
    parser.add_argument('--plm_path', type=str, default='/data/huzhilei/PLM/bert-base-uncased')

    # ESC, event story line
    parser.add_argument('--dataset_name', type=str, default='EventStoryLine')
    parser.add_argument('--sample_dir', type=str, default='data/EventStoryLine/sent_sample/')

    # amr dictionary path
    parser.add_argument('--vertex_dict', type=str,
                        default='data/EventStoryLine/amr_sample/amr_dict/vertex_dict.txt')
    parser.add_argument('--edge_dict', type=str,
                        default='data/EventStoryLine/amr_sample/amr_dict/edge_dict.txt')
    parser.add_argument('--graph_dir', type=str,
                        default='data/EventStoryLine/amr_sample/amr_graph/')
    parser.add_argument('--align_dir', type=str,
                        default='data/EventStoryLine/amr_sample/amr_align/')
    parser.add_argument('--path_dir', type=str,
                        default='data/EventStoryLine/amr_sample/amr_path/')

    # train setting
    parser.add_argument('--batch_size', type=int, default=20)
    parser.add_argument('--num_train_epochs', type=int, default=50)
    parser.add_argument('--h_dropout', type=float, default=0.5)

    # bert
    parser.add_argument('--bert_learning_rate', type=float, default=1e-5)

    # rgcn
    parser.add_argument('--rgcn_name', type=str, default="rgcn", choices=["rgcn"])
    parser.add_argument('--rgcn_layers', type=int, default=3)
    parser.add_argument('--rgcn_learning_rate', type=float, default=1e-5)
    parser.add_argument("--rgcn_hidden_size", type=int, default=768, help="number of hidden units")
    parser.add_argument("--rgcn_dropout", type=float, default=0.5, help="dropout probability")
    parser.add_argument("--rgcn_bases", type=int, default=16, help="number of bases")
    parser.add_argument('--max_grad_norm', type=float, default=1.0)

    # rnn
    parser.add_argument('--rnn_hidden_size', type=int, default=384)
    parser.add_argument('--rnn_layers', type=int, default=1)
    parser.add_argument('--rnn_bidirectional', type=bool, default=True)

    # focal loss
    parser.add_argument("--gama", type=float, default=2.0)
    parser.add_argument("--alpha", type=float, default=-1)

    return parser
