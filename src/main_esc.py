import os
import pickle
import random
import dgl
import numpy as np
import torch

from transformers import BertTokenizer
from arguments_esc import get_parser
from data_loader import Dataset, load_dict, load_pkl
from framework import Framework
from models.bert_event import BertEvent
from models.gcn_event import GCNEvent

from utils.sampling import negative_sampling
from utils.logger import get_logger


def set_seed(seed):
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.enabled = False
    np.random.seed(seed)
    random.seed(seed)

    dgl.seed(seed)
    dgl.random.seed(seed)


if __name__ == "__main__":
    parser = get_parser()
    args = parser.parse_args()
    seed = args.random_seed
    set_seed(seed)
    logger = get_logger()
    logger.info("Set seed: {}".format(seed))

    args.save_model_name = f"{args.save_model}_bert_fold_{args.fold}.pt"
    args.save_model_name2 = f"{args.save_model}_gcn_fold_{args.fold}.pt"
    args.save_model_name3 = f"{args.save_model}_init_fold_{args.fold}.pt"

    # get current path
    file_dir = os.path.dirname(__file__)
    args.project_path = os.path.abspath(os.path.join(file_dir, '..'))
    logger.info("project_path: {}".format(args.project_path))

    # create save_model_dir
    for file_dir in [args.save_model_dir, args.output_dir]:
        file_dir = os.path.join(args.project_path, file_dir)
        if not os.path.exists(file_dir):
            os.mkdir(file_dir)

    args.save_model_path = os.path.join(args.project_path, args.save_model_dir)
    logger.info("save_model_path: {}".format(args.save_model_path))

    # set device
    args.device = torch.device("cuda:{}".format(args.device_num) if torch.cuda.is_available() else "cpu")
    # args.device = torch.device("cpu")
    logger.info("Device: {}".format(args.device))

    logger.info("Load data from fold {}".format(args.fold))
    # load sample data
    train_file = os.path.join(args.project_path, args.sample_dir, '5fold', f'train_{args.fold}.pkl')
    dev_file = os.path.join(args.project_path, args.sample_dir, 'dev.pkl')
    test_file = os.path.join(args.project_path, args.sample_dir, '5fold', f'test_{args.fold}.pkl')
    train_set = load_pkl(train_file)
    dev_set = load_pkl(dev_file)
    test_set = load_pkl(test_file)

    # read ent and rel dict
    ent2id = load_dict(os.path.join(args.project_path, args.vertex_dict))
    rel2id = load_dict(os.path.join(args.project_path, args.edge_dict))
    args.num_ents = len(ent2id.keys())
    args.num_rels = len(rel2id.keys())

    # load amr graphs
    graph_train = os.path.join(args.project_path, args.graph_dir, '5fold', f'train_{args.fold}.pkl')
    graph_dev = os.path.join(args.project_path, args.graph_dir, 'dev.pkl')
    graph_test = os.path.join(args.project_path, args.graph_dir, '5fold', f'test_{args.fold}.pkl')
    train_graphs = load_pkl(graph_train)
    dev_graphs = load_pkl(graph_dev)
    test_graphs = load_pkl(graph_test)

    # load align info
    align_train = os.path.join(args.project_path, args.align_dir, '5fold', f'train_{args.fold}.pkl')
    align_dev = os.path.join(args.project_path, args.align_dir, 'dev.pkl')
    align_test = os.path.join(args.project_path, args.align_dir, '5fold', f'test_{args.fold}.pkl')
    train_align_info = load_pkl(align_train)
    dev_align_info = load_pkl(align_dev)
    test_align_info = load_pkl(align_test)

    # load path graph
    path_train = os.path.join(args.project_path, args.path_dir, '5fold', f'train_{args.fold}.pkl')
    path_dev = os.path.join(args.project_path, args.path_dir, 'dev.pkl')
    path_test = os.path.join(args.project_path, args.path_dir, '5fold', f'test_{args.fold}.pkl')
    train_data_path = load_pkl(path_train)
    dev_data_path = load_pkl(path_dev)
    test_data_path = load_pkl(path_test)

    assert (len(train_set) == len(train_graphs))
    assert (len(train_set) == len(train_align_info))
    assert (len(train_set) == len(train_data_path))

    if args.negative_sample:
        train_set, train_graphs, train_align_info, train_data_path = \
            negative_sampling(train_set, train_graphs, train_align_info, train_data_path, ratio=args.sample_ratio)

    logger.info("The number of train set is {}".format(len(train_set)))
    logger.info("The number of test set is {}".format(len(test_set)))

    tokenizer = BertTokenizer.from_pretrained(args.plm_path, do_lower_case=True)
    ADDITIONAL_SPECIAL_TOKENS = ["<e1>", "</e1>", "<e2>", "</e2>"]
    tokenizer.add_special_tokens({"additional_special_tokens": ADDITIONAL_SPECIAL_TOKENS})

    # set model
    logger.info("Model is {}.".format(args.model))

    model_bert = BertEvent(args).to(args.device)
    model_bert.bert.resize_token_embeddings(len(tokenizer))
    model_init = BertEvent(args).to(args.device)
    model_init.bert.resize_token_embeddings(len(tokenizer))

    model_gcn = GCNEvent(args).to(args.device)

    # create training framework
    framework = Framework(args)
    logger.info(args)

    logger.info("Loading test datasets ...")
    test_dataset = Dataset(args, test_set, test_graphs, test_align_info, test_data_path, tokenizer)
    test_dataset_batch = [batch for batch in test_dataset.reader(args.device)]

    if not args.only_test:
        logger.info("Loading train dataset ...")
        train_dataset = Dataset(args, train_set, train_graphs, train_align_info, train_data_path, tokenizer)
        framework.train(train_dataset, test_dataset_batch, model_bert, model_gcn, model_init)

    logger.info("Loading best model ...")
    model_bert.load_state_dict(
        torch.load(os.path.join(args.save_model_path, args.save_model_name), map_location=args.device))
    model_gcn.load_state_dict(
        torch.load(os.path.join(args.save_model_path, args.save_model_name2), map_location=args.device))
    model_init.load_state_dict(
        torch.load(os.path.join(args.save_model_path, args.save_model_name3), map_location=args.device))

    precision, recall, f1 = framework.evaluate(test_dataset_batch, model_bert, model_gcn, model_init)
    logger.info("Precision: {:.3f}, recall: {:.3f}, f1: {:.3f}\n".format(precision, recall, f1))
