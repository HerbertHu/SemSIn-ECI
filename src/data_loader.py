import torch
import numpy as np
import random
import pickle
import os
import copy
import dgl
import networkx as nx

from tqdm import tqdm
from transformers import BertTokenizer

from utils.logger import get_logger

logger = get_logger()


# from allennlp.nn.util
def get_mask_from_sequence_lengths(
        sequence_lengths: torch.Tensor, max_length: int
) -> torch.BoolTensor:
    """
    Given a variable of shape `(batch_size,)` that represents the sequence lengths of each batch
    element, this function returns a `(batch_size, max_length)` mask variable.  For example, if
    our input was `[2, 2, 3]`, with a `max_length` of 4, we'd return
    `[[1, 1, 0, 0], [1, 1, 0, 0], [1, 1, 1, 0]]`.

    We require `max_length` here instead of just computing it from the input `sequence_lengths`
    because it lets us avoid finding the max, then copying that value from the GPU to the CPU so
    that we can use it to construct a new tensor.
    """
    # (batch_size, max_length)
    ones = sequence_lengths.new_ones(sequence_lengths.size(0), max_length)
    range_tensor = ones.cumsum(dim=1)
    return sequence_lengths.unsqueeze(1) >= range_tensor


# from allennlp.common.util
def pad_sequence_to_length(
        sequence: list,
        desired_length: int,
        default_value: int = 0,
        padding_on_right: bool = True,
) -> list:
    """
    Take a list of objects and pads it to the desired length, returning the padded list.  The
    original list is not modified.

    # Parameters

    sequence : `List`
        A list of objects to be padded.

    desired_length : `int`
        Maximum length of each sequence. Longer sequences are truncated to this length, and
        shorter ones are padded to it.

    default_value: `Callable`, optional (default=`lambda: 0`)
        Callable that outputs a default value (of any type) to use as padding values.  This is
        a lambda to avoid using the same object when the default value is more complex, like a
        list.

    padding_on_right : `bool`, optional (default=`True`)
        When we add padding tokens (or truncate the sequence), should we do it on the right or
        the left?

    # Returns

    padded_sequence : `List`
    """
    sequence = list(sequence)
    # Truncates the sequence to the desired length.
    if padding_on_right:
        padded_sequence = sequence[:desired_length]
    else:
        padded_sequence = sequence[-desired_length:]
    # Continues to pad with default_value() until we reach the desired length.
    pad_length = desired_length - len(padded_sequence)
    # This just creates the default value once, so if it's a list, and if it gets mutated
    # later, it could cause subtle bugs. But the risk there is low, and this is much faster.
    values_to_pad = [default_value] * pad_length
    if padding_on_right:
        padded_sequence = padded_sequence + values_to_pad
    else:
        padded_sequence = values_to_pad + padded_sequence
    return padded_sequence


def load_amr_graphs(num_ents, num_rels, graph_base_dir, mode):
    amr_graph = os.path.join(graph_base_dir, mode + ".pkl")
    # for the first run
    if os.path.exists(amr_graph):
        logger.info("Loading dgl graphs from {}".format(amr_graph))
        with open(amr_graph, "rb") as f:
            graph_list = pickle.load(f)
    else:
        logger.info("Generate and save dgl graphs to {}".format(amr_graph))
        graph_list = []

        with open(os.path.join(graph_base_dir, mode + ".txt"), 'r') as f:
            res_list = f.read().split('\n\n\n')
            for res in res_list:
                triples = []
                for line in res.split('\n'):
                    try:
                        h, r, t = eval(line.strip())
                        triples.append((h, r, t))
                    except:
                        print("Read triple data error!")
                        print(line)
                graph = build_sub_graph(num_ents, num_rels, np.array(triples))
                graph_list.append(graph)
        with open(amr_graph, "wb") as f:
            pickle.dump(graph_list, f)
    return graph_list


def build_sub_graph(num_nodes, num_rels, triples):
    """
    build subgraph as dgl object
    :param num_nodes: the number of node in the graph
    :param num_rels: the number of relations in the graph
    :param triples: the triples in the graph
    :return:
    """

    def comp_deg_norm(g):
        # indegrees normaliztion, if indegree is 0, set indegree is 1
        in_deg = g.in_degrees(range(g.number_of_nodes())).float()
        in_deg[torch.nonzero(in_deg == 0).view(-1)] = 1
        norm = 1.0 / in_deg
        return norm

    src, rel, dst = triples.transpose()
    src, dst = np.concatenate((src, dst)), np.concatenate((dst, src))  # add reverse, the edge is bidirectional
    rel = np.concatenate((rel, rel + num_rels))

    # edges is indices to reconstruct array
    uniq_v, edges = np.unique((src, dst), return_inverse=True)
    src, dst = np.reshape(edges, (2, -1))

    g = dgl.DGLGraph()
    g.add_nodes(len(uniq_v))
    g.add_edges(src, dst)  # node id is converted to index, index -> index is edges
    norm = comp_deg_norm(g)
    g.ndata.update({'id': torch.from_numpy(uniq_v).long().view(-1, 1), 'norm': norm.view(-1, 1)})
    g.apply_edges(lambda edges: {'norm': edges.dst['norm'] * edges.src['norm']})
    g.edata['type'] = torch.LongTensor(rel)

    g.ids = {}
    for idx, idd in enumerate(uniq_v):
        g.ids[idd] = idx
    # if use_cuda:
    #     g.to(gpu)
    return g


def load_dict(file_path):
    with open(file_path, 'r') as f:
        dict_ = {line.rstrip('\n'): idx for idx, line in enumerate(f.readlines())}

    return dict_


def load_pkl(file_path):
    with open(file_path, "rb") as f:
        pkl_data = pickle.load(f)

    return pkl_data


def add_reverse_path(sample_paths, rel_num):
    sample_paths_reverse = []
    for path in sample_paths:
        reversed_path = []

        rel_list, node_list = [], []
        for pair in path:
            node_list.append(pair[0])
            rel = pair[1]
            if rel != 'Padr':
                if rel >= rel_num:
                    rel = rel - rel_num
                else:
                    rel = rel + rel_num
                rel_list.append(rel)

        rel_list = list(reversed(rel_list)) + ['Padr']
        node_list = list(reversed(node_list))

        for node, rel in zip(node_list, rel_list):
            reversed_path.append((node, rel))
        sample_paths_reverse.append(reversed_path)

    return sample_paths + sample_paths_reverse


class Dataset(object):
    def __init__(self, args, dataset, graph_list, align_info, path_graph, tokenizer):
        super(Dataset, self).__init__()

        self.batch_size = args.batch_size
        self.y_label = {
            'NULL': 0,
            'null': 1,  # plot link, but no direction
            'plot_null': 1,
            'FALLING_ACTION': 1,
            'PRECONDITION': 1,
            'Coref': 1,
            'Cause-Effect': 1,
            'Cause-Effect1': 1,
            'Cause-Effect2': 1,
            'Cause': 1
        }  # data label

        self.args = args
        self.graph_list = graph_list
        self.align_info = align_info
        self.path_graph = path_graph
        self.tokenizer = tokenizer  # bert tokenizer
        self.index_length = len(dataset)
        self.shuffle_list = list(range(0, self.index_length))

        self.tokenize_data = self.tokenize(dataset)
        self.dataset = dataset

        self.paths_len_list = []

    def reader(self, device, shuffle=False):
        """
        read dataset
        :param device: model used device
        :param shuffle: every epoch shuffle
        :return: None
        """
        cur_idx = 0
        while cur_idx < self.index_length:
            end_index = min(cur_idx + self.batch_size, self.index_length)

            batch_samples = [self.dataset[self.shuffle_list[index]] for index in range(cur_idx, end_index)]
            batch_tokenize = [self.tokenize_data[self.shuffle_list[index]] for index in range(cur_idx, end_index)]
            batch_graphs = [self.graph_list[self.shuffle_list[index]] for index in range(cur_idx, end_index)]
            batch_align = [self.align_info[self.shuffle_list[index]] for index in range(cur_idx, end_index)]
            batch_path = [self.path_graph[self.shuffle_list[index]] for index in range(cur_idx, end_index)]
            cur_idx = end_index
            yield self.batchify(batch_samples, batch_tokenize, batch_graphs, batch_align, batch_path, device)
            # return [self.batchify(batch_samples, batch_tokenize, batch_graphs, batch_align, batch_path, device)]
        if shuffle:
            random.shuffle(self.shuffle_list)
            logger.info("Data shuffle finish.")

    def tokenize(self, batch_samples):
        """
        convert tokens to id
        :param batch_samples: tokens batch data
        :return: tokenized batch data
        """
        batch_tokenize = []
        for item in batch_samples:
            sent_s = copy.deepcopy(item[1])
            span1 = item[2]
            span2 = item[3]

            e1_start = span1[0]
            e1_end = span1[-1] + 1
            e2_start = span2[0]
            e2_end = span2[-1] + 1

            offset = 2 if e1_start < e2_start else 0
            sent_s.insert(e1_start, '<e1>')
            sent_s.insert(e1_end + 1, '</e1>')
            sent_s.insert(e2_start + offset, '<e2>')
            sent_s.insert(e2_end + 1 + offset, '</e2>')
            sent_s = ' '.join(sent_s)

            tokenized_text = self.tokenizer.tokenize(sent_s)
            sent_encoding = self.tokenizer.encode(tokenized_text)
            e1_index = sent_encoding.index(self.tokenizer.convert_tokens_to_ids('<e1>'))
            e2_index = sent_encoding.index(self.tokenizer.convert_tokens_to_ids('<e2>'))
            encoding_text = self.tokenizer.convert_ids_to_tokens(sent_encoding)

            batch_tokenize.append([item[0], sent_encoding, e1_index, e2_index, item[4]])

        return batch_tokenize

    def map_event(self, graph, event_index, align_info):
        node_id_list, node_id_list_origin = [], []
        for index in event_index:
            if index in align_info.keys():
                _id = align_info[index]
                node_id = graph.ids[_id]
                node_id_list.append(node_id)
                node_id_list_origin.append(_id)

        return node_id_list, node_id_list_origin

    def select_map_path(self, paths, graph):
        paths.sort(key=lambda x: len(x), reverse=False)
        shortest_len = len(paths[0])
        map_paths = []
        for path in paths:
            if len(path) == shortest_len:
                temp = []
                for idx, item in enumerate(path):
                    temp.append((graph.ids[item[0]], item[2]))
                    if idx == len(path) - 1:
                        temp.append((graph.ids[item[1]], 'Padr'))
                map_paths.append(temp)
            else:
                break
        return map_paths

    def graph_init(self, graph, sample, align_info, sent_item):
        bert_init_dict = {}
        random_init_dict = {}
        new_graph_id = {v: k for k, v in graph.ids.items()}
        new_align_info = {v: k for k, v in align_info.items()}

        for node_id in new_graph_id.keys():
            dict_id = new_graph_id[node_id]
            if dict_id in new_align_info.keys():
                token_index = new_align_info[dict_id]
                token = sample[1][token_index]
                tokenized_text = self.tokenizer.tokenize(token)
                tokenized_ids = self.tokenizer.convert_tokens_to_ids(tokenized_text)
                embed_index = [sent_item.index(i) for i in tokenized_ids]
                bert_init_dict[node_id] = embed_index
            else:
                random_init_dict[node_id] = dict_id

        return bert_init_dict, random_init_dict

    def batchify(self, batch_samples, batch_tokenize, batch_graphs, batch_align, batch_path, device):
        """
        padding batch data
        :param batch: tokenized batch data
        :param device: model used device
        :return: batch data tensor
        """

        sentence_len_s, event1_lens, event2_lens = [], [], []
        sentences_s, event1, event2, data_y = [], [], [], []
        for data in batch_tokenize:
            sentences_s.append(data[1])
            event1.append(data[2])
            event2.append(data[3])
            sentence_len_s.append((len(data[1])))

            y = self.y_label[data[4]] if data[4] in self.y_label else 0
            data_y.append(y)

        # max sentence length and max event length, used for padding
        max_sentence_len_s = max(sentence_len_s)
        # padding
        sentences_s = list(map(lambda x: pad_sequence_to_length(x, max_sentence_len_s), sentences_s))
        mask_sentences_s = get_mask_from_sequence_lengths(torch.LongTensor(sentence_len_s), max_sentence_len_s)

        # graph data
        graphs_len = [len(g.ndata['id']) for g in batch_graphs]
        graph_event1_list, graph_event2_list = [], []
        graph_event1_lens, graph_event2_lens = [], []
        paths_list, paths_len_list = [], []
        bert_init_list, random_init_list = [], []
        for graph, sample, align_info, g_path, sent_item in zip(
                batch_graphs, batch_samples, batch_align, batch_path, sentences_s):
            # get gcn bert init index
            bert_init_dict, random_init_dict = self.graph_init(graph, sample, align_info, sent_item)
            bert_init_list.append(bert_init_dict)
            random_init_list.append(random_init_dict)

            # get graph path
            event1_index = sample[2]
            event2_index = sample[3]
            try:
                graph_event1, graph_event1_origin = self.map_event(graph, event1_index, align_info)
                graph_event2, graph_event2_origin = self.map_event(graph, event2_index, align_info)
            except:
                print('map event error', sample[0])

            graph_event1_list.append(graph_event1)
            graph_event2_list.append(graph_event2)
            graph_event1_lens.append(len(graph_event1))
            graph_event2_lens.append(len(graph_event2))

            if len(graph_event1_origin) == 0 or len(graph_event2_origin) == 0:
                paths_len_list.append(0)
                continue
            sample_path_list = []
            for begin in graph_event1_origin:
                for end in graph_event2_origin:
                    if begin == end:
                        # abnormal sample num is 7 * 2 = 14
                        # logger.warning("begin and end node is same, num is {}".format(begin))
                        continue
                    all_paths_list = [p for p in nx.all_simple_edge_paths(g_path, begin, end)]
                    short_paths_list = self.select_map_path(all_paths_list, graph)
                    sample_path_list.extend(short_paths_list)
            if len(sample_path_list) == 0:
                paths_len_list.append(0)
                continue

            sample_short_paths = add_reverse_path(sample_path_list, self.args.num_rels)
            paths_list.extend(sample_short_paths)
            paths_len_list.append(len(sample_short_paths))

        graph_max_event_lens = max(graph_event1_lens + graph_event2_lens)
        # event padding max max_event_lens
        graph_event1_list = list(map(lambda x: pad_sequence_to_length(x, graph_max_event_lens), graph_event1_list))
        graph_event2_list = list(map(lambda x: pad_sequence_to_length(x, graph_max_event_lens), graph_event2_list))

        # if event len = 3, max = 5, return[[1, 1, 1, 0, 0]]
        graph_event1_mask = get_mask_from_sequence_lengths(torch.LongTensor(graph_event1_lens), graph_max_event_lens)
        graph_event2_mask = get_mask_from_sequence_lengths(torch.LongTensor(graph_event2_lens), graph_max_event_lens)

        return [torch.LongTensor(sentences_s).to(device), mask_sentences_s.to(device),
                torch.LongTensor(event1).to(device), None,
                torch.LongTensor(event2).to(device), None,
                batch_graphs, graphs_len,
                torch.LongTensor(graph_event1_list).to(device),
                graph_event1_mask.to(device),
                torch.LongTensor(graph_event2_list).to(device),
                graph_event2_mask.to(device),
                torch.LongTensor(data_y).to(device),
                paths_list, paths_len_list,
                bert_init_list, random_init_list
                ]
