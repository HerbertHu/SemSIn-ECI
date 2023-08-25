import os
import collections
import pickle
import argparse

from lxml import etree


def create_folder(filepath):
    directory = os.path.dirname(filepath)
    if not os.path.exists(directory):
        os.makedirs(directory)


def check_entry_dict(event_tokens, d):
    if event_tokens in d:
        return " ".join(d[event_tokens])
    else:
        return event_tokens


def get_sentence(num, all_token):
    temp = []
    for token in all_token:
        if token[1] == num:
            temp.append(token[-1])
    return temp


def transfter_to_token(s, all_token):
    tmp = []
    for c in s.split('_'):
        token = all_token[int(c) - 1]
        tmp.append(token[-1])

    return ' '.join(tmp)


def get_sentence_num(s, all_token):
    c = s.split('_')[0]
    return all_token[int(c) - 1][1]


def generate_feature(s, t, value, all_token):
    s_text = transfter_to_token(s, all_token)
    t_text = transfter_to_token(t, all_token)
    s_sentence = get_sentence(get_sentence_num(s, all_token), all_token)
    t_sentence = get_sentence(get_sentence_num(t, all_token), all_token)

    return s_text, t_text


def all_tokens(filename):
    ecbplus = etree.parse(filename, etree.XMLParser(remove_blank_text=True))
    root_ecbplus = ecbplus.getroot()
    root_ecbplus.getchildren()

    all_token = []

    for elem in root_ecbplus.findall('token'):
        temp = (elem.get('t_id'), elem.get('sentence'),
                elem.get('number'), elem.text)
        all_token.append(temp)
    return all_token


def extract_event_CAT(etreeRoot):
    """
    :param etreeRoot: ECB+/ESC XML root
    :return: dictionary with annotaed events in ECB+
    """

    # 创建默认字典
    event_dict = collections.defaultdict(list)

    # exclude aspectual, causative, perception and reporting event mentions

    exclude_list = ['ACTION_ASPECTUAL', 'NEG_ACTION_ASPECTUAL',
                    'ACTION_CAUSATIVE', 'NEG_ACTION_CAUSATIVE',
                    'ACTION_PERCEPTION', 'NEG_ACTION_PERCEPTION',
                    'ACTION_REPORTING', 'NEG_ACTION_REPORTING']
    for elem in etreeRoot.findall('Markables/'):
        if elem.tag.startswith("ACTION") or elem.tag.startswith("NEG_ACTION"):
            if elem.tag not in exclude_list:
                for token_id in elem.findall('token_anchor'):  # the event should have at least one token
                    event_mention_id = elem.get('m_id', 'nothing')
                    token_mention_id = token_id.get('t_id', 'nothing')
                    event_dict[event_mention_id].append(token_mention_id)

    return event_dict


def extract_time_CAT(etreeRoot):
    time_dict = collections.defaultdict(list)

    for elem in etreeRoot.findall('Markables/'):
        if elem.tag.startswith("TIME_DATE"):
            for token_id in elem.findall('token_anchor'):  # the event should have at least one token
                event_mention_id = elem.get('m_id', 'nothing')
                token_mention_id = token_id.get('t_id', 'nothing')
                time_dict[event_mention_id].append(token_mention_id)

    return time_dict


def extract_corefRelations(etreeRoot, d):
    """
    :param etreeRoot: ECB+ XML root
    :return: dictionary with annotaed events in ECB+ (event_dict)
    :return:
    """

    relations_dict_appo = collections.defaultdict(list)
    relations_dict = {}

    # many source one target
    for elem in etreeRoot.findall('Relations/'):
        target_element = elem.find('target').get('m_id', 'null')  # the target is a non-event
        for source in elem.findall('source'):
            source_elem = source.get('m_id', 'null')
            if source_elem in d:
                val = "_".join(d[source_elem])
                relations_dict_appo[target_element].append(val)  # coreferential sets of events

    for k, v in relations_dict_appo.items():
        for i in v:
            relations_dict[i] = v

    # mention to mention, '65_66':['65_66', '32_33']
    return relations_dict


def extract_plotLink(etreeRoot, d):
    """
    :param etreeRoot: ESC XML root
    :param d: dictionary with annotaed events in ESC (event_dict)
    :return:
    """

    plot_dict = collections.defaultdict(list)

    for elem in etreeRoot.findall('Relations/'):
        if elem.tag == "PLOT_LINK":
            source_pl = elem.find('source').get('m_id', 'null')
            target_pl = elem.find('target').get('m_id', 'null')
            relvalu = elem.get('relType', 'null')

            if source_pl in d:
                val1 = "_".join(d[source_pl])
                if target_pl in d:
                    val2 = "_".join(d[target_pl])
                    plot_dict[(val1, val2)] = relvalu

    # mention_token_id_1, mention_token_id_2, realtion
    return plot_dict


def extract_timeLink(etreeRoot, d):
    tlink_dict = collections.defaultdict(list)

    for elem in etreeRoot.findall('Relations/'):
        if elem.tag == "TLINK":
            try:
                source_pl = elem.find('source').get('m_id', 'null')
                target_pl = elem.find('target').get('m_id', 'null')
            except:
                continue
            relvalu = elem.get('relType', 'null')

            if source_pl in d:
                val1 = "_".join(d[source_pl])
                if target_pl in d:
                    val2 = "_".join(d[target_pl])
                    tlink_dict[(val1, val2)] = relvalu

    # time link, mention_1_tokens, mention_2_tokens : relation
    return tlink_dict


def read_evaluation_file(fn):
    res = []
    if not os.path.exists(fn):
        return res
    for line in open(fn):
        fileds = line.strip().split('\t')
        res.append(fileds)
    return res


def read_file(ecbplus_original, ecbstart_new, evaluate_file, evaluate_coref_file):
    """
    :param ecbplus_original: ECB+ CAT data
    :param ecbstart_new: ESC CAT data
    :param outfile1: event mention extended
    :param outfile2: event extended coref chain
    :return:
    """

    ecbplus = etree.parse(ecbplus_original, etree.XMLParser(remove_blank_text=True))
    root_ecbplus = ecbplus.getroot()

    root_ecbplus.getchildren()

    ecb_event_mentions = extract_event_CAT(root_ecbplus)  # dict, m_id : [t_id1, t_id2]
    ecb_coref_relations = extract_corefRelations(root_ecbplus, ecb_event_mentions)

    """
    ecbstar data
    """

    ecbstar = etree.parse(ecbstart_new, etree.XMLParser(remove_blank_text=True))
    ecbstar_root = ecbstar.getroot()
    ecbstar_root.getchildren()

    ecb_star_events = extract_event_CAT(ecbstar_root)
    ecbstar_events_plotLink = extract_plotLink(ecbstar_root, ecb_star_events)

    # time
    ecb_star_time = extract_time_CAT(ecbstar_root)
    ecb_star_time.update(ecb_star_events)  # 在时间基础上加入事件
    ecbstar_timelink = extract_timeLink(ecbstar_root, ecb_star_time)

    evaluation_data = read_evaluation_file(evaluate_file)
    evaluationcrof_data = read_evaluation_file(evaluate_coref_file)
    # TLINK ??

    # print(ecb_star_events) # all the events
    # print(ecb_star_time) # all the time expressions
    # print(ecbstar_events_plotLink) # direct event plot link
    # print(ecbstar_timelink)

    return ecb_star_events, ecb_coref_relations, ecb_star_time, ecbstar_events_plotLink, ecbstar_timelink, evaluation_data, evaluationcrof_data


def make_corpus(ecbtopic, ecbstartopic, evaluationtopic, evaluationcoreftopic, datadict):
    """
    :param ecbtopic: ECB+ topic folder in CAT format
    :param ecbstartopic: ESC topic folder in CAT format
    :param outdir: output folder for evaluation data format
    :return:
    """

    if os.path.isdir(ecbtopic) and os.path.isdir(ecbstartopic) and os.path.isdir(evaluationtopic):
        if ecbtopic[-1] != '/':
            ecbtopic += '/'
        if ecbstartopic[-1] != '/':
            ecbstartopic += '/'
        if evaluationtopic[-1] != '/':
            evaluationtopic += '/'
        if evaluationcoreftopic[-1] != '/':
            evaluationcoreftopic += '/'

        ecb_subfolder = os.path.dirname(ecbtopic).split("/")[-1]

        f_list = os.listdir(ecbtopic)
        f_list.sort()
        for f in f_list:
            if f.endswith('plus.xml'):
                ecb_file = f
                star_file = ecbstartopic + f + ".xml"
                evaluate_file = evaluationtopic + f
                evaluate_coref_file = evaluationcoreftopic + f

                #  coref_chain and event_mentions_extended file in evaluation is end with .tab
                evaluate_file = evaluate_file.rstrip('.xml') + '.tab'
                evaluate_coref_file = evaluate_coref_file.rstrip('.xml') + '.tab'

                ecb_star_events, ecb_coref_relations, ecb_star_time, ecbstar_events_plotLink, ecbstar_timelink, evaluation_data, evaluationcrof_data = read_file(
                    ecbtopic + ecb_file, star_file, evaluate_file, evaluate_coref_file)
                for key in ecb_star_events:
                    ecb_star_events[key] = '_'.join(ecb_star_events[key])
                for key in ecb_star_time:
                    ecb_star_time[key] = '_'.join(ecb_star_time[key])
                all_token = all_tokens(star_file)

                file_name = star_file.split('raw_data/')[-1]
                # alltoken, event_mention, mention's coref relation, time and event mention, event mention plotlink,
                # mention tlink, evaluation_data, crof_data
                datadict[file_name] = [all_token, ecb_star_events, ecb_coref_relations, ecb_star_time,
                                       ecbstar_events_plotLink, ecbstar_timelink, evaluation_data, evaluationcrof_data]
                # for elem in ecbstar_events_plotLink:
                #     (s, t), value = elem, ecbstar_events_plotLink[elem]
                #     s_text, t_text = generate_feature(s, t, value, all_token)
                #     print(s_text, t_text, value)


def main(argv=None):
    project_path = os.path.abspath('..')
    print(project_path)

    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_name', type=str, default="EventStoryLine")
    args = parser.parse_args()

    version = 'v0.9'
    ECBplusTopic = project_path + '/data/raw_data/EventStoryLine/ECB+_LREC2014/ECB+/'
    ECBstarTopic = project_path + '/data/raw_data/EventStoryLine/annotated_data/' + version + '/'
    EvaluationTopic = project_path + '/data/raw_data/EventStoryLine/evaluation_format/full_corpus/' + version + '/event_mentions_extended/'
    EvaluationCrofTopic = project_path + '/data/raw_data/EventStoryLine/evaluation_format/full_corpus/' + version + '/coref_chain/'

    data_dict = {}
    topic_list = os.listdir(project_path + '/data/raw_data/EventStoryLine/annotated_data/' + version + '/')
    topic_list.sort()
    for topic in topic_list:
        if os.path.isdir(project_path + '/data/raw_data/EventStoryLine/annotated_data/' + version + '/' + topic):
            dir1, dir2, dir3, dir4 = ECBplusTopic + topic, ECBstarTopic + topic, EvaluationTopic + topic, EvaluationCrofTopic + topic
            make_corpus(dir1, dir2, dir3, dir4, data_dict)

    for key in data_dict:
        print(key)

    data_dir = os.path.join(project_path, 'data', args.dataset_name)
    if not os.path.exists(data_dir):
        os.mkdir(data_dir)
    with open(os.path.join(project_path, 'data', args.dataset_name, 'document_raw.pkl'), 'wb') as f:
        pickle.dump(data_dict, f)

    # data = data_dict['EventStoryLine/annotated_data/v0.9/14/14_2ecbplus.xml.xml']
    # data[0] -> all_token
    # ecb_star_events
    # ecb_coref_relations
    # ecb_star_time
    # ecbstar_events_plotLink
    # ecbstar_timelink
    # evaluation_data
    # evaluationcrof_data


if __name__ == '__main__':
    main()
