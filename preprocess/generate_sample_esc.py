import os
import pickle
import itertools
from collections import defaultdict


def get_sentence_number(s, all_token):
    tid = s.split('_')[0]
    for token in all_token:
        if token[0] == tid:
            return token[1]


def nth_sentence(sen_no):
    res = []
    for token in all_token:
        if token[1] == sen_no:
            res.append(token[-1])
    return res


# sentence inside offset
def get_sentence_offset(s, all_token):
    positions = []
    for c in s.split('_'):
        token = all_token[int(c) - 1]
        positions.append(token[2])
    return '_'.join(positions)


if __name__ == '__main__':
    project_path = os.path.abspath('..')
    print("Project path: {}".format(project_path))
    data_path = os.path.join(project_path, 'data/EventStoryLine')

    version = 'v0.9'
    with open(os.path.join(data_path, 'document_raw.pkl'), 'rb') as f:
        documents = pickle.load(f)

    index = 0
    data_set = defaultdict(list)
    event_num = 0
    for doc in documents:
        [all_token, ecb_star_events, ecb_coref_relations,
         ecb_star_time, ecbstar_events_plotLink, ecbstar_timelink,
         evaluation_data, evaluationcrof_data] = documents[doc]

        event_num += len(ecb_star_events)
        doc_name = doc.split('/')[-1].rstrip('.xml.xml')
        print('doc', doc_name)
        topic = doc_name.split('_')[0]

        # not consider the relation direction
        event_combination = itertools.combinations(ecb_star_events.values(), 2)  # unidirection
        doc_data = []
        for item in event_combination:
            offset1 = item[0]
            offset2 = item[1]

            # Causal Relation
            rel = 'NULL'
            for elem in evaluation_data:
                e1, e2, value = elem
                if e1 == offset1 and e2 == offset2:
                    rel = value
                elif e2 == offset1 and e1 == offset2:
                    rel = value

            sen_s = get_sentence_number(offset1, all_token)
            sen_t = get_sentence_number(offset2, all_token)

            # same sentence
            if abs(int(sen_s) - int(sen_t)) == 0:
                sentence_s = nth_sentence(sen_s)

                sen_offset1 = get_sentence_offset(offset1, all_token)
                sen_offset2 = get_sentence_offset(offset2, all_token)

                span1 = [int(x) for x in sen_offset1.split('_')]
                span2 = [int(x) for x in sen_offset2.split('_')]

                doc_data.append([index, sentence_s, span1, span2, rel, doc_name])
                index += 1

        data_set[topic].extend(doc_data)

    print('topic num:', len(data_set))  # 22
    print('doc num:', len(documents))  # 258
    print('event num:', event_num)  # 4695

    with open(os.path.join(data_path, 'data_samples.pkl'), 'wb') as f:
        pickle.dump(data_set, f)
