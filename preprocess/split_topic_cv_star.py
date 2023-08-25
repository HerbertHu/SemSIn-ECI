import os
import pickle
import random

from sklearn.model_selection import KFold


def get_sent(dataset):
    sent_list = []
    for item in dataset:
        sent = ' '.join(item[1])
        sent_list.append(sent)

    return sent_list


if __name__ == '__main__':
    project_path = os.path.abspath('..')
    print("Project path: {}".format(project_path))
    data_path = os.path.join(project_path, 'data/EventStoryLine_star')

    dir_list = ['sent_sample/5fold']
    for dir in dir_list:
        dir_abs = os.path.join(data_path, dir)
        if not os.path.exists(dir_abs):
            os.makedirs(dir_abs)

    with open(os.path.join(data_path, 'data_samples.pkl'), 'rb') as f:
        documents = pickle.load(f)

    cv_data_set = {}
    for doc_name in documents.keys():
        topic = doc_name.split('_')[0]
        if topic not in ['37', '41']:
            cv_data_set[doc_name] = documents[doc_name]

    # ids = sorted(cv_data_set.keys(), key=lambda x: int(x))
    ids = list(cv_data_set.keys())
    random.seed(6688)
    random.shuffle(ids)
    kfold = KFold(n_splits=5)

    for fold, (train_doc_ids, dev_doc_ids) in enumerate(kfold.split(ids)):
        print('fold', fold)

        train_set = []
        dev_set =[]
        for train_id in train_doc_ids:
            train_set.extend(cv_data_set[ids[train_id]])
        for dev_id in dev_doc_ids:
            dev_set.extend(cv_data_set[ids[dev_id]])

        random.shuffle(train_set)
        with open(os.path.join(data_path, 'sent_sample/5fold', f'train_{fold}.pkl'), 'wb') as f:
            pickle.dump(train_set, f)
        with open(os.path.join(data_path, 'sent_sample/5fold', f'test_{fold}.pkl'), 'wb') as f:
            pickle.dump(dev_set, f)

    print('finish')
