import os
import pickle
import random
import argparse

from sklearn.model_selection import KFold


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_name', type=str, default="EventStoryLine")
    args = parser.parse_args()

    project_path = os.path.abspath('..')
    print("Project path: {}".format(project_path))
    data_path = os.path.join(project_path, 'data', args.dataset_name)

    dir_list = ['sent_sample/5fold']
    for dir in dir_list:
        dir_abs = os.path.join(data_path, dir)
        if not os.path.exists(dir_abs):
            os.makedirs(dir_abs)

    with open(os.path.join(data_path, 'data_samples.pkl'), 'rb') as f:
        documents = pickle.load(f)

    cv_data_set = {}
    for topic in documents.keys():
        if topic not in ['37', '41']:
            cv_data_set[topic] = documents[topic]

    # sort by topic id
    ids = sorted(cv_data_set.keys(), key=lambda x: int(x))
    kfold = KFold(n_splits=5)
    for fold, (train_topic_ids, dev_topic_ids) in enumerate(kfold.split(ids)):
        print('fold', fold)

        train_set = []
        test_set = []
        for train_id in train_topic_ids:
            train_set.extend(cv_data_set[ids[train_id]])
        for dev_id in dev_topic_ids:
            test_set.extend(cv_data_set[ids[dev_id]])

        random.seed(6688)
        random.shuffle(train_set)
        with open(os.path.join(data_path, 'sent_sample/5fold', f'train_{fold}.pkl'), 'wb') as f:
            pickle.dump(train_set, f)
        with open(os.path.join(data_path, 'sent_sample/5fold', f'test_{fold}.pkl'), 'wb') as f:
            pickle.dump(test_set, f)

    print('finish')
