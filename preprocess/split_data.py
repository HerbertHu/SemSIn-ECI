import os
import pickle
import random
import argparse


def split_train_dev(dataset):
    train_set = []
    dev_set = []

    dev_topic = ['37', '41']
    for topic in dataset.keys():
        if topic in dev_topic:
            dev_set.extend(dataset[topic])
        else:
            train_set.extend(dataset[topic])
    return train_set, dev_set


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_name', type=str, default="EventStoryLine")
    args = parser.parse_args()

    seed = 6688
    random.seed(seed)

    project_path = os.path.abspath('..')
    print("Project path: {}".format(project_path))
    data_path = os.path.join(project_path, 'data', args.dataset_name)

    dir_list = ['sent_sample']
    for dir in dir_list:
        dir_abs = os.path.join(data_path, dir)
        if not os.path.exists(dir_abs):
            os.makedirs(dir_abs)

    # split data set
    with open(os.path.join(data_path, 'data_samples.pkl'), 'rb') as f:
        data = pickle.load(f)
    train_set, dev_set = split_train_dev(data)

    # data shuffle
    random.shuffle(train_set)

    with open(os.path.join(data_path, 'sent_sample/train.pkl'), 'wb') as f:
        pickle.dump(train_set, f)
    with open(os.path.join(data_path, 'sent_sample/dev.pkl'), 'wb') as f:
        pickle.dump(dev_set, f)
    print('split finish')
