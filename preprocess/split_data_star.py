import os
import pickle
import random


def split_train_dev(dataset):
    train_set = []
    dev_set = []

    dev_topic = ['37', '41']
    for doc_name in dataset.keys():
        topic = doc_name.split('_')[0]
        if topic in dev_topic:
            dev_set.extend(dataset[doc_name])
        else:
            train_set.extend(dataset[doc_name])
    return train_set, dev_set


def get_sent(dataset):
    sent_list = []
    for item in dataset:
        sent = ' '.join(item[1])
        sent_list.append(sent)

    return sent_list


if __name__ == '__main__':
    seed = 6688
    random.seed(seed)

    project_path = os.path.abspath('..')
    print("Project path: {}".format(project_path))
    data_path = os.path.join(project_path, 'data/EventStoryLine_star')

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
