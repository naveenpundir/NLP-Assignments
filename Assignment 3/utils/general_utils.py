import pickle

def get_pickle(path):
    data = pickle.load(open(path, "rb"))
    return data


def dump_pickle(data, path):
    with open(path, "wb") as f:
        pickle.dump(data, f)


def get_vocab_dict(items):
    item2idx = {}
    idx = 0
    for item in items:
        item2idx[item] = idx
        idx += 1
    return item2idx