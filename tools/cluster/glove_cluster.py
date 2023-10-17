# import os
# import torch
# import sys
# import zipfile
# import array
# import six
# from tqdm import tqdm
# from six.moves.urllib.request import urlretrieve
import numpy as np

from sklearn.cluster import AgglomerativeClustering


ID2NAME = {
    # "0": "__background__",
    "1": "airplane",
    "2": "animal",
    "3": "arm",
    "4": "bag",
    "5": "banana",
    "6": "basket",
    "7": "beach",
    "8": "bear",
    "9": "bed",
    "10": "bench",
    "11": "bike",
    "12": "bird",
    "13": "board",
    "14": "boat",
    "15": "book",
    "16": "boot",
    "17": "bottle",
    "18": "bowl",
    "19": "box",
    "20": "boy",
    "21": "branch",
    "22": "building",
    "23": "bus",
    "24": "cabinet",
    "25": "cap",
    "26": "car",
    "27": "cat",
    "28": "chair",
    "29": "child",
    "30": "clock",
    "31": "coat",
    "32": "counter",
    "33": "cow",
    "34": "cup",
    "35": "curtain",
    "36": "desk",
    "37": "dog",
    "38": "door",
    "39": "drawer",
    "40": "ear",
    "41": "elephant",
    "42": "engine",
    "43": "eye",
    "44": "face",
    "45": "fence",
    "46": "finger",
    "47": "flag",
    "48": "flower",
    "49": "food",
    "50": "fork",
    "51": "fruit",
    "52": "giraffe",
    "53": "girl",
    "54": "glass",
    "55": "glove",
    "56": "guy",
    "57": "hair",
    "58": "hand",
    "59": "handle",
    "60": "hat",
    "61": "head",
    "62": "helmet",
    "63": "hill",
    "64": "horse",
    "65": "house",
    "66": "jacket",
    "67": "jean",
    "68": "kid",
    "69": "kite",
    "70": "lady",
    "71": "lamp",
    "72": "laptop",
    "73": "leaf",
    "74": "leg",
    "75": "letter",
    "76": "light",
    "77": "logo",
    "78": "man",
    "79": "men",
    "80": "motorcycle",
    "81": "mountain",
    "82": "mouth",
    "83": "neck",
    "84": "nose",
    "85": "number",
    "86": "orange",
    "87": "pant",
    "88": "paper",
    "89": "paw",
    "90": "people",
    "91": "person",
    "92": "phone",
    "93": "pillow",
    "94": "pizza",
    "95": "plane",
    "96": "plant",
    "97": "plate",
    "98": "player",
    "99": "pole",
    "100": "post",
    "101": "pot",
    "102": "racket",
    "103": "railing",
    "104": "rock",
    "105": "roof",
    "106": "room",
    "107": "screen",
    "108": "seat",
    "109": "sheep",
    "110": "shelf",
    "111": "shirt",
    "112": "shoe",
    "113": "short",
    "114": "sidewalk",
    "115": "sign",
    "116": "sink",
    "117": "skateboard",
    "118": "ski",
    "119": "skier",
    "120": "sneaker",
    "121": "snow",
    "122": "sock",
    "123": "stand",
    "124": "street",
    "125": "surfboard",
    "126": "table",
    "127": "tail",
    "128": "tie",
    "129": "tile",
    "130": "tire",
    "131": "toilet",
    "132": "towel",
    "133": "tower",
    "134": "track",
    "135": "train",
    "136": "tree",
    "137": "truck",
    "138": "trunk",
    "139": "umbrella",
    "140": "vase",
    "141": "vegetable",
    "142": "vehicle",
    "143": "wave",
    "144": "wheel",
    "145": "window",
    "146": "windshield",
    "147": "wing",
    "148": "wire",
    "149": "woman",
    "150": "zebra"
}


def reporthook(t):
    """https://github.com/tqdm/tqdm"""
    last_b = [0]
    def inner(b=1, bsize=1, tsize=None):
        if tsize is not None:
            t.total = tsize
        t.update((b - last_b[0]) * bsize)
        last_b[0] = b
    return inner


def load_word_vectors(root, wv_type, dim):
    """Load word vectors from a path, trying .pt, .txt, and .zip extensions."""
    URL = {
        'glove.42B': 'http://nlp.stanford.edu/data/glove.42B.300d.zip',
        'glove.840B': 'http://nlp.stanford.edu/data/glove.840B.300d.zip',
        'glove.twitter.27B': 'http://nlp.stanford.edu/data/glove.twitter.27B.zip',
        'glove.6B': 'http://nlp.stanford.edu/data/glove.6B.zip',
        }
    if isinstance(dim, int):
        dim = str(dim) + 'd'
    fname = os.path.join(root, wv_type + '.' + dim)

    if os.path.isfile(fname + '.pt'):
        fname_pt = fname + '.pt'
        print('loading word vectors from', fname_pt)
        try:
            return torch.load(fname_pt, map_location=torch.device("cpu"))
        except Exception as e:
            print("Error loading the model from {}{}".format(fname_pt, str(e)))
            sys.exit(-1)
    else:
        print("INFO File not found: ", fname + '.pt')
    if not os.path.isfile(fname + '.txt'):
        print("INFO File not found: ", fname + '.txt')
    if os.path.isfile(fname + '.txt'):
        fname_txt = fname + '.txt'
        cm = open(fname_txt, 'rb')
        cm = [line for line in cm]
    elif os.path.basename(wv_type) in URL:
        url = URL[wv_type]
        print('downloading word vectors from {}'.format(url))
        filename = os.path.basename(fname)
        if not os.path.exists(root):
            os.makedirs(root)
        with tqdm(unit='B', unit_scale=True, miniters=1, desc=filename) as t:
            fname, _ = urlretrieve(url, fname, reporthook=reporthook(t))
            with zipfile.ZipFile(fname, "r") as zf:
                print('extracting word vectors into {}'.format(root))
                zf.extractall(root)
        if not os.path.isfile(fname + '.txt'):
            raise RuntimeError('no word vectors of requested dimension found')
        return load_word_vectors(root, wv_type, dim)
    else:
        raise RuntimeError('unable to load word vectors')

    wv_tokens, wv_arr, wv_size = [], array.array('d'), None
    if cm is not None:
        for line in tqdm(range(len(cm)), desc="loading word vectors from {}".format(fname_txt)):
            entries = cm[line].strip().split(b' ')
            word, entries = entries[0], entries[1:]
            if wv_size is None:
                wv_size = len(entries)
            try:
                if isinstance(word, six.binary_type):
                    word = word.decode('utf-8')
            except:
                print('non-UTF8 token', repr(word), 'ignored')
                continue
            wv_arr.extend(float(x) for x in entries)
            wv_tokens.append(word)

    wv_dict = {word: i for i, word in enumerate(wv_tokens)}
    wv_arr = torch.Tensor(wv_arr).view(-1, wv_size)
    ret = (wv_dict, wv_arr, wv_size)
    torch.save(ret, fname + '.pt')
    return ret


def obj_edge_vectors(names, wv_dir, wv_type='glove.6B', wv_dim=300):
    wv_dict, wv_arr, wv_size = load_word_vectors(wv_dir, wv_type, wv_dim)

    vectors = torch.Tensor(len(names), wv_dim)
    vectors.normal_(0,1)

    for i, token in enumerate(names):
        wv_index = wv_dict.get(token, None)
        if wv_index is not None:
            vectors[i] = wv_arr[wv_index]
        else:
            # Try the longest word
            lw_token = sorted(token.split(' '), key=lambda x: len(x), reverse=True)[0]
            print("{} -> {} ".format(token, lw_token))
            wv_index = wv_dict.get(lw_token, None)
            if wv_index is not None:
                vectors[i] = wv_arr[wv_index]
            else:
                print("fail on {}".format(token))

    return vectors


def main(n_clusters=148):
    obj_embed_vecs = np.load('glove_word_embeds.npy', allow_pickle=True)

    # model = AgglomerativeClustering(linkage='ward', n_clusters=n_clusters)
    # y = model.fit_predict(obj_embed_vecs)
    # clusters = [[] for _ in range(n_clusters)]
    # for cls, cluster in enumerate(y):
    #     k = str(cls + 1)
    #     clusters[cluster].append(ID2NAME[k])
    # for _ in clusters:
    #     print(_)

    model = AgglomerativeClustering(linkage='ward', n_clusters=None, distance_threshold=10)
    y = model.fit_predict(obj_embed_vecs)
    n_clusters = model.labels_.max() + 1
    clusters = [[] for _ in range(n_clusters)]
    cluster_dict = {}
    for cls, cluster in enumerate(y):
        # k = str(cls + 1)
        # clusters[cluster].append(ID2NAME[k])
        cluster_dict[cls + 1] = cluster
    # for _ in clusters:
    #     print(_)
    print(cluster_dict)


if __name__ == '__main__':
    main()
