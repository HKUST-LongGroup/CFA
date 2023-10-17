import numpy as np
import os

IMG_REL = list(np.load('./gt_relations.npy', allow_pickle=True))
ID2NAME = {
    "0": "__background__",
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


def get_similarity(obj1, obj2):
    def _count_edges(obj, idx=1):
        cnt = 0
        for rels in IMG_REL:
            for rel in rels:
                if rel[idx] in obj:
                    cnt += 1
        return cnt
    def _count_nodes(obj1, obj2, idx=1):
        node1, node2 = set(), set()
        ana_idx = 0 if idx == 1 else 1
        for rels in IMG_REL:
            for rel in rels:
                if rel[idx] in obj1:
                    node1.add(rel[ana_idx])
                elif rel[idx] in obj2:  # sim between itself equals to 0.
                    node2.add(rel[ana_idx])
        return node1 & node2
    
    LS = _count_nodes(obj1, obj2, 1)
    in1, in2 = _count_edges(obj1, 1), _count_edges(obj2, 1)

    LO = _count_nodes(obj1, obj2, 0)
    out1, out2 = _count_edges(obj1, 0), _count_edges(obj2, 0)

    item1 = float(len(LS)) / ( in1 + in2 - len(LS) )
    item2 = float(len(LO)) / ( out1 + out2 - len(LO) )

    return item1 + item2


def get_two_nearest(clusters, sim_metrix, lambdas):
    if sim_metrix is None:
        if os.path.exists('./initial_sim_metrix.npy'):
            sim_metrix = np.load('./initial_sim_metrix.npy', allow_pickle=True)
        else:
            sim_metrix = np.ones((len(clusters), len(clusters)), dtype=np.float32)
            for i in range(len(sim_metrix)):
                for j in range(i, len(sim_metrix), 1):
                    sim_metrix[i][j] = get_similarity(clusters[i], clusters[j])
                    sim_metrix[j][i] = sim_metrix[i][j]
            np.save('./initial_sim_metrix.npy', sim_metrix, allow_pickle=True)
    
    max_v, max_row, max_col = 0., 0, 0
    for i in range(len(sim_metrix)):
        for j in range(i, len(sim_metrix), 1):
            if (sim_metrix[i, j] / (lambdas[i] + lambdas[j])) > max_v:
                max_v = sim_metrix[i, j] / (lambdas[i] + lambdas[j])
                max_row = i
                max_col = j

    return sim_metrix, max_row, max_col


def merge(clusters, sim_metrix, lambdas, row, col):
    (idx1, idx2) = (row, col) if row < col else ((col, row))

    clusters[idx1] = clusters[idx1].union(clusters[idx2])
    clusters.pop(idx2)

    lambdas[idx1] += (lambdas[idx2] + 1)
    lambdas.pop(idx2)

    sim_metrix = np.delete(sim_metrix, idx2, axis=0)
    sim_metrix = np.delete(sim_metrix, idx2, axis=1)
    for c in range(sim_metrix.shape[1]):
        sim_metrix[idx1][c] = get_similarity(clusters[idx1], clusters[c])
        sim_metrix[c][idx1] = sim_metrix[idx1][c]

    assert len(clusters) == sim_metrix.shape[0] == sim_metrix.shape[1] == len(lambdas)
    return clusters, sim_metrix, lambdas


def clustering(N=150, numk=40):
    log = open('log.txt', 'w')
    clusters = [set([i]) for i in range(1, N+1, 1)]
    lambdas = [1 for _ in range(1, N+1, 1)]
    sim_metrix = None
    while len(clusters) > numk:
        sim_metrix, row, column = get_two_nearest(clusters, sim_metrix, lambdas)
        log.write('{}, {} \n'.format([ID2NAME[str(i)] for i in clusters[row]], [ID2NAME[str(i)] for i in clusters[column]]))
        clusters, sim_metrix, lambdas = merge(clusters, sim_metrix, lambdas, row, column)
        print(len(clusters))
    
    log.close()
    np.save('./clusters.npy', clusters, allow_pickle=True)


# clustering()
clusters = np.load('./clusters.npy', allow_pickle=True)
for cluster in clusters:
    print([ID2NAME[str(i)] for i in cluster])
print('Done')
