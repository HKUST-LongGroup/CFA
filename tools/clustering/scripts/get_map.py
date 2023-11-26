import json

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
NAME2ID = {v: int(k) for k, v in ID2NAME.items()}

def get_single_map(fpath):
    clusters = []
    with open(fpath, 'r') as fp:
        for line in fp.readlines():
            clusters.append(eval(line))

    cls2clu = {}
    for clu_idx, cluster in enumerate(clusters):
        for cls in cluster:
            cls2clu[NAME2ID[cls]] = clu_idx
    print(cls2clu)

def get_multiple_map(log_path, out_path, prefix='cls2clu_obj'):
    def get_map(clusters):
        cls2clu = {}
        for clu_idx, cluster in enumerate(clusters):
            for cls in cluster:
                cls2clu[NAME2ID[cls]] = clu_idx
        return cls2clu

    out_fp = open(out_path, 'w')
    log_fp = open(log_path, 'r')
    logs = log_fp.readlines()
    clusters = []
    for line in logs[-2:]:
        clusters.append(sorted(eval(line)))
    
    for line in logs[::-1]:
        if not line.startswith('Merging:'):
            continue
        cls2clu = get_map(clusters)
        out_fp.write('{}{} = '.format(prefix, len(set(cls2clu.values()))))
        out_fp.write(str(cls2clu) + '\n')

        merged_clusters = eval('[{}]'.format(line.strip()[8:]))
        merged_clusters = [sorted(i) for i in merged_clusters]
        idx = clusters.index(sorted(merged_clusters[0] + merged_clusters[1]))
        clusters.pop(idx)
        clusters.extend(merged_clusters)

    cls2clu = get_map(clusters)
    out_fp.write('{}{} = '.format(prefix, len(set(cls2clu.values()))))
    out_fp.write(str(cls2clu) + '\n')
    out_fp.write('{}s = [None, None'.format(prefix))
    for i in range(2, 151):
        out_fp.write(', {}{}'.format(prefix, i))
    out_fp.write(']\n')
    out_fp.close()

if __name__ == '__main__':
    pass
