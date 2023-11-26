import os
import json
from pyecharts.charts import Tree
from pyecharts import options as opts

tree_clusters = {}
path = ''
with open(path, 'r') as fp:
    for line in fp.readlines():
        if not line.startswith('Merging:'):
            continue

        merged_clusters = eval('[{}]'.format(line.strip()[8:]))
        key = str(sorted(merged_clusters[0] + merged_clusters[1]))
        c0, c1 = str(sorted(merged_clusters[0])), str(sorted(merged_clusters[1]))
        if c0 in tree_clusters and c1 in tree_clusters:
            tree_clusters[key] = dict(name='-', children=[tree_clusters[c0], tree_clusters[c1]])
            tree_clusters.pop(c0), tree_clusters.pop(c1)
        elif c0 in tree_clusters:
            tree_clusters[key] = dict(name='-', children=[tree_clusters[c0], dict(name=merged_clusters[1][0])])
            tree_clusters.pop(c0)
        elif c1 in tree_clusters:
            tree_clusters[key] = dict(name='-', children=[tree_clusters[c1], dict(name=merged_clusters[0][0])])
            tree_clusters.pop(c1)
        else:
            if not( (len(merged_clusters[0]) == 1) and (len(merged_clusters[1])) == 1  ):
                import pdb;pdb.set_trace()
            tree_clusters[key] = dict(name='-', children=[dict(name=merged_clusters[0][0]), dict(name=merged_clusters[1][0])])
        # import pdb; pdb.set_trace()
        # print(tree_clusters)

tree_clusters = [v for _, v in tree_clusters.items()]

tree = Tree(init_opts=opts.InitOpts(width="1400px", height="800px"))
# tree = Tree()

tree = tree.add(
    series_name="",
    data=[tree_clusters],
    # pos_top="18%",
    # pos_bottom="14%",
    layout="radial",
    # symbol="emptyCircle",
    # symbol_size=7,
)
# tree = tree.set_global_opts(
#         tooltip_opts=opts.TooltipOpts(trigger="item", trigger_on="mousemove")
#     )
tree.render('clusters.html')


