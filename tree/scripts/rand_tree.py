import networkx as nx
import matplotlib.pyplot as plt
import random

import pydot
import tqdm
from networkx.drawing.nx_pydot import graphviz_layout

prefixes = []
nodes_num = 1
prefix_to_node = {'[-1]': 0}

# return a tree with maximum depth MAX_D that branches with probability p at most N times for each internal node
def rand_tree(d, MAX_D, MAX_N, p, index, prefix):
  global prefixes, nodes_num
  prefix = prefix[:] + [index]
  prefixes.append(prefix)

  prefix_to_node[str(prefix)] = nodes_num
  nodes_num += 1

  if d == MAX_D or random.randint(0, 100) >= p * 100:
    return 
      
  # if the tree branches, at least one branch is made
  n = random.randint(1, MAX_N)
  
  for i in range(n):
    rand_tree(d+1, MAX_D, MAX_N, p, i, prefix)
            
def gen_random_tree(max_d, max_n, branch_p, image_fpath):
  global prefixes 

  n = random.randint(1, max_n)
  for i in tqdm.tqdm(range(n)):
    rand_tree(1, max_d, max_n, branch_p, i, [-1])

  prefixes = sorted(prefixes)

  edges = []
  for prefix in prefixes:
    from_slice = prefix[0:1]
    from_node = prefix_to_node[str(from_slice)]

    for i in range(len(prefix) - 1):
      to_slice = prefix[0:i+2]
      to_node = prefix_to_node[str(to_slice)]

      edges.append((from_node, to_node))

      from_node = to_node

  # remove duplicates from edges
  edges = list(set(edges))

  g = nx.from_edgelist(edges)
  pos = graphviz_layout(g, prog="dot")
  nx.draw(g, pos)
  # plt.savefig(image_fpath)

  assert len(prefixes) == nodes_num - 1

  return [prefix[1:] for prefix in prefixes], edges 

if __name__ == '__main__':
  gen_random_tree(max_d = 4, max_n = 4, branch_p = 0.8, image_fpath = './test.png')
  plt.show()
