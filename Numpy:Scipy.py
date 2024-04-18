
# coding: utf-8

# # Part 3: Sparse matrix storage
# 
# This part is about sparse matrix storage in Numpy/Scipy.
# 
# > **Note:** When you submit this notebook to the autograder, there will be a time-limit of about 180 seconds (3 minutes). If your notebook takes more time than that to run, it's likely you are not writing efficient enough code.
# 
# Start by running the following code cell to get some of the key modules you'll need.

# In[1]:


import sys
print(sys.version)

from random import sample # Used to generate a random sample

import numpy as np
print(np.__version__)

from numpy.random import default_rng
rng = default_rng(12345)

import pandas as pd
print(pd.__version__)

from IPython.display import display

# Custom benchmarking code
get_ipython().magic('load_ext autoreload')
get_ipython().magic('autoreload 2')

from cse6040bench import benchit


# ## Sample data
# 
# For this part, you'll need to download the dataset below. It's a list of pairs of strings. The strings, it turns out, correspond to anonymized Yelp! user IDs; a pair $(a, b)$ exists if user $a$ is friends on Yelp! with user $b$.

# **Exercise 0** (ungraded). Verify that you can obtain the dataset and take a peek by running the two code cells that follow.

# In[2]:


import requests
import os
import hashlib
import io

local_filename = './resource/asnlib/publicdata/UserEdges-1M.csv'
checksum = '4668034bbcd2fa120915ea2d15eafa8d'
with io.open(local_filename, 'r', encoding='utf-8', errors='replace') as f:
    body = f.read()
    body_checksum = hashlib.md5(body.encode('utf-8')).hexdigest()
    assert body_checksum == checksum,         "Data file '{}' has incorrect checksum: '{}' instead of '{}'".format(local_filename,
                                                                             body_checksum,
                                                                             checksum)
    print("==> Checksum test passes: {}".format(checksum))
    
print("==> '{}' is ready!\n".format(local_filename))
print("(Auxiliary files appear to be ready.)")


# In[3]:


# Peek at the data:
edges_raw = pd.read_csv(local_filename)
display(edges_raw.head ())
print("...\n`edges_raw` has {} entries.".format(len(edges_raw)))


# Evidently, this dataframe has one million entries.

# **Exercise 1** (ungraded). Explain what the following code cell does.

# In[4]:


edges_raw_trans = pd.DataFrame({'Source': edges_raw['Target'],
                                'Target': edges_raw['Source']})
edges_raw_symm = pd.concat([edges_raw, edges_raw_trans])
edges = edges_raw_symm.drop_duplicates()

V_names = set(edges['Source'])
V_names.update(set(edges['Target']))

num_edges = len(edges)
num_verts = len(V_names)
print("==> |V| == {}, |E| == {}".format(num_verts, num_edges))


# **Answer.** Give this question some thought before peeking at our suggested answer, which follows.
# 
# Recall that the input dataframe, `edges_raw`, has a row $(a, b)$ if $a$ and $b$ are friends. But here is what is unclear at the outset: if $(a, b)$ is an entry in this table, is $(b, a)$ also an entry? The code in the above cell effectively figures that out, by computing a dataframe, `edges`, that contains both $(a, b)$ and $(b, a)$, with no additional duplicates, i.e., no copies of $(a, b)$.
# 
# It also uses sets to construct a set, `V_names`, that consists of all the names. Evidently, the dataset consists of 107,456 unique names and 441,320 unique pairs, or 882,640 pairs when you "symmetrize" to ensure that both $(a, b)$ and $(b, a)$ appear.

# ## Graphs
# 
# One way a computer scientist thinks of this collection of pairs is as a _graph_: 
# https://en.wikipedia.org/wiki/Graph_(discrete_mathematics)
# 
# The names or user IDs are _nodes_ or _vertices_ of this graph; the pairs are _edges_, or arrows that connect vertices. That's why the final output objects are named `V_names` (for vertex names) and `edges` (for the vertex-to-vertex relationships). The process or calculation to ensure that both $(a, b)$ and $(b, a)$ are contained in `edges` is sometimes referred to as _symmetrizing_ the graph: it ensures that if an edge $a \rightarrow b$ exists, then so does $b \rightarrow a$. If that's true for all edges, then the graph is _undirected_. The Wikipedia page linked to above explains these terms with some examples and helpful pictures, so take a moment to review that material before moving on.
# 
# We'll also refer to this collection of vertices and edges as the _connectivity graph_.

# ## Sparse matrix storage: Baseline methods
# 
# Let's start by reminding ourselves how our previous method for storing sparse matrices, based on nested default dictionaries, works and performs.

# In[4]:


def sparse_matrix(base_type=float):
    """Returns a sparse matrix using nested default dictionaries."""
    from collections import defaultdict
    return defaultdict(lambda: defaultdict (base_type))

def dense_vector(init, base_type=float):
    """
    Returns a dense vector, either of a given length
    and initialized to 0 values or using a given list
    of initial values.
    """
    # Case 1: `init` is a list of initial values for the vector entries
    if type(init) is list:
        initial_values = init
        return [base_type(x) for x in initial_values]
    
    # Else, case 2: `init` is a vector length.
    assert type(init) is int
    return [base_type(0)] * init


# **Exercise 2** (3 points). Implement a function to compute $y \leftarrow A x$. Assume that the keys of the sparse matrix data structure are integers in the interval $[0, s)$ where $s$ is the number of rows or columns as appropriate.
# 
# > **Hint**: Recall that you implemented a _dense_ matrix-vector multiply in Part 2. Think about how to adapt that same piece of code when the data structure for storing `A` has changed to the sparse representation given in this exercise.

# In[5]:


def spmv(A, x, num_rows=None): 
    if num_rows is None:
        num_rows = max(A.keys()) + 1
    y = dense_vector(num_rows) 
    
    # Recall: y = A*x is, conceptually,
    # for all i, y[i] == sum over all j of (A[i, j] * x[j])
    ###
    for i, row_i in A.items():
        s = 0.0  # Initialize the sum for the current row
        for j, a_ij in row_i.items():
            s += a_ij * x[j]  # Multiply each element in the row by the corresponding element in the vector and accumulate the sum
    # The variable 's' now contains the weighted sum for the current row
    # You may use or store 's' as needed in the context of your application
        y[i] = s
    ###
    return y


# In[6]:


# Test cell: `spmv_baseline_test`

#   / 0.   -2.5   1.2 \   / 1. \   / -1.4 \
#   | 0.1   1.    0.  | * | 2. | = |  2.1 |
#   \ 6.   -1.    0.  /   \ 3. /   \  4.0 /

A = sparse_matrix ()
A[0][1] = -2.5
A[0][2] = 1.2
A[1][0] = 0.1
A[1][1] = 1.
A[2][0] = 6.
A[2][1] = -1.

x = dense_vector ([1, 2, 3])
y0 = dense_vector ([-1.4, 2.1, 4.0])


# Try your code:
y = spmv(A, x)

max_abs_residual = max([abs(a-b) for a, b in zip(y, y0)])

print ("==> A:", A)
print ("==> x:", x)
print ("==> True solution, y0:", y0)
print ("==> Your solution, y:", y)
print ("==> Residual (infinity norm):", max_abs_residual)
assert max_abs_residual <= 1e-14

print ("\n(Passed.)")


# Next, let's convert the `edges` input into a sparse matrix representing its connectivity graph. To do so, we'll first want to map names to integers.

# In[8]:


id2name = {} # id2name[id] == name
name2id = {} # name2id[name] == id

for k, v in enumerate (V_names):
    # for debugging
    if k <= 5: print ("Name %s -> Vertex id %d" % (v, k))
    if k == 6: print ("...")
        
    id2name[k] = v
    name2id[v] = k


# **Exercise 3** (3 points). Given `id2name` and `name2id` as computed above, convert `edges` into a sparse matrix, `G`, where there is an entry `G[s][t] == 1.0` wherever an edge `(s, t)` exists.
# 
# **Note** - This step might take time for the kernel to process as there are 1 million rows

# In[9]:


G = sparse_matrix()
print(G)

###
for i in range(len(edges)):  # edges is the table above
    s = edges['Source'].iloc[i]
    t = edges['Target'].iloc[i]
    s_id = name2id[s]
    t_id = name2id[t]
    G[s_id][t_id] = 1.0
###


# In[10]:


# Test cell: `edges2spmat1_test`

G_rows_nnz = [len(row_i) for row_i in G.values()]
print ("G has {} vertices and {} edges.".format(len(G.keys()), sum(G_rows_nnz)))

assert len(G.keys()) == num_verts
assert sum(G_rows_nnz) == num_edges

# Check a random sample
for k in sample(range(num_edges), 1000):
    i = name2id[edges['Source'].iloc[k]]
    j = name2id[edges['Target'].iloc[k]]
    assert i in G
    assert j in G[i]
    assert G[i][j] == 1.0

print ("\n(Passed.)")


# In[11]:


print(len(G_rows_nnz))


# **Exercise 4** (3 points). In the above, we asked you to construct `G` using integer keys. However, since we are, after all, using default dictionaries, we could also use the vertex _names_ as keys. Construct a new sparse matrix, `H`, which uses the vertex names as keys instead of integers.

# In[12]:


H = sparse_matrix()
###
for i in range (len(edges)):
    s = edges['Source'].iloc[i]
    t = edges['Target'].iloc[i]
    H[s][t] = 1.0
###


# In[13]:


# Test cell: `create_H_test`

H_rows_nnz = [len(h) for h in H.values()]
print("`H` has {} vertices and {} edges.".format(len(H.keys()), sum(H_rows_nnz)))

assert len(H.keys()) == num_verts
assert sum(H_rows_nnz) == num_edges

# Check a random sample
for i in sample(G.keys(), 100):
    i_name = id2name[i]
    assert i_name in H
    assert len(G[i]) == len(H[i_name])
    
print ("\n(Passed.)")


# **Exercise 5** (3 points). Implement a sparse matrix-vector multiply for matrices with named keys. In this case, it will be convenient to have vectors that also have named keys; assume we use dictionaries to hold these vectors as suggested in the code skeleton, below.
# 
# > **Hint:** Go back to **Exercise 2** and see what you did there. If it was implemented well, a modest change to the solution for Exercise 2 is likely to be all you need here.

# In[14]:


def vector_keyed(keys=None, values=0, base_type=float):
    if keys is not None:
        if type(values) is not list:
            values = [base_type(values)] * len(keys)
        else:
            values = [base_type(v) for v in values]
        x = dict(zip(keys, values))
    else:
        x = {}
    return x

def spmv_keyed(A, x):
    """Performs a sparse matrix-vector multiply for keyed matrices and vectors."""
    assert type(x) is dict
    
    y = vector_keyed(keys=A.keys(), values=0.0) 
    
    # Recall: y = A*x is, conceptually,
    # for all i, y[i] == sum over all j of (A[i, j] * x[j])
    ###
    for i, row_i in A.items():
        for j, a_ij in row_i.items():
            y[i] += a_ij * x[j]  # Multiply each element in the row by the corresponding element in the vector and accumulate the sum
    
    ###
    return y
    ###


# In[15]:


# Test cell: `spmv_keyed_test`

#   'row':  / 0.   -2.5   1.2 \   / 1. \   / -1.4 \
#  'your':  | 0.1   1.    0.  | * | 2. | = |  2.1 |
#  'boat':  \ 6.   -1.    0.  /   \ 3. /   \  4.0 /

KEYS = ['row', 'your', 'boat']

A_keyed = sparse_matrix ()
A_keyed['row']['your'] = -2.5
A_keyed['row']['boat'] = 1.2
A_keyed['your']['row'] = 0.1
A_keyed['your']['your'] = 1.
A_keyed['boat']['row'] = 6.
A_keyed['boat']['your'] = -1.

x_keyed = vector_keyed (KEYS, [1, 2, 3])
y0_keyed = vector_keyed (KEYS, [-1.4, 2.1, 4.0])


# Try your code:
y_keyed = spmv_keyed (A_keyed, x_keyed)

# Measure the residual:
residuals = [(y_keyed[k] - y0_keyed[k]) for k in KEYS]
max_abs_residual = max ([abs (r) for r in residuals])

print ("==> A_keyed:", A_keyed)
print ("==> x_keyed:", x_keyed)
print ("==> True solution, y0_keyed:", y0_keyed)
print ("==> Your solution:", y_keyed)
print ("==> Residual (infinity norm):", max_abs_residual)
assert max_abs_residual <= 1e-14

print ("\n(Passed.)")


# Let's benchmark `spmv()` against `spmv_keyed()` on the full data set. Do they perform differently?
# 
# > If this benchmark or any of the subsequent ones take an excessively long time to run, including autograder failure, then it's likely you have not implemented `spmv_keyed` efficiently. You'll need to look closely and ensure your implementation is not doing something that leads to excessive running time or memory consumption, which might happen if you don't really understand how dictionaries work.

# In[16]:


x = dense_vector ([1.] * num_verts)
benchit('spmv(G, x)', scope=globals());

x_keyed = vector_keyed (keys=[v for v in V_names], values=1.)
benchit('spmv_keyed(H, x_keyed)', scope=globals());


# ## Alternative formats: 
# 
# Take a look at the following slides: [link](https://www.dropbox.com/s/4fwq21dy60g4w4u/cse6040-matrix-storage-notes.pdf?dl=0). These slides cover the basics of two list-based sparse matrix formats known as _coordinate format_ (COO) and _compressed sparse row_ (CSR). We will also discuss them briefly below.

# ### Coordinate Format (COO)
# In this format we store three lists, one each for rows, columns and the elements of the matrix. Look at the below picture to understand how these lists are formed.

# ![Coordinate (COO) storage](https://github.com/cse6040/labs-fa17/raw/master/lab10-numpy/coo.png)

# **Exercise 6** (3 points). Convert the `edges[:]` data into a coordinate (COO) data structure in native Python using three lists, `coo_rows[:]`, `coo_cols[:]`, and `coo_vals[:]`, to store the row indices, column indices, and matrix values, respectively. Use integer indices and set all values to 1.
# 
# **Hint** - Think of what rows, columns and values mean conceptually when you relate it with our dataset of edges

# In[17]:


###
coo_rows = [name2id[s] for s in edges['Source']]
coo_cols = [name2id[t] for t in edges['Target']]
coo_vals = [1.0] * len(edges)
###


# In[18]:


# Test cell: `create_coo_test`

assert len (coo_rows) == num_edges
assert len (coo_cols) == num_edges
assert len (coo_vals) == num_edges
assert all ([v == 1. for v in coo_vals])

# Randomly check a bunch of values
coo_zip = zip (coo_rows, coo_cols, coo_vals)
for i, j, a_ij in sample (list (coo_zip), 1000):
    assert (i in G) and j in G[i]
    
print ("\n(Passed.)")


# **Exercise 7** (3 points). Implement a sparse matrix-vector multiply routine for COO implementation.

# In[19]:


def spmv_coo(R, C, V, x, num_rows=None):
    """
    Returns y = A*x, where A has 'm' rows and is stored in
    COO format by the array triples, (R, C, V).
    """
    assert type(x) is list
    assert type(R) is list
    assert type(C) is list
    assert type(V) is list
    assert len(R) == len(C) == len(V)
    
    
    if num_rows is None:
        num_rows = max(R) + 1
    
    y = dense_vector(num_rows)
    
    ###


    for k in range(len(R)):
        i, j, a_ij = R[k], C[k], V[k]
        y[i] += a_ij * x[j]
    ###
    
    return y


# In[20]:


# Test cell: `spmv_coo_test`

#   / 0.   -2.5   1.2 \   / 1. \   / -1.4 \
#   | 0.1   1.    0.  | * | 2. | = |  2.1 |
#   \ 6.   -1.    0.  /   \ 3. /   \  4.0 /

A_coo_rows = [0, 0, 1, 1, 2, 2]
A_coo_cols = [1, 2, 0, 1, 0, 1]
A_coo_vals = [-2.5, 1.2, 0.1, 1., 6., -1.]

x = dense_vector([1, 2, 3])
y0 = dense_vector([-1.4, 2.1, 4.0])

# Try your code:
y_coo = spmv_coo(A_coo_rows, A_coo_cols, A_coo_vals, x)

max_abs_residual = max ([abs(a-b) for a, b in zip(y_coo, y0)])

print("==> A_coo:", list(zip(A_coo_rows, A_coo_cols, A_coo_vals)))
print("==> x:", x)
print("==> True solution, y0:", y0)
print("==> Your solution:", y_coo)
print("==> Residual (infinity norm):", max_abs_residual)
assert max_abs_residual <= 1e-15

print("\n(Passed.)")


# Let's see how fast this is...

# In[21]:


x = dense_vector([1.] * num_verts)
benchit('spmv_coo(coo_rows, coo_cols, coo_vals, x)', scope=globals());


# ### Compressed Sparse Row Format
# 
# This format tries to compress the sparse matrix further compared to COO format. Suppose you have the following coordinate representation of a sparse matrix where you sort by row index:
# 
# ```python
#     rows   = [0, 0, 0, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3, 4, 4, 4, 5, 5, 5, 6, 6]
#     cols   = [1, 2, 4, 0, 2, 3, 0, 1, 3, 4, 1, 2, 5, 6, 0, 2, 5, 3, 4, 6, 3, 5]
#     values = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
# ```
# 
# If there are `nnz` nonzeroes, then this representation requires `3*nnz` units of storage since it needs three arrays, each of length `nnz`.

# Observe that when we sort by row index, values are repeated wherever there is more than one nonzero in a row. The idea behind CSR is to exploit this redundancy. From the sorted COO representation, we keep the column indices and values as-is. Then, instead of storing every row index, we just store the starting _offset_ of each row in those two lists, which we'll refer to as the _row pointers_, stored as the list `rowptr`, below:
# 
# ```python
#     cols   = [1, 2, 4, 0, 2, 3, 0, 1, 3, 4, 1, 2, 5, 6, 0, 2, 5, 3, 4, 6, 3, 5]
#     values = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
#     rowptr = [0,       3,       6,         10,         14,      17,      20,   22]
# ```
# 
# If the sparse matrix has `n` rows, then the `rowptr` list has `n+1` elements, where the last element (`rowptr[-1] == rowptr[n]`) is `nnz`. (Why might you need this last element?)

# **Exercise 8** (3 points). Complete the function, `coo2csr(coo_rows, coo_cols, coo_vals)`, below. The inputs are three Python lists corresponding to a sparse matrix in COO format, like the example illustrated above. Your function should return a triple, `(csr_ptrs, csr_inds, csr_vals)`, corresponding to the same matrix but stored in CSR format, again, like what is shown above, where `csr_ptrs` would be the row pointers (`rowptr`), `csr_inds` would be the column indices (`colind`), and `csr_vals` would be the values (`values`).
# 
# To help you out, we show how to calculate `csr_inds` and `csr_vals`. You need to figure out how to compute `csr_ptrs`. The function is set up to return these three lists.
# 
# > Although the test cell does not check it, in principle, your implementation should also work correctly if a row has an _empty_ row. In such cases, what should the CSR data structure look like?

# In[22]:


def coo2csr(coo_rows, coo_cols, coo_vals):
    from operator import itemgetter
    C = sorted(zip(coo_rows, coo_cols, coo_vals), key=itemgetter(0))
    nnz = len(C)
    assert nnz >= 1

    csr_inds = [j for _, j, _ in C]
    csr_vals = [a_ij for _, _, a_ij in C]

    # Your task: Compute `csr_ptrs`
    
    ###
    def solution_0(C):  # "Basic"
        C_rows = [i for i, _, _ in C]  # Sorted row indices
        ptrs = []  # Output
        i_previous = None  # ID of the previously visited row
    
        for k, i in enumerate(C_rows):
            if i_previous is None:  # `i` is the first row
                ptrs += [0] * (i + 1)  # `i` may be greater than 0!
            elif i != i_previous:  # `i` is a new row index
                ptrs += [k] * (i - i_previous)  # handles all-zero rows
            i_previous = i
    
        ptrs += [len(C_rows)]
        return ptrs

    def solution_1(C):  # Advanced!
        from collections import Counter
        from itertools import accumulate
        C_rows = [i for i, _, _ in C]  # sorted rows
        row_counts = Counter(C_rows)
        num_rows = (C_rows[-1] + 1) if C_rows else 0
        offsets = [row_counts.get(i, 0) for i in range(-1, num_rows)]
        return list(accumulate(offsets))

    def solution_2(C):  # Cuter version of `solution_0`
        from itertools import chain
        C_rows = [i for i, _, _ in C]  # sorted rows
        ptrs = (
            [[0] * (C_rows[0] + 1)] +
            [[i] * (C_rows[i] - C_rows[i - 1]) for i in range(1, nnz)] +
            [[len(C)]]
        )
        return [i for i in chain.from_iterable(ptrs)]  # flatten the nested list

    csr_ptrs = solution_0(C)     

    ###
    
    return csr_ptrs, csr_inds, csr_vals


# In[23]:


# Test cell 0: `create_csr_test` (1 point)

csr_ptrs, csr_inds, csr_vals = coo2csr(coo_rows, coo_cols, coo_vals)

assert type(csr_ptrs) is list, "`csr_ptrs` is not a list."
assert type(csr_inds) is list, "`csr_inds` is not a list."
assert type(csr_vals) is list, "`csr_vals` is not a list."

assert len(csr_ptrs) == (num_verts + 1), "`csr_ptrs` has {} values instead of {}".format(len(csr_ptrs), num_verts+1)
assert len(csr_inds) == num_edges, "`csr_inds` has {} values instead of {}".format(len(csr_inds), num_edges)
assert len(csr_vals) == num_edges, "`csr_vals` has {} values instead of {}".format(len(csr_vals), num_edges)
assert csr_ptrs[num_verts] == num_edges, "`csr_ptrs[{}]` == {} instead of {}".format(num_verts, csr_ptrs[num_verts], num_edges)

# Check some random entries
for i in sample(range(num_verts), 10000):
    assert i in G
    a, b = csr_ptrs[i], csr_ptrs[i+1]
    msg_prefix = "Row {} should have these nonzeros: {}".format(i, G[i])
    assert (b-a) == len(G[i]), "{}, which is {} nonzeros; instead, it has just {}.".format(msg_prefix, len(G[i]), b-a)
    assert all([(j in G[i]) for j in csr_inds[a:b]]), "{}. However, it may have missing or incorrect column indices: csr_inds[{}:{}] == {}".format(msg_prefix, a, b, csr_inds[a:b])
    assert all([(j in csr_inds[a:b] for j in G[i].keys())]), "{}. However, it may have missing or incorrect column indices: csr_inds[{}:{}] == {}".format(msg_prefix, a, b, csr_inds[a:b])

print ("\n(Passed.)")


# In[24]:


# Test cell 1: case with empty rows (1 point)

#          B             csr_ptrs

#  / 0 0 0 0 0 0 0 \        0
#  | 0 0 0 0 0 0 0 |        0
#  | 1 0 0 1 0 1 0 |        0
#  | 0 0 0 0 0 0 0 |   -->  3
#  | 0 0 0 0 0 0 0 |        3
#  | 0 1 0 0 1 0 1 |        3
#  \ 0 0 1 1 0 0 0 /        6
#                           8

B_coo_rows = [2, 2, 2, 5, 5, 5, 6, 6]
B_coo_cols = [5, 0, 3, 1, 4, 6, 2, 3]
B_coo_vals = [1, 1, 1, 1, 1, 1, 1, 1]
B_csr_ptrs = [0, 0, 0, 3, 3, 3, 6, 8]

your_csr_ptrs, _, _ = coo2csr(B_coo_rows, B_coo_cols, B_coo_vals)

print(f'COO format:\n\tcoo_rows = {B_coo_rows} \n\tcoo_cols = {B_coo_cols} \n\tcoo_vals = {B_coo_vals}')
print(f'\nCSR pointers:')
print(f'==> True solution: {B_csr_ptrs}')
print(f'==> Your solution: {your_csr_ptrs}\n')

assert your_csr_ptrs == B_csr_ptrs
print("\n(Passed!)")


# In[25]:


# Test cell 2: Spot-check some rows (1 point)

# Given the COO format of our dataset of edges, we will empty randomly selected 3 consecutive rows:
i = rng.integers(num_verts-3)
idx = np.concatenate((np.where(np.array(coo_rows)==i)[0], np.where(np.array(coo_rows)==i+1)[0],
                      np.where(np.array(coo_rows)==i+2)[0]), axis=0)
coo_rows_empty = np.delete(np.array(coo_rows), idx).tolist()
coo_cols_empty = np.delete(np.array(coo_cols), idx).tolist()
coo_vals_empty = np.delete(np.array(coo_vals), idx).tolist()

# Output of coo2csr:
csr_ptrs_empty, _, _ = coo2csr(coo_rows_empty, coo_cols_empty, coo_vals_empty)

assert max(csr_ptrs_empty) == num_edges - len(idx), f'The largest entry of your output is {max(csr_ptrs_empty)} instead of {num_edges - len(idx)}'
assert len(csr_ptrs_empty) == num_verts + 1, f'Your output has {len(csr_ptrs_empty)} entries instead of {num_verts+1}.'

# Check the CSR format by converting back to COO format:
coo_rows_converted = [(csr_ptrs_empty[i+1] - csr_ptrs_empty[i]) * [i] for i in range(len(csr_ptrs_empty)-1)]
from itertools import chain
coo_rows_converted = [i for i in chain.from_iterable(coo_rows_converted)]

assert sorted(coo_rows_empty) == coo_rows_converted, "The function doesn't handle the empty rows correctly."

print ("\n(Passed.)")


# **Exercise 9** (3 points). Now implement a CSR-based sparse matrix-vector multiply.

# In[27]:


def spmv_csr(ptr, ind, val, x, num_rows=None):
    assert type(ptr) == list
    assert type(ind) == list
    assert type(val) == list
    assert type(x) == list
    if num_rows is None: num_rows = len(ptr) - 1
    assert len(ptr) >= (num_rows+1)  # Why?
    assert len(ind) >= ptr[num_rows]  # Why?
    assert len(val) >= ptr[num_rows]  # Why?
    
    y = dense_vector(num_rows)
    ###
    [y.__setitem__(i, y[i] + val[k] * x[ind[k]]) for i in range(num_rows) for k in range(ptr[i], ptr[i + 1])]
    ###
    return y


# In[28]:


# Test cell: `spmv_csr_test`

#   / 0.   -2.5   1.2 \   / 1. \   / -1.4 \
#   | 0.1   1.    0.  | * | 2. | = |  2.1 |
#   \ 6.   -1.    0.  /   \ 3. /   \  4.0 /

A_csr_ptrs = [ 0,        2,       4,       6]
A_csr_cols = [ 1,   2,   0,   1,  0,   1]
A_csr_vals = [-2.5, 1.2, 0.1, 1., 6., -1.]

x = dense_vector([1, 2, 3])
y0 = dense_vector([-1.4, 2.1, 4.0])

# Try your code:
y_csr = spmv_csr(A_csr_ptrs, A_csr_cols, A_csr_vals, x)

max_abs_residual = max([abs(a-b) for a, b in zip(y_csr, y0)])

print ("==> A_csr_ptrs:", A_csr_ptrs)
print ("==> A_csr_{cols, vals}:", list(zip(A_csr_cols, A_csr_vals)))
print ("==> x:", x)
print ("==> True solution, y0:", y0)
print ("==> Your solution:", y_csr)
print ("==> Residual (infinity norm):", max_abs_residual)
assert max_abs_residual <= 1e-14

print ("\n(Passed.)")


# In[29]:


x = dense_vector([1.] * num_verts)
benchit('spmv_csr(csr_ptrs, csr_inds, csr_vals, x)', scope=globals());


# ## Using Scipy's implementations
# 
# What you should have noticed is that the list-based COO and CSR formats do not really lead to sparse matrix-vector multiply implementations that are much faster than the dictionary-based methods. Let's instead try Scipy's native COO and CSR implementations.

# In[ ]:


import numpy as np
import scipy.sparse as sp

A_coo_sp = sp.coo_matrix((coo_vals, (coo_rows, coo_cols)))
A_csr_sp = A_coo_sp.tocsr() # Alternatively: sp.csr_matrix((val, ind, ptr))
x_sp = np.ones(num_verts)

print ("\n==> COO in Scipy:")
benchit('A_coo_sp.dot(x_sp)', scope=globals());

print ("\n==> CSR in Scipy:")
benchit('A_csr_sp.dot(x_sp)', scope=globals());


# **Fin!** If your notebook runs to this point without error, try submitting it.
