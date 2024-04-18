
# coding: utf-8

# # Part 0: Getting the data
# 
# Before beginning, you'll need to download several files containing the data for the exercises below.
# 
# **Exercise 0** (ungraded). Run the code cell below to download the data. (This code will check if each dataset has already been downloaded and, if so, will avoid re-downloading it.)

# In[1]:


dataset = {'iris.csv': 'd1175c032e1042bec7f974c91e4a65ae',
           'table1.csv': '556ffe73363752488d6b41462f5ff3c9',
           'table2.csv': '16e04efbc7122e515f7a81a3361e6b87',
           'table3.csv': '531d13889f191d6c07c27c3c7ea035ff',
           'table4a.csv': '3c0bbecb40c6958df33a1f9aa5629a80',
           'table4b.csv': '8484bcdf07b50a7e0932099daa72a93d',
           'who.csv': '59fed6bbce66349bf00244b550a93544',
           'who2_soln.csv': 'f6d4875feea9d6fca82ae7f87f760f44',
           'who3_soln.csv': 'fba14f1e088d871e4407f5f737cfbc06'}

from cse6040utils import download_dataset
local_data = download_dataset(dataset, url_suffix='tidy/')

print("\n(All data appear to be ready.)")


# # Part 1: Tidy data 
# 
# The overall topic for this lab is what we'll refer to as representing data _relationally_. The topic of this part is a specific type of relational representation sometimes referred to as the _tidy_ (as opposed to _untidy_ or _messy_) form. The concept of tidy data was developed by [Hadley Wickham](http://hadley.nz/), a statistician and R programming maestro. Much of this lab is based on his tutorial materials (see below).
# 
# If you know [SQL](https://en.wikipedia.org/wiki/SQL), then you are already familiar with relational data representations. However, we might discuss it a little differently from the way you may have encountered the subject previously. The main reason is our overall goal in the class: to build data _analysis_ pipelines. If our end goal is analysis, then we often want to extract or prepare data in a way that makes analysis easier.
# 
# You may find it helpful to also refer to the original materials on which this lab is based:
# 
# * Wickham's R tutorial on making data tidy: http://r4ds.had.co.nz/tidy-data.html
# * The slides from a talk by Wickham on the concept: http://vita.had.co.nz/papers/tidy-data-pres.pdf  
# <!--- Broken Link * Wickham's more theoretical paper of "tidy" vs. "untidy" data: http://www.jstatsoft.org/v59/i10/paper -->

# ------------------------------------------------------------
# 
# ## What is tidy data?
# 
# To build your intuition, consider the following data set collected from a survey or study.

# **Representation 1.** [Two-way contigency table](https://en.wikipedia.org/wiki/Contingency_table).
# 
# |            | Pregnant | Not pregnant |
# |-----------:|:--------:|:------------:|
# | **Male**   |     0    |      5       |
# | **Female** |     1    |      4       |
# 
# **Representation 2.** Observation list or "data frame."
# 
# | Gender  | Pregnant | Count |
# |:-------:|:--------:|:-----:|
# | Male    | Yes      | 0     |
# | Male    | No       | 5     |
# | Female  | Yes      | 1     |
# | Female  | No       | 4     |
# 
# These are two entirely equivalent ways of representing the same data. However, each may be suited to a particular task.

# For instance, Representation 1 is a typical input format for statistical routines that implement Pearson's $\chi^2$-test, which can check for independence between factors. (Are gender and pregnancy status independent?) By contrast, Representation 2 might be better suited to regression. (Can you predict relative counts from gender and pregnancy status?)
# 
# While [Representation 1 has its uses](http://simplystatistics.org/2016/02/17/non-tidy-data/), Wickham argues that Representation 2 is often the cleaner and more general way to supply data to a wide variety of statistical analysis and visualization tasks. He refers to Representation 2 as _tidy_ and Representation 1 as _untidy_ or _messy_.
# 
# > The term "messy" is, as Wickham states, not intended to be perjorative since "messy" representations may be exactly the right ones for particular analysis tasks, as noted above.

# **Definition: Tidy datasets.** More specifically, Wickham defines a tidy data set as one that can be organized into a 2-D table such that
# 
# 1. each column represents a _variable_;
# 2. each row represents an _observation_;
# 3. each entry of the table represents a single _value_, which may come from either categorical (discrete) or continuous spaces.

# Here is a visual schematic of this definition, taken from [another source](http://r4ds.had.co.nz/images/tidy-1.png):
# 
# ![Wickham's illustration of the definition of tidy](http://r4ds.had.co.nz/images/tidy-1.png)

# This definition appeals to a statistician's intuitive idea of data he or she wishes to analyze. It is also consistent with tasks that seek to establish a functional relationship between some response (output) variable from one or more independent variables.
# 
# > A computer scientist with a machine learning outlook might refer to columns as _features_ and rows as _data points_, especially when all values are numerical (ordinal or continuous).

# **Definition: Tibbles.** Here's one more bit of terminology: if a table is tidy, we will call it a _tidy table_, or _tibble_, for short.

# ## Part 2: Tidy Basics and Pandas
# 
# In Python, the [Pandas](http://pandas.pydata.org/) module is a convenient way to store tibbles. If you know [R](http://r-project.org), you will see that the design and API of Pandas's data frames derives from [R's data frames](https://stat.ethz.ch/R-manual/R-devel/library/base/html/data.frame.html).
# 
# In this part of this notebook, let's look at how Pandas works and can help us store Tidy data.
# 
# You may find this introduction to the Pandas module's data structures useful for reference:
# 
# * https://pandas.pydata.org/pandas-docs/stable/dsintro.html

# Consider the famous [Iris data set](https://en.wikipedia.org/wiki/Iris_flower_data_set). It consists of 50 samples from each of three species of Iris (_Iris setosa_, _Iris virginica_, and _Iris versicolor_). Four features were measured from each sample: the lengths and the widths of the [sepals](https://en.wikipedia.org/wiki/Sepal) and [petals](https://en.wikipedia.org/wiki/Petal).
# 
# The following code uses Pandas to read and represent this data in a Pandas data frame object, stored in a variable named `irises`.

# In[2]:


# Some modules you'll need in this part
import pandas as pd
from io import StringIO
from IPython.display import display

# Ignore this line. It will be used later.
SAVE_APPLY = getattr(pd.DataFrame, 'apply')

irises = pd.read_csv(local_data['iris.csv'])
print("=== Iris data set: {} rows x {} columns. ===".format(irises.shape[0], irises.shape[1]))
display (irises)


# In a Pandas data frame, every column has a name (stored as a string) and all values within the column must have the same primitive type. This fact makes columns different from, for instance, lists.
# 
# In addition, every row has a special column, called the data frame's _index_. (Try printing `irises.index`.) Any particular index value serves as a name for its row; these index values are usually integers but can be more complex types, like tuples.

# In[3]:


print(irises.index)


# Separate from the index values (row names), you can also refer to rows by their integer offset from the top, where the first row has an offset of 0 and the last row has an offset of `n-1` if the data frame has `n` rows. You'll see that in action in Exercise 1, below.

# **Exercise 1** (ungraded). Run the following commands to understand what each one does. If it's not obvious, try reading the [Pandas documentation](http://pandas.pydata.org/) or going online to get more information.
# 
# ```python
# irises.describe()
# irises['sepal length'].head()
# irises[["sepal length", "petal width"]].head()
# irises.iloc[5:10]
# irises[irises["sepal length"] > 5.0]
# irises["sepal length"].max()
# irises['species'].unique()
# irises.sort_values(by="sepal length", ascending=False).head(1)
# irises.sort_values(by="sepal length", ascending=False).iloc[5:10]
# irises.sort_values(by="sepal length", ascending=False).loc[5:10]
# irises['x'] = 3.14
# irises.rename(columns={'species': 'type'})
# del irises['x']
# ```

# In[4]:


del irises['x']


# In[5]:


irises.sort_values(by="sepal length", ascending=False).iloc[5:10]


# In[6]:


###
irises.describe()
irises['sepal length'].head()
irises[["sepal length", "petal width"]].head()
irises.iloc[5:10]
irises[irises["sepal length"] > 5.0]
irises["sepal length"].max()
irises['species'].unique()
irises.sort_values(by="sepal length", ascending=False).head(1)
irises.sort_values(by="sepal length", ascending=False).iloc[5:10]
irises.sort_values(by="sepal length", ascending=False).loc[5:10]
irises['x'] = 3.14
irises.rename(columns={'species': 'type'})
del irises['x']
###


# ## Merging data frames: join operations
# 
# Another useful operation on data frames is [merging](http://pandas.pydata.org/pandas-docs/stable/generated/pandas.DataFrame.merge.html).
# 
# For instance, consider the following two tables, `A` and `B`:
# 
# | country     | year | cases  |
# |:------------|-----:|-------:|
# | Afghanistan | 1999 |    745 |
# | Brazil      | 1999 |  37737 |
# | China       | 1999 | 212258 |
# | Afghanistan | 2000 |   2666 |
# | Brazil      | 2000 |  80488 |
# | China       | 2000 | 213766 |
# 
# | country     | year | population |
# |:------------|-----:|-----------:|
# | Afghanistan | 1999 |   19987071 |
# | Brazil      | 1999 |  172006362 |
# | China       | 1999 | 1272915272 |
# | Afghanistan | 2000 |   20595360 |
# | Brazil      | 2000 |  174504898 |
# | China       | 2000 | 1280428583 |
# 
# Suppose we wish to combine these into a single table, `C`:
# 
# | country     | year | cases  | population |
# |:------------|-----:|-------:|-----------:|
# | Afghanistan | 1999 |    745 |   19987071 |
# | Brazil      | 1999 |  37737 |  172006362 |
# | China       | 1999 | 212258 | 1272915272 |
# | Afghanistan | 2000 |   2666 |   20595360 |
# | Brazil      | 2000 |  80488 |  174504898 |
# | China       | 2000 | 213766 | 1280428583 |
# 
# In Pandas, you can perform this merge using the [`.merge()` function](http://pandas.pydata.org/pandas-docs/stable/generated/pandas.DataFrame.merge.html):
# 
# ```python
# C = A.merge (B, on=['country', 'year'])
# ```
# 
# In this call, the `on=` parameter specifies the list of column names to use to align or "match" the two tables, `A` and `B`. By default, `merge()` will only include rows from `A` and `B` where all keys match between the two tables.
# 
# The following code cell demonstrates this functionality.

# In[7]:


A_csv = """country,year,cases
Afghanistan,1999,745
Brazil,1999,37737
China,1999,212258
Afghanistan,2000,2666
Brazil,2000,80488
China,2000,213766"""

with StringIO(A_csv) as fp:
    A = pd.read_csv(fp)
print("=== A ===")
display(A)


# In[8]:


B_csv = """country,year,population
Afghanistan,1999,19987071
Brazil,1999,172006362
China,1999,1272915272
Afghanistan,2000,20595360
Brazil,2000,174504898
China,2000,1280428583"""

with StringIO(B_csv) as fp:
    B = pd.read_csv(fp)
print("\n=== B ===")
display(B)


# In[9]:


C = A.merge(B, on=['country', 'year'])
print("\n=== C = merge(A, B) ===")
display(C)


# **Joins.** This default behavior of keeping only rows that match both input frames is an example of what relational database systems call an _inner-join_ operation. But there are several other types of joins.
# 
# - _Inner-join (`A`, `B`)_ (default): Keep only rows of `A` and `B` where the on-keys match in both.
# - _Outer-join (`A`, `B`)_: Keep all rows of both frames, but merge rows when the on-keys match. For non-matches, fill in missing values with not-a-number (`NaN`) values.
# - _Left-join (`A`, `B`)_: Keep all rows of `A`. Only merge rows of `B` whose on-keys match `A`.
# - _Right-join (`A`, `B`)_: Keep all rows of `B`. Only merge rows of `A` whose on-keys match `B`.
# 
# You can use `merge`'s `how=...` parameter, which takes the (string) values, `'inner`', `'outer'`, `'left'`, and `'right'`. Here are some examples of these types of joins.

# In[10]:


with StringIO("""x,y,z
bug,1,d
rug,2,d
lug,3,d
mug,4,d""") as fp:
    D = pd.read_csv(fp)
print("=== D ===")
display(D)

with StringIO("""x,y,w
hug,-1,e
smug,-2,e
rug,-3,e
tug,-4,e
bug,1,e""") as fp:
    E = pd.read_csv(fp)
print("\n=== E ===")
display(E)

print("\n=== Outer-join (D, E) ===")
display(D.merge(E, on=['x', 'y'], how='outer'))

print("\n=== Left-join (D, E) ===")
display(D.merge(E, on=['x', 'y'], how='left'))

print("\n=== Right-join (D, E) ===")
display(D.merge(E, on=['x', 'y'], how='right'))


print("\n=== Inner-join (D, E) ===")
display(D.merge(E, on=['x', 'y']))


# ## Apply functions to data frames
# 
# Another useful primitive is `apply()`, which can apply a function to a data frame or to a series (column of the data frame).
# 
# For instance, suppose we wish to convert the year column in `C` into an abbrievated two-digit form. The following code will do it:

# In[11]:


display(C)
G = C.copy() # If you do not use copy function the original data frame is modified
G['year'] = G['year'].apply(lambda x: "'{:02d}".format(x % 100))
display(G)


# **Exercise 2** (2 points). Suppose you wish to compute the prevalence, which is the ratio of cases to the population.
# 
# The simplest way to do it is as follows:
# 
# ```python
#     G['prevalence'] = G['cases'] / G['population']
# ```
# 
# However, for this exercise, try to figure out how to use `apply()` to do it instead. To figure that out, you'll need to consult the documentation for [`apply()`](http://pandas.pydata.org/pandas-docs/stable/generated/pandas.DataFrame.apply.html) or go online to find some hints.
# 
# Implement your solution in a function, `calc_prevalence(G)`, which given `G` returns a **new copy** `H` that has a column named `'prevalence'` holding the correctly computed prevalence values.
# 
# > **Note 0.** The emphasis on "new copy" is there to remind you that your function should *not* modify the input dataframe, `G`.
# >
# > **Note 1.** Although there is the easy solution above, the purpose of this exercise is to force you to learn more about how `apply()` works, so that you can "apply" it in more settings in the future.

# In[12]:


def calc_prevalence(G):
    assert 'cases' in G.columns and 'population' in G.columns
    ###
    
    H = G.copy() # If you do not use copy function the original data frame is modified
    
    H['prevalence'] = H.apply(lambda row: row.cases/row.population, axis = 1)
    
    return H

    ###


# In[13]:


# Test cell: `prevalence_test`

G_copy = G.copy()
H = calc_prevalence(G)
display(H) # Displayed `H` should have a 'prevalence' column

assert (G == G_copy).all().all(), "Did your function modify G? It shouldn't..."
assert set(H.columns) == (set(G.columns) | {'prevalence'}), "Check `H` again: it should have the same columns as `G` plus a new column, `prevalence`."

Easy_prevalence_method = G['cases'] / G['population']
assert (H['prevalence'] == Easy_prevalence_method).all(), "One or more prevalence values is incorrect."

print("Prevalance values seem correct. But did you use `apply()?` Let's see...")

# Tests that you actually used `apply()` in your function:
def apply_fail():
    raise ValueError("Did you really use apply?")
    
setattr(pd.DataFrame, 'apply', apply_fail)

try:
    calc_prevalence(G)
except (ValueError, TypeError):
    print("You used `apply()`. You may have even used it as intended.")
else:
    assert False, "Are you sure you used `apply()`?"
finally:
    setattr(pd.DataFrame, 'apply', SAVE_APPLY)

print("\n(Passed!)")


# ## Part 3 : Tibbles and Bits 
# 
# Now let's start creating and manipulating tibbles.

# In[14]:


import pandas as pd  # The suggested idiom
from io import StringIO

from IPython.display import display # For pretty-printing data frames


# **Exercise 3** (3 points). Write a function, `canonicalize_tibble(X)`, that, given a tibble `X`, returns a new copy `Y` of `X` in _canonical order_. We say `Y` is in canonical order if it has the following properties.
# 
# 1. The variables appear in sorted order by name, ascending from left to right.
# 2. The rows appear in lexicographically sorted order by variable, ascending from top to bottom.
# 3. The row labels (`Y.index`) go from 0 to `n-1`, where `n` is the number of observations.
# 
# For instance, here is a **non-canonical tibble** ...
# 
# |   |  c  | a | b |
# |:-:|:---:|:-:|:-:|
# | 2 | hat | x | 1 |
# | 0 | rat | y | 4 |
# | 3 | cat | x | 2 |
# | 1 | bat | x | 2 |
# 
# 
# ... and here is its **canonical counterpart.**
# 
# |   | a | b |  c  |
# |:-:|:-:|:-:|:---:|
# | 0 | x | 1 | hat |
# | 1 | x | 2 | bat |
# | 2 | x | 2 | cat |
# | 3 | y | 4 | rat |
# 
# A partial solution appears below, which ensures that Property 1 above holds. Complete the solution to ensure Properties 2 and 3 hold. Feel free to consult the [Pandas API](http://pandas.pydata.org/pandas-docs/stable/api.html).
# 
# > **Hint**. For Property 3, you may find `reset_index()` handy: https://pandas.pydata.org/pandas-docs/stable/generated/pandas.DataFrame.reset_index.html

# In[15]:


def canonicalize_tibble(X):
    # Enforce Property 1:
    var_names = sorted(X.columns)
    Y = X[var_names].copy()
    
    
    Y.sort_values(by = var_names, inplace = True) 
    
    Z = Y.set_index([list(range(0, len(Y)))])
    ###
    return Z


# In[16]:


# Test: `canonicalize_tibble_test`

# Test input
canonical_in_csv = """,c,a,b
2,hat,x,1
0,rat,y,4
3,cat,x,2
1,bat,x,2"""

with StringIO(canonical_in_csv) as fp:
    canonical_in = pd.read_csv(fp, index_col=0)
print("=== Input ===")
display(canonical_in)
print("")
    
# Test output solution
canonical_soln_csv = """,a,b,c
0,x,1,hat
1,x,2,bat
2,x,2,cat
3,y,4,rat"""

with StringIO(canonical_soln_csv) as fp:
    canonical_soln = pd.read_csv(fp, index_col=0)
print("=== True solution ===")
display(canonical_soln)
print("")

canonical_out = canonicalize_tibble(canonical_in)
print("=== Your computed solution ===")
display(canonical_out)
print("")

canonical_matches = (canonical_out == canonical_soln)
print("=== Matches? (Should be all True) ===")
display(canonical_matches)
assert canonical_matches.all().all()

print ("\n(Passed.)")


# **Exercise 4** (1 point). Write a function, `tibbles_are_equivalent(A, B)` to determine if two tibbles, `A` and `B`, are equivalent. "Equivalent" means that `A` and `B` have identical variables and observations, up to permutations. If `A` and `B` are equivalent, then the function should return `True`. Otherwise, it should return `False`.
# 
# The last condition, "up to permutations," means that the variables and observations might not appear in the table in the same order. For example, the following two tibbles are equivalent:
# 
# 
# | a | b |  c  |
# |:-:|:-:|:---:|
# | x | 1 | hat |
# | y | 2 | cat |
# | z | 3 | bat |
# | w | 4 | rat |
# 
# | b |  c  | a |
# |:-:|:---:|:-:|
# | 2 | cat | y |
# | 3 | bat | z |
# | 1 | hat | x |
# | 4 | rat | w |
# 
# By contrast, the following table would not be equivalent to either of the above tibbles.
# 
# | a | b |  c  |
# |:-:|:-:|:---:|
# | 2 | y | cat |
# | 3 | z | bat |
# | 1 | x | hat |
# | 4 | w | rat |
# 
# > **Note**: Unlike Pandas data frames, tibbles conceptually do not have row labels. So you should ignore row labels.

# In[17]:


def tibbles_are_equivalent(A, B):
    """Given two tidy tables ('tibbles'), returns True iff they are
    equivalent.
    """
    # Check if the variables (keys) are the same in both tables
    if set(A.keys()) != set(B.keys()):
        return False
    
    # Check if the number of observations is the same
    if len(A[next(iter(A))]) != len(B[next(iter(B))]):
        return False
    
    # Check if the values for each variable are equivalent up to permutations
    for key in A.keys():
        if sorted(A[key]) != sorted(B[key]):
            return False
    
    # If all checks passed, the tibbles are equivalent
    return True
    ###


# In[18]:


# Test: `tibble_are_equivalent_test`

A = pd.DataFrame(columns=['a', 'b', 'c'],
                 data=list(zip (['x', 'y', 'z', 'w'],
                                [1, 2, 3, 4],
                                ['hat', 'cat', 'bat', 'rat'])))
print("=== Tibble A ===")
display(A)

# Permute rows and columns, preserving equivalence
import random

obs_ind_orig = list(range(A.shape[0]))
var_names = list(A.columns)

obs_ind = obs_ind_orig.copy()
while obs_ind == obs_ind_orig:
    random.shuffle(obs_ind)
    
while var_names == list(A.columns):
    random.shuffle(var_names)

B = A[var_names].copy()
B = B.iloc[obs_ind].reset_index(drop=True)

print ("=== Tibble B == A ===")
display(B)

print ("=== Tibble C != A ===")
C = A.copy()
C.columns = var_names
display(C)

assert tibbles_are_equivalent(A, B)
assert not tibbles_are_equivalent(A, C)
assert not tibbles_are_equivalent(B, C)

print ("\n(Passed.)")


# # Basic tidying transformations: Melting and casting
# 
# Given a data set and a target set of variables, there are at least two common issues that require tidying.
# ## Melting
# First, values often appear as columns. Table 4a is an example. To tidy up, you want to turn columns into rows:
# 
# ![Gather example](http://r4ds.had.co.nz/images/tidy-9.png)
# 
# Because this operation takes columns into rows, making a "fat" table more tall and skinny, it is sometimes called _melting_.
# 
# 
# 

# To melt the table, you need to do the following.
# 
# 1. Extract the _column values_ into a new variable. In this case, columns `"1999"` and `"2000"` of `table4` need to become the values of the variable, `"year"`.
# 2. Convert the values associated with the column values into a new variable as well. In this case, the values formerly in columns `"1999"` and `"2000"` become the values of the `"cases"` variable.
# 
# In the context of a melt, let's also refer to `"year"` as the new _key_ variable and `"cases"` as the new _value_ variable.

# **Exercise 5** (4 points). Implement the melt operation as a function,
# 
# ```python
#     def melt(df, col_vals, key, value):
#         ...
# ```
# 
# It should take the following arguments:
# - `df`: the input data frame, e.g., `table4` in the example above;
# - `col_vals`: a list of the column names that will serve as values;  column `1999` & `2000` in example  table
# - `key`: name of the new variable, e.g., `year` in the example above;
# - `value`: name of the column to hold the values. `cases` in the example above
# 
# > You may need to refer to the Pandas documentation to figure out how to create and manipulate tables. The bits related to [indexing](http://pandas.pydata.org/pandas-docs/stable/indexing.html) and [merging](http://pandas.pydata.org/pandas-docs/stable/merging.html) may be especially helpful.

# In[19]:


def melt(df, col_vals, key, value):
    assert type(df) is pd.DataFrame
    ###
    df = df.melt(id_vars=['country'], var_name=key, value_name=value)

    return df
    ###


# In[20]:


# Test: `melt_test`

table4a = pd.read_csv(local_data['table4a.csv'])
print("\n=== table4a ===")
display(table4a)

m_4a = melt(table4a, col_vals=['1999', '2000'], key='year', value='cases')
print("=== melt(table4a) ===")
display(m_4a)

table4b = pd.read_csv(local_data['table4b.csv'])
print("\n=== table4b ===")
display(table4b)

m_4b = melt(table4b, col_vals=['1999', '2000'], key='year', value='population')
print("=== melt(table4b) ===")
display(m_4b)

m_4 = pd.merge(m_4a, m_4b, on=['country', 'year'])
print ("\n=== inner-join(melt(table4a), melt (table4b)) ===")
display(m_4)

m_4['year'] = m_4['year'].apply (int)

table1 = pd.read_csv(local_data['table1.csv'])
print ("=== table1 (target solution) ===")
display(table1)
assert tibbles_are_equivalent(table1, m_4)
print ("\n(Passed.)")


# ## Casting
# The second most common issue is that an observation might be split across multiple rows. Table 2 is an example. To tidy up, you want to merge rows:
# 
# ![Spread example](http://r4ds.had.co.nz/images/tidy-8.png)
# 
# Because this operation is the moral opposite of melting, and "rebuilds" observations from parts, it is sometimes called _casting_.
# 
# <!--- link to original paper is broken [his original paper on tidying data](http://www.jstatsoft.org/v59/i10/paper) -->
# > Melting and casting are Wickham's terms from his original paper on tidying data. In his more recent writing, [on which this tutorial is based](http://r4ds.had.co.nz/tidy-data.html), he refers to the same operation as _gathering_. Again, this term comes from Wickham's original paper, whereas his more recent summaries use the term _spreading_.
# 
# The signature of a cast is similar to that of melt. However, you only need to know the `key`, which is column of the input table containing new variable names, and the `value`, which is the column containing corresponding values.

# **Exercise 6** (4 points). Implement a function to cast a data frame into a tibble, given a key column containing new variable names and a value column containing the corresponding cells.
# 
# We've given you a partial solution that
# 
# - verifies that the given `key` and `value` columns are actual columns of the input data frame;
# - computes the list of columns, `fixed_vars`, that should remain unchanged; and
# - initializes and empty tibble.
# 
# Observe that we are asking your `cast()` to accept an optional parameter, `join_how`, that may take the values `'outer'` or `'inner'` (with `'outer'` as the default). Why do you need such a parameter?

# In[21]:


def cast(df, key, value, join_how='outer'):
    """Casts the input data frame into a tibble,
    given the key column and value column.
    """
    assert type(df) is pd.DataFrame
    assert key in df.columns and value in df.columns
    assert join_how in ['outer', 'inner']
    
    fixed_vars = df.columns.difference([key, value])
    tibble = pd.DataFrame(columns=fixed_vars) # empty frame
    
    ###
    df = df.pivot(index=['country','year'],columns=key,values=value)
    df = df.reset_index()
    ###
    
    return df


# In[22]:


# Test: `cast_test`

table2 = pd.read_csv(local_data['table2.csv'])
print('=== table2 ===')
display(table2)

print('\n=== tibble2 = cast (table2, "type", "count") ===')
tibble2 = cast(table2, 'type', 'count')
display(tibble2)

assert tibbles_are_equivalent(table1, tibble2)
print('\n(Passed.)')


# ## Separating variables
# 
# Consider the following table.

# In[23]:


table3 = pd.read_csv(local_data['table3.csv'])
display(table3)


# In this table, the `rate` variable combines what had previously been the `cases` and `population` data. This example is an instance in which we might want to _separate_ a column into two variables.

# **Exercise 7A** (3 points). Write a function that takes a data frame (`df`) and separates an existing column (`key`) into new variables (given by the list of new variable names, `into`).
# 
# How will the separation happen? The caller should provide a function, `splitter(x)`, that given a value returns a _list_ containing the components. Observe that the partial solution below defines a default splitter, which uses the regular expression, `(\d+\.?\d+)`, to find all integer or floating-point values in a string input `x`.

# In[24]:


import re

def default_splitter(text):
    """Searches the given spring for all integer and floating-point
    values, returning them as a list _of strings_.
    
    E.g., the call
    
      default_splitter('Give me $10.52 in exchange for 91 kitten stickers.')
      
    will return ['10.52', '91'].
    """
    fields = re.findall('(\d+\.?\d+)', text)
    return fields

def separate(df, key, into, splitter=default_splitter):
    """Given a data frame, separates one of its columns, the key,
    into new variables.
    """
    assert type(df) is pd.DataFrame
    assert key in df.columns
    
    # Hint: http://stackoverflow.com/questions/16236684/apply-pandas-function-to-column-to-create-multiple-new-columns
    
    ###
    df[['cases', 'population']] = df['rate'].str.split('/', expand=True)
    del df['rate']
    
    return df
    ###


# In[25]:


# Test: `separate_test`

print("=== Recall: table3 ===")
display(table3)

tibble3 = separate(table3, key='rate', into=['cases', 'population'])
print("\n=== tibble3 = separate (table3, ...) ===")
display(tibble3)

assert 'cases' in tibble3.columns
assert 'population' in tibble3.columns
assert 'rate' not in tibble3.columns

tibble3['cases'] = tibble3['cases'].apply(int)
tibble3['population'] = tibble3['population'].apply(int)

assert tibbles_are_equivalent(tibble3, table1)
print("\n(Passed.)")


# **Exercise 7B** (2 points). Implement the inverse of separate, which is `unite`. This function should take a data frame (`df`), the set of columns to combine (`cols`), the name of the new column (`new_var`), and a function that takes the subset of the `cols` variables from a single observation. It should return a new value for that observation.

# In[26]:


def str_join_elements(x, sep=""):
#     print("x - to be united:", x)
    assert type(sep) is str
    yy = sep.join([str(xi) for xi in x])
#     print("Separated into string as: ",  yy)
    return yy

def unite(df, cols, new_var, combine=str_join_elements):
    # Hint: http://stackoverflow.com/questions/13331698/how-to-apply-a-function-to-two-columns-of-pandas-dataframe
    ###
    df[new_var] = df[cols].apply(
    lambda x: ''.join(x.dropna().astype(str)),
    axis=1
    )
                     
    return df
    ###


# In[27]:


# Test: `unite_test`

table3_again = unite(tibble3, ['cases', 'population'], 'rate',
                     combine=lambda x: str_join_elements(x, "/"))
display(table3_again)
assert tibbles_are_equivalent(table3, table3_again)

print("\n(Passed.)")


# 
# # Putting it all together #
# 
# Let's use primitives to tidy up the original WHO TB data set. First, here is the raw data.
# 

# In[28]:


who_raw = pd.read_csv(local_data['who.csv'])

print("=== WHO TB data set: {} rows x {} columns ===".format(who_raw.shape[0],
                                                              who_raw.shape[1]))
print("Column names:", who_raw.columns)

print("\n=== A few randomly selected rows ===")
import random
row_sample = sorted(random.sample(range(len(who_raw)), 5))
display(who_raw.iloc[row_sample])


# The data set has 7,240 rows and 60 columns. Here is how to decode the columns.
# - Columns `'country'`, `'iso2'`, and `'iso3'` are different ways to designate the country and redundant, meaning you only really need to keep one of them.
# - Column `'year'` is the year of the report and is a natural variable.
# - Among columns `'new_sp_m014'` through `'newrel_f65'`, the `'new...'` prefix indicates that the column's values count new cases of TB. In this particular data set, all the data are for new cases.
# - The short codes, `rel`, `ep`, `sn`, and `sp` describe the type of TB case. They stand for relapse, extrapulmonary, pulmonary not detectable by a pulmonary smear test ("smear negative"), and pulmonary detectable by such a test ("smear positive"), respectively.
# - The codes `'m'` and `'f'` indicate the gender (male and female, respectively).
# - The trailing numeric code indicates the age group: `014` is 0-14 years of age, `1524` for 15-24 years, `2534` for 25-34 years, etc., and `65` stands for 65 years or older.
# 
# In other words, it looks like you are likely to want to treat all the columns as values of multiple variables!

# **Exercise 8** (3 points). As a first step, start with `who_raw` and create a new data frame, `who2`, with the following properties:
# 
# - All the `'new...'` columns of `who_raw` become values of a _single_ variable, `case_type`. Store the counts associated with each `case_type` value as a new variable called `'count'`.
# - Remove the `iso2` and `iso3` columns, since they are redundant with `country` (which you should keep!).
# - Keep the `year` column as a variable.
# - Remove all not-a-number (`NaN`) counts. _Hint_: You can test for a `NaN` using Python's [`math.isnan()`](https://docs.python.org/3/library/math.html).
# - Convert the counts to integers. (Because of the presence of NaNs, the counts will be otherwise be treated as floating-point values, which is undesirable since you do not expect to see non-integer counts.)

# In[29]:


from math import isnan
import numpy as np

who2 = who_raw.copy()

del who2['iso2']
del who2['iso3']

who2 = who2.melt(id_vars=['country', 'year'], var_name='case_type', value_name='count')

who2 = who2[who2['count'].notna()]

who2['count'] = who2['count'].apply(np.int64)

print(who2)



# In[30]:


# Test: `who2_test`

print("=== First few rows of your solution ===")
display(who2.head())

print ("=== First few rows of the instructor's solution ===")
who2_soln = pd.read_csv(local_data['who2_soln.csv'])
display(who2_soln.head())

# Check it
assert tibbles_are_equivalent(who2, who2_soln)
print ("\n(Passed.)")


# **Exercise 9** (5 points). Starting from your `who2` data frame, create a new tibble, `who3`, for which each `'key'` value is split into three new variables:
# - `'type'`, to hold the TB type, having possible values of `rel`, `ep`, `sn`, and `sp`;
# - `'gender'`, to hold the gender as a string having possible values of `female` and `male`; and
# - `'age_group'`, to hold the age group as a string having possible values of `0-14`, `15-24`, `25-34`, `35-44`, `45-54`, `55-64`, and `65+`.
# 
# > The input data file is large enough that your solution might take a minute to run. But if it appears to be taking much more than that, you may want to revisit your approach.

# In[34]:


import re
import pandas as pd

def extract_who_components(text):
    # If 'new-val' starts, followed by possible values
    # Then _, and finally f or m and 2, 3, 4 digits
    m = re.match("^new_?(rel|ep|sn|sp)_(f|m)(\d{2,4})$", text)
    
    # If there is no match or not exactly 3 matches, return three empty strings
    if m is None or len(m.groups()) != 3:
        return ('', '', '')
    
    fields = list(m.groups())
    
    # Map 'f' to 'female' and 'm' to 'male'
    if fields[1] == 'f':
        fields[1] = 'female'
    elif fields[1] == 'm':
        fields[1] = 'male'
    
    # Map age groups
    if fields[2] == '014':
        fields[2] = '0-14'
    elif fields[2] == '65':
        fields[2] = '65+'
    elif len(fields[2]) == 4 and fields[2].isdigit():
        fields[2] = fields[2][:2] + '-' + fields[2][2:4]
    
    return fields

# Apply the extract_who_components function to the 'case_type' column
who3 = who2['case_type'].apply(extract_who_components).apply(pd.Series)

# Rename the columns
who3.columns = ['type', 'gender', 'age_group']

# Concatenate who3 with the original who2 dataframe
who3 = pd.concat([who2, who3], axis=1)

# Drop the original 'case_type' column
who3.drop('case_type', axis=1, inplace=True)

who3.head()


# In[35]:


who3['age_group'].unique()


# In[36]:


# Test: `who3_test`

print("=== First few rows of your solution ===")
display(who3.head())

who3_soln = pd.read_csv(local_data['who3_soln.csv'])
print("\n=== First few rows of the instructor's solution ===")
display(who3_soln.head())

assert tibbles_are_equivalent(who3, who3_soln)
print("\n(Passed.)")


# **Fin!** That's the end of this assignment. Don't forget to restart and run this notebook from the beginning to verify that it works top-to-bottom before submitting.

# In[ ]:




