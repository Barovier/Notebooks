
# coding: utf-8

# # Part 1: NYC 311 calls
# 
# This notebook derives from a [demo by the makers of plot.ly](https://plot.ly/ipython-notebooks/big-data-analytics-with-pandas-and-sqlite/). We've adapted it to use [Bokeh (and HoloViews)](http://bokeh.pydata.org/en/latest/).
# 
# You will start with a large database of complaints filed by residents of New York City via 311 calls. The full dataset is available at the [NYC open data portal](https://data.cityofnewyork.us/Social-Services/311-Service-Requests-from-2010-to-Present/erm2-nwe9). Our subset is about 6 GB and 10 million complaints, so you can infer that a) you might not want to read it all into memory at once, and b) NYC residents have a lot to complain about. (Maybe only conclusion "a" is valid.) The notebook then combines the use of `sqlite`, `pandas`, and `bokeh`.

# ## Module setup
# 
# Before diving in, run the following cells to preload some functions you'll need later. These include a few functions from Notebook 7.

# In[212]:


import sys
print(sys.version) # Print Python version -- On Vocareum, it should be 3.7+

from IPython.display import display
import pandas as pd

from nb7utils import canonicalize_tibble, tibbles_are_equivalent, cast


# Lastly, some of the test cells will need some auxiliary files, which the following code cell will check for and, if they are missing, download.

# In[213]:


from nb9utils import download, get_path, auxfiles

for filename, checksum in auxfiles.items():
    download(filename, checksum=checksum, url_suffix="lab9-sql/")
    
print("(Auxiliary files appear to be ready.)")


# In[214]:


from nb9utils import make_barchart, make_stacked_barchart
from bokeh.io import show


# In[215]:


def demo_bar():
    from bokeh.plotting import figure
    from bokeh.models import ColumnDataSource
    data = [
        ['201720', 'cat1', 20],
        ['201720', 'cat2', 30],
        ['201720', 'cat3', 40],
        ['201721', 'cat1', 20],
        ['201721', 'cat2', 0],
        ['201721', 'cat3', 40],
        ['201722', 'cat1', 50],
        ['201722', 'cat2', 60],
        ['201722', 'cat3', 10],
    ]
    df = pd.DataFrame(data, columns=['week', 'category', 'count'])
    pt = df.pivot('week', 'category', 'count')
    pt.cumsum(axis=1)
    return df, pt

df_demo, pt_demo = demo_bar()
pt_demo


# ## Viz setup
# 
# This notebook includes some simple visualizations. This section just ensures you have the right software setup to follow along.

# In[216]:


def demo_stacked_bar(pt):
    from bokeh.models.ranges import FactorRange
    from bokeh.io import show
    from bokeh.plotting import figure
    p = figure(title="count",
               x_axis_label='week', y_axis_label='category',
               x_range = FactorRange(factors=list(pt.index)),
               plot_height=500, plot_width=800)
    p.vbar(x=pt.index, bottom=0, top=pt.cat1, width=0.2, color='red', legend='cat1')
    p.vbar(x=pt.index, bottom=pt.cat1, top=pt.cat2+pt.cat1, width=0.2, color='blue', legend='cat2')
    p.vbar(x=pt.index, bottom=pt.cat2+pt.cat1,top=pt.cat3+pt.cat2+pt.cat1, width=0.2, color='green', legend='cat3')
    return p
    
show(demo_stacked_bar(pt_demo))


# In[217]:


# Build a Pandas data frame
names = ['Bob','Jessica','Mary','John','Mel']
births = [968, 155, 77, 578, 973]
name_birth_pairs = list(zip(names, births))
baby_names = pd.DataFrame(data=name_birth_pairs, columns=['Names', 'Births'])
display(baby_names)


# In[218]:


p = make_barchart(baby_names, 'Names', 'Births', kwargs_figure={'plot_width': 640, 'plot_height': 320})
show(p)


# ## Data setup
# 
# You'll also need the NYC 311 calls dataset. What we've provided is actually a small subset (about 250+ MiB) of the full data as of 2015.
# 
# > If you are not running on Vocareum, you will need to download this file manually from the following link and place it locally in a (nested) subdirectory or folder named `resource/asnlib/publicdata`.
# >
# > [Link to the pre-constructed NYC 311 Database on MS OneDrive](https://onedrive.live.com/download?cid=FD520DDC6BE92730&resid=FD520DDC6BE92730%21616&authkey=AEeP_4E1uh-vyDE)

# In[219]:


from nb9utils import download_nyc311db
DB_FILENAME = download_nyc311db()


# **Connecting.** Let's open up a connection to this dataset.

# In[220]:


# Connect
import sqlite3 as db
disk_engine = db.connect('file:{}?mode=ro'.format(DB_FILENAME), uri=True)


# **Preview the data.** This sample database has just a single table, named `data`. Let's query it and see how long it takes to read. To carry out the query, we will use the SQL reader built into `pandas`.

# In[221]:


import time

print ("Reading ...")
start_time = time.time ()

# Perform SQL query through the disk_engine connection.
# The return value is a pandas data frame.
df = pd.read_sql_query ('select * from data', disk_engine)

elapsed_time = time.time () - start_time
print ("==> Took %g seconds." % elapsed_time)

# Dump the first few rows
df.head()


# **Partial queries: `LIMIT` clause.** The preceding command was overkill for what we wanted, which was just to preview the table. Instead, we could have used the `LIMIT` option to ask for just a few results.

# In[222]:


query = '''
  SELECT *
    FROM data
    LIMIT 5
'''
start_time = time.time ()
df = pd.read_sql_query (query, disk_engine)
elapsed_time = time.time () - start_time
print ("==> LIMIT version took %g seconds." % elapsed_time)

df


# **Finding unique values: `DISTINCT` qualifier.** Another common idiom is to ask for the unique values of some attribute, for which you can use the `DISTINCT` qualifier.

# In[223]:


query = 'SELECT DISTINCT City FROM data'
df = pd.read_sql_query(query, disk_engine)

print("Found {} unique cities. The first few are:".format(len(df)))
df.head()


# However, `DISTINCT` applied to strings is case-sensitive. We'll deal with that momentarily.

# **Grouping Information: GROUP BY operator.** The GROUP BY operator lets you group information using a particular column or multiple columns of the table. The output generated is more of a pivot table.

# In[224]:


query = '''
  SELECT ComplaintType, Descriptor, Agency
    FROM data
    GROUP BY ComplaintType
'''

df = pd.read_sql_query(query, disk_engine)
print(df.shape)
df.head()


# **`GROUP BY` aggregations.** A common pattern is to combine grouping with aggregation. For example, suppose we want to count how many times each complaint occurs. Here is one way to do it.

# In[225]:


query = '''
  SELECT ComplaintType, COUNT(*)
    FROM data
    GROUP BY ComplaintType
    LIMIT 10
'''

df = pd.read_sql_query(query, disk_engine)
df.head()


# **Character-case conversions.** From the two preceding examples, observe that the strings employ a mix of case conventions (i.e., lowercase vs. uppercase vs. mixed case). A convenient way to query and "normalize" case is to apply SQL's `UPPER()` and `LOWER()` functions. Here is an example:

# In[226]:


query = '''
  SELECT LOWER(ComplaintType), LOWER(Descriptor), LOWER(Agency)
    FROM data
    GROUP BY LOWER(ComplaintType)
    LIMIT 10
'''

df = pd.read_sql_query(query, disk_engine)
df.head()


# **Filtered aggregations: `HAVING` clauses.** A common pattern for aggregation queries (e.g., `GROUP BY` plus `COUNT()`) is to filter the grouped results. You cannot do that with a `WHERE` clause alone, because `WHERE` is applied *before* grouping.
# 
# As an example, recall that some `ComplaintType` values are in all uppercase whereas some use mixed case. Since we didn't inspect all of them, there might even be some are all lowercase. Worse, you would expect some inconsistencies. For instance, it turns out that both `"Plumbing"` (mixed case) and `"PLUMBING"` (all caps) appear. Here is a pair of queries that makes this point.

# In[227]:


query0 = "SELECT DISTINCT ComplaintType FROM data"
df0 = pd.read_sql_query(query0, disk_engine)
print("Found {} unique `ComplaintType` strings.".format(len(df0)))
display(df0.head())

query1 = "SELECT DISTINCT LOWER(ComplaintType) FROM data"
df1 = pd.read_sql_query(query1, disk_engine)
print("\nFound {} unique `LOWER(ComplaintType)` strings.".format(len(df1)))
display(df1.head())

print("\n==> Therefore, there are {} cases that are duplicated. Which ones?".format(len(df0) - len(df1)))


# What if we wanted a query that identifies these inconsistent capitalizations? Here is one way to do it, which demonstrates the `HAVING` clause. (It also uses a **nested query**, that is, it performs one query and then selects immediately from that result.) Can you read it and figure out what it is doing and why it works?

# In[228]:


query2 = '''
    SELECT ComplaintType, COUNT(*)
      FROM (SELECT DISTINCT ComplaintType FROM data)
      GROUP BY LOWER(ComplaintType)
      HAVING COUNT(*) >= 2
'''
df2 = pd.read_sql_query(query2, disk_engine)
df2


# You should see that "elevator" and "plumbing" complaints use inconsistent case, which we can then verify directly using the next technique, the `IN` operator.

# **Set membership: `IN` operator.** Another common idiom is to ask for rows whose attributes fall within a set, for which you can use the `IN` operator. Let's use it to see the two inconsistent-capitalization complaint types from above.

# In[229]:


query = '''
    SELECT DISTINCT ComplaintType
      FROM data
      WHERE LOWER(ComplaintType) IN ("plumbing", "elevator")
'''
df = pd.read_sql_query(query, disk_engine)
df.head()


# **Renaming columns: `AS` operator.** Sometimes you might want to rename a result column. For instance, the following query counts the number of complaints by "Agency," using the `COUNT(*)` function and `GROUP BY` clause, which we discussed in an earlier lab. If you wish to refer to the counts column of the resulting data frame, you can give it a more "friendly" name using the `AS` operator.

# In[230]:


query = '''
  SELECT Agency, COUNT(*) AS NumComplaints
    FROM data
    GROUP BY Agency
'''
df = pd.read_sql_query(query, disk_engine)
df.head()


# **Ordering results: `ORDER BY` clause.** You can also order the results. For instance, suppose we want to execute the previous query by number of complaints.

# In[231]:


query = '''
  SELECT Agency, COUNT(*) AS NumComplaints
    FROM data
    GROUP BY UPPER(Agency)
    ORDER BY NumComplaints
'''
df = pd.read_sql_query(query, disk_engine)
df.tail()


# Note that the above example prints the bottom (tail) of the data frame. You could have also asked for the query results in reverse (descending) order, by prefixing the `ORDER BY` attribute with a `-` (minus) symbol. Alternatively, you can use `DESC` to achieve the same result.

# In[232]:


query = '''
  SELECT Agency, COUNT(*) AS NumComplaints
    FROM data
    GROUP BY UPPER(Agency)
    ORDER BY -NumComplaints
'''

# Alternative: query =
'''
SELECT Agency, COUNT(*) AS NumComplaints 
    FROM data 
    GROUP BY UPPER(Agency)
    ORDER BY NumComplaints DESC 
'''

df = pd.read_sql_query(query, disk_engine)
df.head()


# And of course we can plot all of this data!
# 
# **Exercise 0** (ungraded). Run the following code cell, which will create an interactive bar chart from the data in the previous query.

# In[233]:


p = make_barchart(df[:20], 'Agency', 'NumComplaints',
                  {'title': 'Top 20 agencies by number of complaints',
                   'plot_width': 800, 'plot_height': 320})
p.xaxis.major_label_orientation = 0.66
show(p)


# **Exercise 1** (2 points). Create a string, `query`, containing an SQL query that will return the number of complaints by type. The columns should be named `type` and `freq`, and the results should be sorted in descending order by `freq`. Also, since we know some complaints use an inconsistent case, for your function convert complaints to lowercase.
# 
# > What is the most common type of complaint? What, if anything, does it tell you about NYC?

# In[234]:


del query# clears any existing `query` variable; you should define it, below!

# Define a variable named `query` containing your solution
###
query = '''
  SELECT lower(ComplaintType) AS type, COUNT(*) AS freq
    FROM data
    GROUP BY lower(ComplaintType)
    ORDER BY -freq
'''
###
# Runs your `query`:
df_complaint_freq = pd.read_sql_query(query, disk_engine)
df_complaint_freq.head()


# In[235]:


# Test cell: `complaints_test`

print("Top 10 complaints:")
display(df_complaint_freq.head(10))

assert set(df_complaint_freq.columns) == {'type', 'freq'}, "Output columns should be named 'type' and 'freq', not {}".format(set(df_complaint_freq.columns))

soln = ['heat/hot water', 'street condition', 'street light condition', 'blocked driveway', 'illegal parking', 'unsanitary condition', 'paint/plaster', 'water system', 'plumbing', 'noise', 'noise - street/sidewalk', 'traffic signal condition', 'noise - commercial', 'door/window', 'water leak', 'dirty conditions', 'sewer', 'sanitation condition', 'dof literature request', 'electric', 'rodent', 'flooring/stairs', 'general construction/plumbing', 'building/use', 'broken muni meter', 'general', 'missed collection (all materials)', 'benefit card replacement', 'derelict vehicle', 'noise - vehicle', 'damaged tree', 'consumer complaint', 'derelict vehicles', 'taxi complaint', 'overgrown tree/branches', 'graffiti', 'snow', 'opinion for the mayor', 'appliance', 'maintenance or facility', 'animal abuse', 'dead tree', 'elevator', 'hpd literature request', 'root/sewer/sidewalk condition', 'safety', 'food establishment', 'scrie', 'air quality', 'agency issues', 'construction', 'highway condition', 'other enforcement', 'water conservation', 'sidewalk condition', 'indoor air quality', 'street sign - damaged', 'traffic', 'fire safety director - f58', 'homeless person assistance', 'homeless encampment', 'special enforcement', 'street sign - missing', 'noise - park', 'vending', 'for hire vehicle complaint', 'food poisoning', 'special projects inspection team (spit)', 'hazardous materials', 'electrical', 'dot literature request', 'litter basket / request', 'taxi report', 'illegal tree damage', 'dof property - reduction issue', 'unsanitary animal pvt property', 'asbestos', 'lead', 'vacant lot', 'dca / doh new license application request', 'street sign - dangling', 'smoking', 'violation of park rules', 'outside building', 'animal in a park', 'noise - helicopter', 'school maintenance', 'dpr internal', 'boilers', 'industrial waste', 'sweeping/missed', 'overflowing litter baskets', 'non-residential heat', 'curb condition', 'drinking', 'standing water', 'indoor sewage', 'water quality', 'eap inspection - f59', 'derelict bicycle', 'noise - house of worship', 'dca literature request', 'recycling enforcement', 'dof parking - tax exemption', 'broken parking meter', 'request for information', 'taxi compliment', 'unleashed dog', 'urinating in public', 'unsanitary pigeon condition', 'investigations and discipline (iad)', 'bridge condition', 'ferry inquiry', 'bike/roller/skate chronic', 'public payphone complaint', 'vector', 'best/site safety', 'sweeping/inadequate', 'disorderly youth', 'found property', 'mold', 'senior center complaint', 'fire alarm - reinspection', 'for hire vehicle report', 'city vehicle placard complaint', 'cranes and derricks', 'ferry complaint', 'illegal animal kept as pet', 'posting advertisement', 'harboring bees/wasps', 'panhandling', 'scaffold safety', 'oem literature request', 'plant', 'bus stop shelter placement', 'collection truck noise', 'beach/pool/sauna complaint', 'complaint', 'compliment', 'illegal fireworks', 'fire alarm - modification', 'dep literature request', 'drinking water', 'fire alarm - new system', 'poison ivy', 'bike rack condition', 'emergency response team (ert)', 'municipal parking facility', 'tattooing', 'unsanitary animal facility', 'animal facility - no permit', 'miscellaneous categories', 'misc. comments', 'literature request', 'special natural area district (snad)', 'highway sign - damaged', 'public toilet', 'adopt-a-basket', 'ferry permit', 'invitation', 'window guard', 'parking card', 'illegal animal sold', 'stalled sites', 'open flame permit', 'overflowing recycling baskets', 'highway sign - missing', 'public assembly', 'dpr literature request', 'fire alarm - addition', 'lifeguard', 'transportation provider complaint', 'dfta literature request', 'bottled water', 'highway sign - dangling', 'dhs income savings requirement', 'legal services provider complaint', 'foam ban enforcement', 'tunnel condition', 'calorie labeling', 'fire alarm - replacement', 'x-ray machine/equipment', 'sprinkler - mechanical', 'hazmat storage/use', 'tanning', 'radioactive material', 'rangehood', 'squeegee', 'srde', 'building condition', 'sg-98', 'standpipe - mechanical', 'agency', 'forensic engineering', 'public assembly - temporary', 'vacant apartment', 'laboratory', 'sg-99']
assert all(soln[:25] == df_complaint_freq['type'].iloc[:25])
assert df_complaint_freq.shape[0] == len(soln), 'Output has {} rows while the instructor solution has {} rows'.format(df_complaint_freq.shape[0], len(soln))

print("\n(Passed.)")


# Let's also visualize the result, as a bar chart showing complaint types on the x-axis and the number of complaints on the y-axis.

# In[236]:


p = make_barchart(df_complaint_freq[:25], 'type', 'freq',
                  {'title': 'Top 25 complaints by type',
                   'plot_width': 800, 'plot_height': 320})
p.xaxis.major_label_orientation = 0.66
show(p)


# # Lesson 3: More SQL stuff

# **Simple substring matching: the `LIKE` operator.** Suppose we just want to look at the counts for all complaints that have the word `noise` in them. You can use the `LIKE` operator combined with the string wildcard, `%`, to look for case-insensitive substring matches.

# In[237]:


query = '''
  SELECT LOWER(ComplaintType) AS type, COUNT(*) AS freq
    FROM data
    WHERE LOWER(ComplaintType) LIKE '%noise%'
    GROUP BY type
    ORDER BY -freq
'''

df_noisy = pd.read_sql_query(query, disk_engine)
print("Found {} queries with 'noise' in them.".format(len(df_noisy)))
df_noisy


# **Exercise 2** (2 points). Create a string variable, `query`, that contains an SQL query that will return the top 10 cities with the largest number of complaints, in descending order. It should return a table with two columns, one named `name` holding the name of the city, and one named `freq` holding the number of complaints by that city. 
# 
# Like complaint types, cities are not capitalized consistently. Therefore, standardize the city names by converting them to **uppercase**.

# In[238]:


del query # define a new `query` variable, below

# Define your `query`, here:
###
query = '''
  SELECT City AS name, COUNT(ComplaintType) AS freq
    FROM data
    GROUP BY UPPER(City)
    ORDER BY -freq
    LIMIT 10
'''

# Runs your `query`:
df_whiny_cities = pd.read_sql_query(query, disk_engine)
df_whiny_cities


# Brooklynites are "vocal" about their issues, evidently.

# In[239]:


# Test cell: `whiny_cities__test`

assert df_whiny_cities['name'][0] == 'BROOKLYN'
assert df_whiny_cities['name'][1] == 'NEW YORK'
assert df_whiny_cities['name'][2] == 'BRONX'
assert df_whiny_cities['name'][3] is None
assert df_whiny_cities['name'][4] == 'STATEN ISLAND'

print ("\n(Passed partial test.)")


# **Case-insensitive grouping: `COLLATE NOCASE`.** Another way to carry out the preceding query in a case-insensitive way is to add a `COLLATE NOCASE` qualifier to the `GROUP BY` clause.
# 
# The next example demonstrates this clause. Note that it also filters out the 'None' cases, where the `<>` operator denotes "not equal to." Lastly, this query ensures that the returned city names are uppercase.
# 
# > The `COLLATE NOCASE` clause modifies the column next to which it appears. So if you are grouping by more than one key and want to be case-insensitive, you need to write, `... GROUP BY ColumnA COLLATE NOCASE, ColumnB COLLATE NOCASE ...`.

# In[240]:


query = '''
  SELECT UPPER(City) AS name, COUNT(*) AS freq
    FROM data
    WHERE name <> 'None'
    GROUP BY City COLLATE NOCASE
    ORDER BY -freq
    LIMIT 10
'''
df_whiny_cities2 = pd.read_sql_query(query, disk_engine)
df_whiny_cities2


# Lastly, for later use, let's save the names of just the top seven (7) cities by numbers of complaints.

# In[241]:


TOP_CITIES = list(df_whiny_cities2.head(7)['name'])
TOP_CITIES


# **Exercise 3** (1 point). Implement a function that takes a list of strings, `str_list`, and returns a single string consisting of each value, `str_list[i]`, enclosed by double-quotes and separated by a comma-space delimiters. For example, if
# 
# ```python
#    assert str_list == ['a', 'b', 'c', 'd']
# ```
# 
# then
# 
# ```python
#    assert strs_to_args(str_list) == '"a", "b", "c", "d"'
# ```
# 
# > **Tip.** Try to avoid manipulating the input `str_list` directly and returning the updated `str_list`. This may result in your function adding `""` to the strings in your list each time the function is used (which will be more than once in this notebook!)

# In[242]:


def strs_to_args(str_list):
    assert type (str_list) is list
    assert all ([type (s) is str for s in str_list])
    ###
    
    lst = str_list.copy()
    
    
    return ', '.join(['"{}"'.format(s) for s in lst])

    ###


# In[243]:


# Test cell: `strs_to_args__test`

print ("Your solution, applied to TOP_CITIES:", strs_to_args(TOP_CITIES))

TOP_CITIES_as_args = strs_to_args(TOP_CITIES)
assert TOP_CITIES_as_args ==        '"BROOKLYN", "NEW YORK", "BRONX", "STATEN ISLAND", "Jamaica", "Flushing", "ASTORIA"'.upper()
assert TOP_CITIES == list(df_whiny_cities2.head(7)['name']),        "Does your implementation cause the `TOP_CITIES` variable to change? If so, you need to fix that."
    
print ("\n(Passed.)")


# **Exercise 4** (3 points). Suppose we want to look at the number of complaints by type _and_ by city **for only the top cities**, i.e., those in the list `TOP_CITIES` computed above. Execute an SQL query to produce a tibble named `df_complaints_by_city` with the variables {`complaint_type`, `city_name`, `complaint_count`}.
# 
# In your output `DataFrame`, convert all city names to uppercase and convert all complaint types to lowercase.

# In[244]:


###



query = '''SELECT LOWER(ComplaintType) AS complaint_type, UPPER(City) as city_name, COUNT(ComplaintType) AS complaint_count
            FROM data
            WHERE city_name IN ({})
            GROUP BY city_name, complaint_type
'''.format(TOP_CITIES_as_args)

df_complaints_by_city = pd.read_sql_query(query, disk_engine)

###

# Previews the results of your query:
print("Found {} records.".format(len(df_complaints_by_city)))
display(df_complaints_by_city.head(10))


# In[245]:


# Test cell: `df_complaints_by_city__test`

print("Reading instructor's solution...")
if False:
    df_complaints_by_city.to_csv(get_path('df_complaints_by_city_soln.csv'), index=False)
df_complaints_by_city_soln = pd.read_csv(get_path('df_complaints_by_city_soln.csv'))

print("Checking...")
assert tibbles_are_equivalent(df_complaints_by_city,
                              df_complaints_by_city_soln)

print("\n(Passed.)")
del df_complaints_by_city_soln


# Let's use Bokeh to visualize the results as a stacked bar chart.

# In[246]:


# Let's consider only the top 25 complaints (by total)
top_complaints = df_complaint_freq[:25]
print("Top complaints:")
display(top_complaints)


# In[247]:


# Plot subset of data corresponding to the top complaints
df_plot = top_complaints.merge(df_complaints_by_city,
                               left_on=['type'],
                               right_on=['complaint_type'],
                               how='left')
df_plot.dropna(inplace=True)
print("Data to plot (first few rows):")
display(df_plot.head())
print("...")


# In[248]:


# Some code to render a Bokeh stacked bar chart

kwargs_figure = {'title': "Distribution of the top 25 complaints among top 7 cities with the most complaints",
                 'width': 800,
                 'height': 400,
                 'tools': "hover,crosshair,pan,box_zoom,wheel_zoom,save,reset,help"}

def plot_complaints_stacked_by_city(df, y='complaint_count'):
    p = make_stacked_barchart(df, 'complaint_type', 'city_name', y,
                              x_labels=list(top_complaints['type']), bar_labels=TOP_CITIES,
                              kwargs_figure=kwargs_figure)
    p.xaxis.major_label_orientation = 0.66
    from bokeh.models import HoverTool
    hover_tool = p.select(dict(type=HoverTool))
    hover_tool.tooltips = [("y", "$y{int}")]
    return p

show(plot_complaints_stacked_by_city(df_plot))


# **Exercise 5** (2 points). Suppose we want to create a different stacked bar plot that shows, for each complaint type $t$ and city $c$, the fraction of all complaints of type $t$ (across all cities, not just the top ones) that occurred in city $c$. Store your result in a dataframe named `df_plot_fraction`. It should have the same columns as `df_plot`, **except** that the `complaint_count` column should be replaced by one named `complaint_frac`, which holds the fractional values.
# 
# > **Hint.** Everything you need is already in `df_plot`.
# >
# > **Note.** The test cell will create the chart in addition to checking your result. Note that the normalized bars will not necessarily add up to 1; why not?

# In[249]:


###
df_plot_fraction = df_plot.copy()
df_plot_fraction['complaint_frac'] = df_plot['complaint_count']/df_plot['freq']
del df_plot_fraction['complaint_count']
###

df_plot_fraction.head()


# In[250]:


# Test cell: `norm_above_test`

df_plot_stacked_fraction = cast(df_plot_fraction, key='city_name', value='complaint_frac')


if False:
    df_plot_stacked_fraction.to_csv(get_path('df_plot_stacked_fraction_soln.csv'), index=False)

show(plot_complaints_stacked_by_city(df_plot_fraction, y='complaint_frac'))

def all_tol(x, tol=1e-14):
    return all([abs(i) <= tol for i in x])

df_plot_fraction_soln = canonicalize_tibble(pd.read_csv(get_path('df_plot_stacked_fraction_soln.csv')))
df_plot_fraction_yours = canonicalize_tibble(df_plot_stacked_fraction)

nonfloat_cols = df_plot_stacked_fraction.columns.difference(TOP_CITIES)
assert tibbles_are_equivalent(df_plot_fraction_yours[nonfloat_cols],
                              df_plot_fraction_soln[nonfloat_cols])
for c in TOP_CITIES:
    assert all(abs(df_plot_fraction_yours[c] - df_plot_fraction_soln[c]) <= 1e-13),            "Fractions for city {} do not match the values we are expecting.".format(c)

print("\n(Passed!)")


# ## Dates and times in SQL
# 
# Recall that the input data had a column with timestamps corresponding to when someone submitted a complaint. Let's quickly summarize some of the features in SQL and Python for reasoning about these timestamps.

# The `CreatedDate` column is actually a specially formatted date and time stamp, where you can query against by comparing to strings of the form, `YYYY-MM-DD hh:mm:ss`.
# 
# For example, let's look for all complaints on September 15, 2015.

# In[251]:


query = '''
  SELECT LOWER(ComplaintType), CreatedDate, UPPER(City)
    from data
    where CreatedDate >= "2015-09-15 00:00:00.0"
      and CreatedDate < "2015-09-16 00:00:00.0"
    order by CreatedDate
'''
df = pd.read_sql_query (query, disk_engine)
df


# This next example shows how to extract just the hour from the time stamp, using SQL's `strftime()`.

# In[252]:


query = '''
  SELECT CreatedDate, STRFTIME('%H', CreatedDate) AS Hour, LOWER(ComplaintType)
    FROM data
    LIMIT 5
'''
df = pd.read_sql_query (query, disk_engine)
df


# **Exercise 6** (3 points). Construct a tibble called `df_complaints_by_hour`, which contains the total number of complaints during a given hour of the day. That is, the variables or column names should be {`hour`, `count`} where each observation is the total number of complaints (`count`) that occurred during a given `hour`.
# 
# > Interpret `hour` as follows: when `hour` is `02`, that corresponds to the open time interval [`02:00:00`, `03:00:00.0`).

# In[253]:


# Your task: Construct `df_complaints_by_hour` as directed.
###
query = '''
  SELECT STRFTIME('%H', CreatedDate) AS hour, COUNT(ComplaintType) AS count
    FROM data
    group by hour
'''
df_complaints_by_hour = pd.read_sql_query (query, disk_engine)
# Displays your answer:
display(df_complaints_by_hour)


# In[254]:


# Test cell: `df_complaints_by_hour_test`
    
print ("Reading instructor's solution...")
if False:
    df_complaints_by_hour_soln.to_csv(get_path('df_complaints_by_hour_soln.csv'), index=False)
df_complaints_by_hour_soln = pd.read_csv (get_path('df_complaints_by_hour_soln.csv'))
display (df_complaints_by_hour_soln)

df_complaints_by_hour_norm = df_complaints_by_hour.copy ()
df_complaints_by_hour_norm['hour'] =     df_complaints_by_hour_norm['hour'].apply (int)
assert tibbles_are_equivalent (df_complaints_by_hour_norm,
                               df_complaints_by_hour_soln)
print ("\n(Passed.)")


# Let's take a quick look at the hour-by-hour breakdown above.

# In[255]:


p = make_barchart(df_complaints_by_hour, 'hour', 'count',
                  {'title': 'Complaints by hour',
                   'plot_width': 800, 'plot_height': 320})
show(p)


# An unusual aspect of these data are the excessively large number of reports associated with hour 0 (midnight up to but excluding 1 am), which would probably strike you as suspicious. Indeed, the reason is that there are some complaints that are dated but with no associated time, which was recorded in the data as exactly `00:00:00.000`.

# In[256]:


query = '''
  SELECT COUNT(*)
    FROM data
    WHERE STRFTIME('%H:%M:%f', CreatedDate) = '00:00:00.000'
'''

pd.read_sql_query(query, disk_engine)


# **Exercise 7** (2 points). What is the most common hour for noise complaints? Compute a tibble called `df_noisy_by_hour` whose variables are {`hour`, `count`} and whose observations are the number of noise complaints that occurred during a given `hour`. Consider a "noise complaint" to be any complaint string containing the word `noise`. Be sure to filter out any dates _without_ an associated time, i.e., a timestamp of `00:00:00.000`.

# In[267]:


###
query = '''
  SELECT STRFTIME('%H', CreatedDate) AS hour, COUNT(*) AS count
  FROM data
  WHERE (strftime('%H:%M:%f', CreatedDate) != '00:00:00.000')
  AND (LOWER(ComplaintType) LIKE '%noise%')
  GROUP BY hour  
'''

df_noisy_by_hour = pd.read_sql_query (query, disk_engine)
display(df_noisy_by_hour)


# In[268]:


# Test cell: `df_noisy_by_hour_test`

print ("Reading instructor's solution...")
if False:
    df_noisy_by_hour.to_csv(get_path('df_noisy_by_hour_soln.csv'), index=False)
df_noisy_by_hour_soln = pd.read_csv (get_path('df_noisy_by_hour_soln.csv'))
display(df_noisy_by_hour_soln)

df_noisy_by_hour_norm = df_noisy_by_hour.copy()
df_noisy_by_hour_norm['hour'] =     df_noisy_by_hour_norm['hour'].apply(int)
assert tibbles_are_equivalent (df_noisy_by_hour_norm,
                               df_noisy_by_hour_soln)
print ("\n(Passed.)")


# In[269]:


p = make_barchart(df_noisy_by_hour, 'hour', 'count',
                  {'title': 'Noise complaints by hour',
                   'plot_width': 800, 'plot_height': 320})
show(p)


# **Exercise 8** (ungraded). Create a line chart to show the fraction of complaints (y-axis) associated with each hour of the day (x-axis), with each complaint type shown as a differently colored line. Show just the top 5 complaints (`top_complaints[:5]`). Remember to exclude complaints with a zero-timestamp (i.e., `00:00:00.000`).
# 
# > **Note.** This exercise is ungraded but if your time permits, please give it a try! Feel free to discuss your approaches to this problem on the discussion forums (but do try to do it yourself first). One library you may find useful to try out is holoviews (http://holoviews.org/index.html)

# In[280]:


top_5_complaints = strs_to_args(top_complaints[:5]['type'].tolist())

query = '''SELECT ComplaintType, STRFTIME('%H', CreatedDate) AS hour, COUNT(*) AS count
  FROM data
  WHERE (strftime('%H:%M:%f', CreatedDate) != '00:00:00.000')
  AND (LOWER(ComplaintType) LIKE '%noise%')
  GROUP BY ComplaintType, hour 
            '''.format(top_5_complaints)
df_top5_complaints_by_hour = pd.read_sql_query(query, disk_engine)
df_top5_complaints_by_hour.head(10)


# In[281]:


df_pivot = pd.pivot_table(df_top5_complaints_by_hour, values='count', columns='ComplaintType', index='hour')
df_pivot.head(10)


# In[282]:


import matplotlib.pyplot as plt
df_pivot.plot(figsize=(13, 8))
plt.xlabel('Hour of Day')
plt.ylabel('Number of Complaints')
plt.show()


# In[272]:


import holoviews as hv
hv.extension('bokeh')
from holoviews import Bars

###
# YOUR CODE HERE


###


# ### Learn more
# 
# - Find more open data sets on [Data.gov](https://data.gov) and [NYC Open Data](https://nycopendata.socrata.com)
# - Learn how to setup [MySql with Pandas and Plotly](http://moderndata.plot.ly/graph-data-from-mysql-database-in-python/)
# - Big data workflows with [HDF5 and Pandas](http://stackoverflow.com/questions/14262433/large-data-work-flows-using-pandas)

# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




