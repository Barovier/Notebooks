
# coding: utf-8

# # Part 1 of 2: Processing an HTML file
# 
# 
# **Exercise ordering:** Each exercise builds logically on previous exercises, but you may solve them in any order. That is, if you can't solve an exercise, you can still move on and try the next one. Use this to your advantage, as the exercises are **not** necessarily ordered in terms of difficulty. Higher point values generally indicate more difficult exercises. 
# 
# **Demo cells:** Code cells starting with the comment `### define demo inputs` load results from prior exercises applied to the entire data set and use those to build demo inputs. These must be run for subsequent demos to work properly, but they do not affect the test cells. The data loaded in these cells may be rather large (at least in terms of human readability). You are free to print or otherwise use Python to explore them, but we did not print them in the starter code.
# 
# **Debugging your code:** Right before each exercise test cell, there is a block of text explaining the variables available to you for debugging. You may use these to test your code and can print/display them as needed (careful when printing large objects, you may want to print the head or chunks of rows at a time).
# 
# **Exercise point breakdown:**
# 
# - Exercise 0: 5 points
# 
# **Final reminders:** 
# 
# - Submit after **every exercise**
# - Review the generated grade report after you submit to see what errors were returned
# - Stay calm, skip problems as needed, and take short breaks at your leisure
# 

# ## Topic Introduction
# 
# One of the richest sources of information is [the Web](http://www.computerhistory.org/revolution/networking/19/314)! In this notebook, we ask you to use string processing and regular expressions to mine a web page, which is stored in HTML format.
# 
# > **Note 0.** The exercises below involve processing of HTML files. However, you don't need to know anything specific about HTML; you can solve (and we have solved) all of these exercises assuming only that the data is a semi-structured string, amenable to simple string manipulation and regular expression processing techniques. In Notebook 6 (optional), you'll see a different method that employs the [Beautiful Soup module](https://www.crummy.com/software/BeautifulSoup/bs4/doc/).
# >
# > **Note 1.** Following Note 0, there are some outspoken people who believe you should never use regular expressions on HTML. Your instructor finds these arguments to be overly pedantic. For an entertaining take on the subject, see [this blog post](https://blog.codinghorror.com/parsing-html-the-cthulhu-way/).
# >
# > **Note 2.** The data below is a snapshot from an older version of the Yelp! site. Therefore, you should complete the exercises using the data we've provided, rather than downloading a copy directly from Yelp!.

# **The data: Yelp! reviews.** The data you will work with is a snapshot of a recent search on the [Yelp! site](https://yelp.com) for the best fried chicken restaurants in Atlanta. That snapshot is hosted here: https://cse6040.gatech.edu/datasets/yelp-example
# 
# If you go ahead and open that site, you'll see that it contains a ranked list of places:
# 
# ![Top 10 Fried Chicken Spots in ATL as of September 12, 2017](https://cse6040.gatech.edu/datasets/yelp-example/ranked-list-snapshot.png)

# **Your task.** In this part of this assignment, we'd like you to write some code to extract this list.

# ## Getting the data
# 
# First things first: you need an HTML file. The following Python code opens a copy of the sample Yelp! page from above.

# In[1]:


### Global Imports

# Use this cell to import anything common, e.g. numpy, pandas, sqlite3
# Use this cell to bring in the starter data
import hashlib

with open('resource/asnlib/publicdata/yelp.htm', 'r', encoding='utf-8') as f:
    yelp_html = f.read().encode(encoding='utf-8')
    checksum = hashlib.md5(yelp_html).hexdigest()
    assert checksum == "4a74a0ee9cefee773e76a22a52d45a8e", "Downloaded file has incorrect checksum!"
    
print("'yelp.htm' is ready!")


# **Viewing the raw HTML in your web browser.** The file you just downloaded is the raw HTML version of the data described previously. Before moving on, you should go back to that site and use your web browser to view the HTML source for the web page. Do that now to get an idea of what is in that file.
# 
# > If you don't know how to view the page source in your browser, try the instructions on [this site](http://www.wikihow.com/View-Source-Code).

# **Reading the HTML file into a Python string.** Let's also open the file in Python and read its contents into a string named, `yelp_html`.

# In[2]:


with open('resource/asnlib/publicdata/yelp.htm', 'r', encoding='utf-8') as yelp_file:
    yelp_html = yelp_file.read()
    
# Print first few hundred characters of this string:
print("*** type(yelp_html) == {} ***".format(type(yelp_html)))
n = 1000
print("*** Contents (first {} characters) ***\n{} ...".format(n, yelp_html[:n]))


# Oy, what a mess! It will be great to have some code read and process the information contained within this file.

# ## Exercise (5 points): Extracting the ranking
# 
# Create a new function that will return a variable named `rankings`, which is a list of dictionaries set up as follows:
# 
# * `rankings[i]` is a dictionary corresponding to the restaurant whose rank is `i+1`. For example, from the screenshot above, `rankings[0]` should be a dictionary with information about Gus's World Famous Fried Chicken.
# * Each dictionary, `rankings[i]`, should have these keys:
#     * `rankings[i]['name']`: The name of the restaurant, a string.
#     * `rankings[i]['stars']`: The star rating, as a string, e.g., `'4.5'`, `'4.0'`
#     * `rankings[i]['numrevs']`: The number of reviews, as an **integer.**
#     * `rankings[i]['price']`: The price range, as dollar signs, e.g., `'$'`, `'$$'`, `'$$$'`, or `'$$$$'`.
#     
# Of course, since the current topic is regular expressions, you might try to apply them (possibly combined with other string manipulation methods) find the particular patterns that yield the desired information.

# In[3]:


### Define demo inputs
demo_str_ex0 = yelp_html


# <!-- Expected demo output text block -->
# The demo included in the solution cell below should display the following output:
# ```
# extract_ranking(demo_str_ex0) ->
# 
# [{'name': 'Gus’s World Famous Fried Chicken',
#   'stars': '4.0',
#   'numrevs': 549,
#   'price': '$$'},
#  {'name': 'South City Kitchen - Midtown',
#   'stars': '4.5',
#   'numrevs': 1777,
#   'price': '$$'},
#  {'name': 'Mary Mac’s Tea Room',
#   'stars': '4.0',
#   'numrevs': 2241,
#   'price': '$$'},
#  {'name': 'Busy Bee Cafe', 'stars': '4.0', 'numrevs': 481, 'price': '$$'},
#  {'name': 'Richards’ Southern Fried',
#   'stars': '4.0',
#   'numrevs': 108,
#   'price': '$$'},
#  {'name': 'Greens &amp; Gravy', 'stars': '3.5', 'numrevs': 93, 'price': '$$'},
#  {'name': 'Colonnade Restaurant',
#   'stars': '4.0',
#   'numrevs': 350,
#   'price': '$$'},
#  {'name': 'South City Kitchen Buckhead',
#   'stars': '4.5',
#   'numrevs': 248,
#   'price': '$$'},
#  {'name': 'Poor Calvin’s', 'stars': '4.5', 'numrevs': 1558, 'price': '$$'},
#  {'name': 'Rock’s Chicken &amp; Fries',
#   'stars': '4.0',
#   'numrevs': 67,
#   'price': '$'}]
# ```
# <!-- Include any shout outs here -->

# In[6]:


from bs4 import BeautifulSoup
import re

def extract_ranking(yelp_html):
    # Parse HTML using BeautifulSoup
    soup = BeautifulSoup(yelp_html, 'html.parser')
    results = soup.find_all('li', class_='regular-search-result')

    # Extract raw names and reviews
    raw_names = [result.get_text()[25:57] for result in results]
    raw_reviews = [result.get_text()[57:100] for result in results]

    # Process and clean restaurant names
    cleaned_names = []
    name_pattern = re.compile('.*')
    
    for raw_name in raw_names:
        match = name_pattern.search(raw_name)
        cleaned_names.append(match.group().strip())

    cleaned_names = [name.replace('&', '&amp;') for name in cleaned_names]

    # Process reviews and costs
    reviews = []
    costs = []
    review_pattern = re.compile(r'[\d]+')
    cost_pattern = re.compile(r'[\$]+')

    for raw_review in raw_reviews:
        review_match = review_pattern.search(raw_review)
        reviews.append(int(review_match.group()))
        cost_match = cost_pattern.search(raw_review)
        costs.append(cost_match.group())

    # Process star ratings
    raw_ratings = [image.get('alt', '') for result in results for image in result.find_all('img')]
    ratings = [re.search(r'[\d\.]+', item).group() for item in raw_ratings if item and item[0].isnumeric()]

    # Create a list of dictionaries for rankings
    rankings = [{'name': a, 'stars': b, 'numrevs': c, 'price': d} for a, b, c, d in zip(cleaned_names, ratings, reviews, costs)]
    
    return rankings

# Test the function with the provided HTML
yelp_html = "..."  # Replace with the actual HTML content
result = extract_ranking(yelp_html)
print(result)

    
### demo function call
extract_ranking(demo_str_ex0)

#reference: chatgpt 


# <!-- Test Cell Boilerplate -->
# The cell below will test your solution for Exercise 0. The testing variables will be available for debugging under the following names in a dictionary format.
# - `input_vars` - Input variables for your solution. 
# - `original_input_vars` - Copy of input variables from prior to running your solution. These _should_ be the same as `input_vars` - otherwise the inputs were modified by your solution.
# - `returned_output_vars` - Outputs returned by your solution.
# - `true_output_vars` - The expected output. This _should_ "match" `returned_output_vars` based on the question requirements - otherwise, your solution is not returning the correct output. 

# In[7]:


### test_cell_ex0
from tester_fw.testers import Tester

conf = {
    'case_file':'tc_0', 
    'func': extract_ranking, # replace this with the function defined above
    'inputs':{ # input config dict. keys are parameter names
        'yelp_html':{
            'dtype':'str', # data type of param.
            'check_modified':True,
        }
    },
    'outputs':{
        'output_0':{
            'index':0,
            'dtype':'list',
            'check_dtype': True,
            'check_col_dtypes': True, # Ignored if dtype is not df
            'check_col_order': True, # Ignored if dtype is not df
            'check_row_order': True, # Ignored if dtype is not df
            'check_column_type': True, # Ignored if dtype is not df
            'float_tolerance': 10 ** (-6)
        }
    }
}
tester = Tester(conf, key=b'bcdBk1WLcJ7DmbnNvGz65Oub0ZpaNa1bnbT-5L_G0Yk=', path='resource/asnlib/publicdata/')
for _ in range(10):
    try:
        tester.run_test()
        (input_vars, original_input_vars, returned_output_vars, true_output_vars) = tester.get_test_vars()
    except:
        (input_vars, original_input_vars, returned_output_vars, true_output_vars) = tester.get_test_vars()
        raise

print('Passed! Please submit.')


# In[8]:


# Print the returned output and true output for the problematic test case
print("Returned Output:", returned_output_vars['output_0'])
print("True Output:", true_output_vars['output_0'])


# **Fin!** This cell marks the end of Part 1. Don't forget to save, restart and rerun all cells, and submit it. When you are done, proceed to Part 2.
