"""

 categorical_matrix.py (author: Anson Wong / git: ankonzoid)

 Functions for setting up general categorical matrices.

 In main():
  - reads Movielens 1M data (movies.dat) into data matrix X, then converts to expanded categorical matrix Xcat
  - afterwards, split Xcat into 2 pieces: 
 
  1) itemid_list: a list of itemids 
  2) itemid_vector_list: list of itemid categorical vector representations
  
"""
import numpy as np
import pandas as pd
import os, sys

def main():

    # ========================
    # Create set data parameters (i.e. set data filename, column names, and toggle for categorical features)
    # ========================
    data_param = \
    {
        'data_filename': "../input/movies.dat",  # data filename
        'Xheader': ['movie_id', 'movie_name', 'movie_genre'],  # X column header names
        'Xuse':    [1, 1, 1],  # toggle which features to output
        'Xiscat':  [0, 0, 1],  # toggle for features to treat as categorical features
        'data_delimiter': '::',  # delimiter separating columns of data file
        'split_delimiter': '|'  # delimiter within the columns of data file
    }

    # ========================
    # Create categorical matrix
    # ========================
    X, Xcat, Xheader, Xuse, Xiscat, XNcat, Xcatnames, Xcatind = find_X_Xcat(data_param)

    if 0:
        print(X)
        print(Xcat)
        print(Xuse)
        print(Xiscat)
        print(XNcat)
        print(Xcatnames)
        print(Xcatind)

    # ========================
    # Split Xcat -> (i) itemid list + (ii) item categorical vector representation
    # ========================
    itemid_index = find_obj_index("movie_id", Xheader)  # itemid column index
    cat_index = find_obj_index("movie_genre", Xheader)  # interested categorical column index
    if itemid_index < 0 or cat_index < 0:
        raise IOError("Error! Could not find itemid_index or cat_index!")

    itemid_list, itemid_vector_list = split_Xcat(Xcat, Xcatind, itemid_index, cat_index)

    # Print
    if 1:
        # Print: categorical name objects
        for row in Xcat:
            print(row)
        print(XNcat)
        print(Xcatnames)
        print(Xcatind)
        # Print: itemid_list and itemid_vector_list
        for i, itemid in enumerate(itemid_list):
            print("itemid", itemid_list[i], ":", itemid_vector_list[i])

# find_X_Xcat:
def find_X_Xcat(data_param):

    # Extract data parameters
    data_filename = data_param['data_filename']
    Xheader = data_param['Xheader']
    Xuse = data_param['Xuse']
    Xiscat = data_param['Xiscat']
    data_delimiter = data_param['data_delimiter']
    split_delimiter = data_param['split_delimiter']

    # Read datafile into matrix X (Xheader = headers, Xiscat = is categorical column)
    X, Xheader, Xuse, Xiscat = read_X(data_filename, Xheader, Xuse, Xiscat, data_delimiter)

    # Find category names for each categorical column
    XNcat, Xcatnames, Xcatind = find_categories(X, Xiscat, split_delimiter)

    # Contruct expanded categorical matrix Xcat encoding all requested categorical data
    Xcat = construct_Xcat(X, Xiscat, Xcatnames, split_delimiter)

    return (X, Xcat, Xheader, Xuse, Xiscat, XNcat, Xcatnames, Xcatind)

# split_Xcat: splits Xcat into list of itemids, and a list of itemid categorical vector representation list
def split_Xcat(Xcat, Xcatind, itemid_index, cat_index):
    itemid_list = []
    itemid_vector_list = []
    for i, row in enumerate(Xcat):
        itemid = Xcat[i][itemid_index]
        itemid_vector = Xcat[i][Xcatind[cat_index][0]:Xcatind[cat_index][1] + 1]
        itemid_list.append(itemid)  # append
        itemid_vector_list.append(itemid_vector)  # append
    return (itemid_list, itemid_vector_list)

# construct_Xcat: construct full categorical matrix given we know our categories
def construct_Xcat(X, Xiscat, Xcatnames, split_delimiter):
    Xcat = []
    for i, row in enumerate(X):
        vector = []
        for j, Xij in enumerate(row):
            if Xiscat[j] > 0:
                vector_category = find_vector_category(Xij, Xcatnames[j], split_delimiter)
                vector = vector + vector_category
            else:
                vector.append(Xij)
        Xcat.append(vector)
    return Xcat

# find_categories: from X we find all the categories for the columns specified by Xiscat
def find_categories(X, Xiscat, split_delimiter):
    [N,d] = X.shape
    XNcat = [0] * d  # keeps track of number categories for each item feature
    Xcatnames = [[]] * d  # list of categories names for each item feature
    Xcatind = [[0, 0]] * d  # final categorical column indices for each item feature
    catind_current = 0  # current catind (internal calculation dummy index tracker)
    for j, iscat in enumerate(Xiscat):
        # If this column is a categorical feature -> we encode it
        if iscat > 0:
            catnames = []
            for row, x in enumerate(X[:, j]):
                x_split = x.split(split_delimiter)
                # For each names in x_split, append to category names list if it is not there
                for x_name in x_split:
                    if is_element_of(x_name, catnames) == False:
                        catnames.append(x_name)
                        XNcat[j] = len(catnames)
            Xcatnames[j] = catnames
        # If this column is not a categorical feature
        else:
            XNcat[j] = 1
            Xcatnames[j] = ['NONE']

        # Increment catind_current by 1 for next item feature
        catind_start = catind_current
        catind_end = catind_current + XNcat[j] - 1
        Xcatind[j] = [catind_start, catind_end]
        catind_current = catind_end + 1
    return (XNcat, Xcatnames, Xcatind)

# find_vector_category: given string x, find the vector representation based of of categorynames basis
def find_vector_category(x, categorynames, split_delimiter):
    x_split = x.split(split_delimiter)
    vector_category = [0] * len(categorynames)
    for name in x_split:
        obj_index = find_obj_index(name, categorynames)
        if obj_index >= 0:
            vector_category[obj_index] += 1
    return vector_category

# find_obj_position: finds index of list where obj exists, if it doesn't then return -1
def find_obj_index(obj, list):
    index = -1
    for i, obj_list in enumerate(list):
        if obj == obj_list:
            index = i
            break
    return index

# is_element_of: checks if obj is in list
def is_element_of(obj, list):
    result = False
    for obj_list in list:
        if obj == obj_list:
            result = True
            break
    return result

# read_X: from the data filename
def read_X(data_filename, Xheader, Xuse, Xiscat, data_delimiter):
    # Read csv file and store data into (X, Xheader) if data_filename exists
    if os.path.isfile(data_filename) == True:
        X = np.array(pd.read_csv(data_filename, dtype=object, skiprows=0, names=Xheader, sep=data_delimiter, engine='python'))
    else:
        raise IOError("Error! No data file: %s" % data_filename)

    # Use columns based on Xuse
    columns_use = find_indices(Xuse, lambda x: x > 0)
    X = X[:, columns_use]
    Xheader = [Xheader[index] for index in columns_use]
    Xuse = [Xuse[index] for index in columns_use]
    Xiscat = [Xiscat[index] for index in columns_use]

    # Make checks before returning X
    [N, d] = X.shape
    if len(Xheader) != len(Xiscat):
        raise IOError("Error! Xheader and Xiscat lengths are not the same!")
    elif len(Xheader) != d:
        raise IOError("Error! Xheader length is not equal to d!")

    return (X, Xheader, Xuse, Xiscat)

# find_indices: find the indices of list 'a' that satisfy the conditions of func
def find_indices(a, func):
    return [i for (i, val) in enumerate(a) if func(val)]

# find_catind_from_Xiscat: based on the start/end indices, we find the categorical indices
def find_catind_from_Xiscat(Xcat, Xiscat, Xcatind):
    dcat = len(Xcat[0])
    toggle_cat_vector = [0] * dcat
    feature_column_use = find_indices(Xiscat, lambda x: x > 0)
    for feature in feature_column_use:
        i_start = Xcatind[feature][0]
        i_end = Xcatind[feature][1]
        for k in range(i_start, i_end + 1):
            toggle_cat_vector[k] = 1
    ind_feature_column_use = find_indices(toggle_cat_vector, lambda x: x > 0)
    return ind_feature_column_use

# construct_Xcat_relevant: with Xcat, and using Xcatind and Xiscat, we only take Xcat are fully filled with categorical values
def construct_Xcat_relevant(Xcat, Xiscat, Xcatind):
    Xcat_relevant = []
    ind_Xcat_relevant = find_catind_from_Xiscat(Xcat, Xiscat, Xcatind)
    for row in Xcat:
        new_row = []
        for i in ind_Xcat_relevant:
            new_row.append(row[i])
        Xcat_relevant.append(new_row)
    return Xcat_relevant

#
# Driver file
#
if __name__ == '__main__':
    main()