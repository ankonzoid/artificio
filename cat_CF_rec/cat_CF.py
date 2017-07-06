"""

 cat_CF.py (author: Anson Wong / git: ankonzoid)

 Implements a general categorical collaborative filtering scheme based on categorical data of your choice.

 In this example we use MovieLens data.

"""
import numpy as np
import sys

from functions import categorical_matrix as CATMAT  # import categorical matrix
from functions import similarity_table as SIMTAB  # import similarity table

def main():

    # Set data filename, column names, and toggle for categorical features
    data_param_list = \
    [
        # users.dat (uu-interaction)
        {
            'interaction_type': 'uu',  # choose interaction type (uu, ii, ui)
            'data_filename': "query/users.dat",  # data filename
            'Xheader': ['uu_userid', 'uu_gender', 'uu_age', 'uu_occupation', 'uu_zipcode'],  # X column header names
            'Xuse':    [1, 1, 1, 1, 0],  # toggle which features to output
            'Xiscat':  [0, 1, 1, 1, 0],  # toggle for features to treat as categorical features
            'data_delimiter': '::',  # delimiter separating columns of data file
            'split_delimiter': '|'  # delimiter within the columns of data file
        },

        # movies.dat (ii-interaction)
        {
            'interaction_type': 'ii',  # choose interaction type (uu, ii, ui)
            'data_filename': "query/movies.dat",  # data filename
            'Xheader': ['ii_itemid', 'ii_name', 'ii_genre'],  # X column header names
            'Xuse':    [1, 1, 1],  # toggle which features to output
            'Xiscat':  [0, 0, 1],  # toggle for features to treat as categorical features
            'data_delimiter': '::',  # delimiter separating columns of data file
            'split_delimiter': '|'  # delimiter within the columns of data file
        },

        # ratings.dat (ui-interaction)
        {
            'interaction_type': 'ui',  # choose interaction type (uu, ii, ui)
            'data_filename': "query/ratings.dat",  # data filename
            'Xheader': ['ui_userid', 'ui_itemid', 'ui_rating', 'ui_time'],  # X column header names
            'Xuse':    [1, 1, 0, 0],  # toggle which features to output
            'Xiscat':  [0, 0, 0, 0],  # toggle for features to treat as categorical features
            'data_delimiter': '::',  # delimiter separating columns of data file
            'split_delimiter': '|'  # delimiter within the columns of data file
        }
    ]

    # ================================================
    #
    # Create categorical matrix Xcat
    # Uses above data parameters.
    #
    # ================================================
    print("Creating categorical matrix...")
    X, Xcat, Xheader, Xuse, Xiscat, XNcat, Xcatnames, Xcatind = CATMAT.find_X_Xcat(data_param_list[0])
    Xcat_relevant = CATMAT.construct_Xcat_relevant(Xcat, Xiscat, Xcatind)  # Xcat has non-categorical features, this eliminates them

    if 0:
        for row in Xcat_relevant:
            print(row)

    # ================================================
    #
    # Find similarities (between items and between users)
    #
    # ================================================
    print("Building similarity table...")
    U = np.array(Xcat_relevant)
    [N, d] = U.shape

    sim_uu = SIMTAB.build_uu(U)

    print("Printing similarity table")
    print(sim_uu)
    print(sim_uu.shape)

    # ================================================
    #
    # Find scores based on similarities
    #
    # ================================================
    print("Training... building ui score table SCORE_ui= [%d,%d] with top k_agg= %d similar users...")
    k_users_aggregate = 30
    score_ui = np.zeros((N, d), dtype=float)
    score_norm_ui = np.zeros((N, d), dtype=float)
    for u in range(0, N):
        # Find top k similar users: take min(k_use,N_users) elements, sort ascend, flip to descend, find indices
        ind_topkscore = get_topk(sim_uu[u, :], k_users_aggregate)  # get top k_users score indices

        # Aggregate the top k_use similar users to give an item list with scores (higher being better recommendation)
        score_ui[u, :] = np.sum(U[ind_topkscore, :], axis=0)
        score_norm_ui[u, :] = score_ui[u, :] / np.sum(score_ui[u, :])

    if 0:
        print(score_norm_ui)
        print(score_ui)

        for row in score_ui:
            print(row)

    """==================================================
         Prediction: Make recommendations for training users
    =================================================="""
    k_rec = 3
    itemindex_train_rec = np.ones((N, k_rec), dtype=int)
    itemid_train_rec = np.ones((N, k_rec), dtype=int)
    for utrain in range(0, N):

        # Make recommendations
        ind_rec = get_topk(score_ui[utrain, :], k_rec)  # top score indices
        itemindex_train_rec[utrain, :] = ind_rec
        itemid_train_rec[utrain, :] = ind_rec + 1

        if (utrain < 3):
            print('')
            print("Item recommendations [training user %d]:" % utrain)
            print(itemindex_train_rec[utrain, :])
            print("Corresponding item scores [training user %d]:" % utrain)
            print(score_ui[utrain, itemindex_train_rec[utrain, :]])
            print('')



    """
    for i, data_param_i in enumerate(data_param_list):
        X, Xcat, Xheader, Xuse, Xiscat, XNcat, Xcatnames, Xcatind = CATMAT.find_X_Xcat(data_param_i)
        if 1:
            print("")
            print(X)
            print(Xcat)
            print(Xuse)
            print(Xiscat)
            print(XNcat)
            print(Xcatnames)
            print(Xcatind)
            print("")
    """



    # ================================================
    #
    # User-categorical matrices (user descriptions).
    # These are user vs user-features matrices (item purchases, age, gender, location,...)
    #
    # ================================================

    # ================================================
    #
    # Item-categorical matrices (item descriptions).
    # These are item vs item-features matrices (item purchases, age, gender, location,...)
    #
    # ================================================


    # ========================
    # Split Xcat -> (i) itemid list + (ii) item categorical vector representation
    # ========================
    '''
    itemid_index = CATMAT.find_obj_index("ii_itemid", Xheader)  # itemid column index
    cat_index = CATMAT.find_obj_index("ii_genre", Xheader)  # interested categorical column index
    if itemid_index < 0 or cat_index < 0:
        raise IOError("Error! Could not find itemid_index or cat_index!")

    itemid_list, itemid_vector_list = CATMAT.split_Xcat(Xcat, Xcatind, itemid_index, cat_index)

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
    '''


# ================================================================
# 
# Side functions
#
# ================================================================

# Gets top-k (highest) score indices of a vector ()
def get_topk(vec,k):
    k_use = min(k, len(vec))
    ind = np.argpartition(vec, -k_use)[-k_use:]
    ind = ind[np.flipud(np.argsort(vec[ind]))]
    return ind


#
# Driver file
#
if __name__ == '__main__':
    main()