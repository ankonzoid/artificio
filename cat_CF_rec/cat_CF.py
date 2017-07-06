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
    
    # ===========================================
    # Run settings
    # ============================================
    k_rec = 3  # number of recommendations to make for each user
    k_users_aggregate = 30  # hyperparameter for computing scores

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
    print("Reading data and creating categorical matrix...")
    X, Xcat, Xheader, Xuse, Xiscat, XNcat, Xcatnames, Xcatind = CATMAT.find_X_Xcat(data_param_list[0])
    Xcat_relevant = CATMAT.construct_Xcat_relevant(Xcat, Xiscat, Xcatind)  # Xcat has non-categorical features, this eliminates them

    if 0:
        for row in Xcat_relevant:
            print(row)

    # ================================================
    #
    # From utility matrix, build uu-similarity matrix
    #
    # ================================================
    U = np.array(Xcat_relevant)
    [N, d] = U.shape
    sim_uu = SIMTAB.build_uu(U)

    print("Building uu-similarity table = {0}...".format(sim_uu.shape))

    # ================================================
    #
    # Compute item scores using uu-similarity table
    #
    # ================================================
    score_ui = np.zeros((N, d), dtype=float)
    score_norm_ui = np.zeros((N, d), dtype=float)
    for u in range(0, N):

        # Find top k similar users: take min(k_use,N_users) elements, sort ascend, flip to descend, find indices
        ind_topkscore = get_topk(sim_uu[u, :], k_users_aggregate)  # get top k_users score indices

        # Aggregate the top k_use similar users to give an item list with scores (higher being better recommendation)
        score_ui[u, :] = np.sum(U[ind_topkscore, :], axis=0)
        score_norm_ui[u, :] = score_ui[u, :] / np.sum(score_ui[u, :])


    print("Building ui-score table = {0} with top k_agg = {1} similar users...".format(score_ui.shape, k_users_aggregate))
    if 0:
        print(score_norm_ui)
        print(score_ui)

        for row in score_ui:
            print(row)

    """==================================================
         Prediction: Make recommendations for training users
    =================================================="""
    print("Making k_rec = {0} recommendations for each training user...".format(k_rec))
    itemindex_train_rec = np.ones((N, k_rec), dtype=int)
    itemid_train_rec = np.ones((N, k_rec), dtype=int)
    for uid_train in range(0, N):

        # Make recommendations
        ind_rec = get_topk(score_ui[uid_train, :], k_rec)  # top score indices
        itemindex_train_rec[uid_train, :] = ind_rec
        itemid_train_rec[uid_train, :] = ind_rec + 1

        print("Itemid rec [userid {0}]: {1}".format(uid_train, itemindex_train_rec[uid_train, :]))
        #print("Itemid rec scores [user {0}]: {1}".format(uid_train, score_ui[uid_train, itemindex_train_rec[uid_train, :]]))
        #print("")



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
