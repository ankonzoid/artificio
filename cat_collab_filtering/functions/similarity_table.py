"""

 similarity_table.py (author: Anson Wong / git: ankonzoid))

 Functions for building similarity tables.

"""
import numpy as np
from numpy import linalg as LA  # for norm function
import sys

"""=======================================
Build user-user similarity table (created on May 15, 2017)
======================================="""
def build_uu(U):
    Unorm = U / U.sum(axis=1)[:, np.newaxis]  # normalize rows of U into a new matrix Unorm
    SIM = np.matmul(Unorm, np.transpose(U))  # this is the cosine-similarity between users of U i.e. U*U^T
    # Check that transpose(SIM) == SIM
    if ~(SIM.transpose() == SIM).all():
        raise IOError("SIM_uu similarity matrix is not symmetric!")
    return SIM

"""=======================================
Find similarity of test user with provided U
======================================="""
def build_SIM_vec(utest_vec, Utrain):
    [N_users, N_items] = Utrain.shape
    SIM_vec = np.zeros(N_users)  # user-user similarity matrix

    # Compute test user utest similarity vector with U
    for u in range(0, N_users):  # Go through each user (u)

        u_vec = Utrain[u, :]  # u

        if (not np.any(u_vec)) or (not np.any(utest_vec)):  # u_vec=0 or up_vec=0 -> SIM(u,u)=0 (will be filtered out)
            SIM_vec[u] = 0
        else:
            u_vec_norm = LA.norm(u_vec, 2)  # this norm (not norm-squared)
            utest_vec_norm = LA.norm(utest_vec, 2)  # this norm (not norm-squared)
            SIM_vec[u] = np.dot(utest_vec, u_vec) / (u_vec_norm * utest_vec_norm)  # cosine similarity

    return SIM_vec


"""=======================================
Build item-item similarity table
======================================="""
def build_ii(U):
    [N_users, N_items] = U.shape
    SIM = np.zeros((N_items, N_items))  # user-user similarity matrix

    # Compute item-item similarity matrix (SIM)
    for i in range(0, N_items):  # Go through each user (i)
        for ip in range(0, i + 1):  # Compute similarities between all other items (ip)

            i_vec = np.transpose(U[:, i])  # i
            ip_vec = np.transpose(U[:, ip])  # i'

            if i == ip:  # SIM(i,i) = 1
                SIM[i, ip] = 1
            elif (not np.any(i_vec)) or (not np.any(ip_vec)):  # i_vec=0 or ip_vec=0 -> SIM(u,u)=0 (will be filtered out)
                SIM[i, ip] = 0
                SIM[ip, i] = SIM[i, ip]  # take advantage of SIM symmetric
            else:
                i_vec_norm = LA.norm(i_vec, 2)  # this norm (not norm-squared)
                ip_vec_norm = LA.norm(ip_vec, 2)  # this norm (not norm-squared)
                SIM[i, ip] = np.dot(i_vec, ip_vec) / (i_vec_norm * ip_vec_norm)  # cosine similarity
                SIM[ip, i] = SIM[i, ip]  # take advantage of SIM symmetric

    if ~(SIM.transpose() == SIM).all():
        raise IOError("SIM_ii similarity matrix is not symmetric!")

    return SIM