# Cosine Similarity
import numpy as np
import pandas as pd


def cosine_sim(vec_A, vec_B):
    dot_product = np.dot(vec_A, vec_B.T)

    norm_A = np.linalg.norm(vec_A, axis=1, keepdims=True)
    norm_B = np.linalg.norm(vec_B, axis=1, keepdims=True)

    cosine_similarity = dot_product / (norm_A * norm_B.T)

    return cosine_similarity

# Fungsi menghitung cosine similarity ke data


def calculate_cosine(dataSI, dataMK):
    similarity_matrix = cosine_sim(dataSI, dataMK)
    similarity_df = pd.DataFrame(similarity_matrix, index=[f'SI_{i}' for i in range(
        len(dataSI))], columns=[f'MK_{i}' for i in range(len(dataMK))])

    return similarity_df.round(3).T

# Directed Sim A->B


def direct_sim_AB(vec_A, vec_B):
    dot_product = np.dot(vec_A, vec_B.T)

    norm_A = np.linalg.norm(vec_A, axis=1, keepdims=True)
    norm_B = np.linalg.norm(vec_B, axis=1, keepdims=True)

    similarity_matrix = dot_product / (norm_A)

    return similarity_matrix


def calculate_directed_AB(dataSI, dataMK):
    similarity_matrix = direct_sim_AB(dataSI, dataMK)
    similarity_df = pd.DataFrame(similarity_matrix, index=[f'SI_{i}' for i in range(
        len(dataSI))], columns=[f'MK_{i}' for i in range(len(dataMK))])

    return similarity_df.round(3).T

# Directed Sim B->A
def direct_sim_BA(vec_A, vec_B):
    dot_product = np.dot(vec_A, vec_B.T)

    norm_A = np.linalg.norm(vec_A, axis=1, keepdims=True)
    norm_B = np.linalg.norm(vec_B, axis=1, keepdims=True)

    similarity_matrix = dot_product / (norm_B.T)

    return similarity_matrix


def calculate_directed_BA(dataSI, dataMK):
    similarity_matrix = direct_sim_BA(dataSI, dataMK)
    similarity_df = pd.DataFrame(similarity_matrix, index=[f'SI_{i}' for i in range(
        len(dataSI))], columns=[f'MK_{i}' for i in range(len(dataMK))])

    return similarity_df.round(3).T
