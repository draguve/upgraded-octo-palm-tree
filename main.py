import numpy as np
import requests
from tqdm import tqdm
from pprint import pprint
from scipy.spatial import cKDTree
from scipy.optimize import linear_sum_assignment

def get_data(hour:int):
    try:
        url = f"https://a.windbornesystems.com/treasure/{hour:02}.json"
        data = requests.get(url).json()
        data = np.array(data)
        data = data[~np.isnan(data).any(axis=1)]
        return data
    except:
        return None

def get_all_data():
    all_data = []
    hours = []
    for hour in tqdm(range(2),"getting data"):
        data = get_data(hour)
        if data is not None:
            all_data.append(data)
            hours.append(hour)
    return list(zip(hours,all_data))

def find_nearest_one_to_one(array1: np.ndarray, array2: np.ndarray):
    if array1.shape[1] != 3 or array2.shape[1] != 3:
        raise ValueError("Both arrays must have shape (N, 3) and (M, 3)")
    dist_matrix = np.linalg.norm(array1[:, None] - array2[None, :], axis=2)
    row_ind, col_ind = linear_sum_assignment(dist_matrix)
    return [(i, j, dist_matrix[i, j]) for i, j in zip(row_ind, col_ind)]

if __name__ == '__main__':
    data = get_all_data()
    matches = find_nearest_one_to_one(data[0][1],data[1][1])
    for i, j, dist in matches:
        print(f"Point {i} in array1 -> Point {j} in array2 (Distance: {dist:.4f})")