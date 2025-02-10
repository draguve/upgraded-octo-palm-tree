import numpy as np
import requests
from tqdm import tqdm
from scipy.optimize import linear_sum_assignment

def get_data(hour: int):
    try:
        url = f"https://a.windbornesystems.com/treasure/{hour:02}.json"
        data = requests.get(url).json()
        data = np.array(data)
        data = data[~np.isnan(data).any(axis=1)]  # Remove rows with NaN values
        return data
    except:
        return None

def get_all_data():
    all_data = []
    hours = []
    for hour in tqdm(range(24), "getting data"):  # Looping over 24 hours
        data = get_data(hour)
        if data is not None:
            all_data.append(data)
            hours.append(hour)
    return list(zip(hours, all_data))

def find_nearest_one_to_one(array1: np.ndarray, array2: np.ndarray):
    if array1.shape[1] != 3 or array2.shape[1] != 3:
        raise ValueError("Both arrays must have shape (N, 3) and (M, 3)")
    dist_matrix = np.linalg.norm(array1[:, None] - array2[None, :], axis=2)
    row_ind, col_ind = linear_sum_assignment(dist_matrix)
    return [(i, j, dist_matrix[i, j]) for i, j in zip(row_ind, col_ind)]

def track_points(data):
    all_tracked_points = []
    paths = {i: [(data[0][0],data[0][1][i])] for i in range(len(data[0][1]))}
    for i in range(len(data) - 1):  # Looping through consecutive hours
        hour1, array1 = data[i]
        hour2, array2 = data[i + 1]
        matches = find_nearest_one_to_one(array1, array2)
        for index, j, _ in matches:
            if index not in paths:
                paths[index] = [(hour1,array1[index])]
            paths[index].append((hour2,array2[j]))
    all_tracked_points = list(paths.values())
    return all_tracked_points

def haversine(lat1, lon1, lat2, lon2):
    R = 6371000
    phi1 = np.radians(lat1)
    phi2 = np.radians(lat2)
    delta_phi = np.radians(lat2 - lat1)
    delta_lambda = np.radians(lon2 - lon1)
    a = np.sin(delta_phi / 2)**2 + np.cos(phi1) * np.cos(phi2) * np.sin(delta_lambda / 2)**2
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
    return R * c

def calculate_3d_distance(point1, point2):
    lat1, lon1, alt1 = point1
    lat2, lon2, alt2 = point2
    horizontal_distance = haversine(lat1, lon1, lat2, lon2)
    vertical_distance = (alt2 - alt1) * 1000  # Convert km to meters
    total_distance = np.sqrt(horizontal_distance**2 + vertical_distance**2)
    return total_distance

if __name__ == '__main__':
    data = get_all_data()
    tracked_points = track_points(data)

    selected_point = 3
    print(f"Tracked points: {len(tracked_points)}")
    print(f"Showing data for point {selected_point}")
    print(f"\nObject {selected_point} tracked across hours:")
    for hour, point in tracked_points[selected_point]:
        print(f"  Hour {hour} -> Point: {point}")

    hours,points = list(zip(*(tracked_points[selected_point])))
