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
    all_tracked_points = []  # List to store all tracked points across all hours

    # Initialize a dictionary to store the paths of each tracked point
    paths = {i: [data[0][1][i]] for i in range(len(data[0][1]))}

    # Iterate through each pair of consecutive hours
    for i in range(len(data) - 1):  # Looping through consecutive hours
        hour1, array1 = data[i]
        hour2, array2 = data[i + 1]

        # Find nearest matches for the points in array1 (current hour) and array2 (next hour)
        matches = find_nearest_one_to_one(array1, array2)

        # Update the paths for each tracked point
        for index, j, _ in matches:
            if index not in paths:  # Make sure the index exists in paths
                paths[index] = [array1[index]]  # Initialize the path if it doesn't exist
            paths[index].append(array2[j])  # Add the new point to the path

    # Convert paths dictionary to a list of lists
    all_tracked_points = list(paths.values())

    return all_tracked_points

if __name__ == '__main__':
    data = get_all_data()
    tracked_points = track_points(data)

    selected_point = 3
    print(f"Tracked points: {len(tracked_points)}")
    print(f"Showing data for point {selected_point}")

    print(f"\nObject {selected_point} tracked across hours:")
    for hour, point in enumerate(tracked_points[selected_point]):
        print(f"  Hour {hour} -> Point: {point}")
