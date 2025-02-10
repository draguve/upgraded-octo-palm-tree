import numpy as np
import requests
from tqdm import tqdm
from pprint import pprint

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
    for hour in tqdm(range(24),"getting data"):
        data = get_data(hour)
        if data is not None:
            all_data.append(data)
            hours.append(hour)
    return list(zip(hours,all_data))

if __name__ == '__main__':
    print(get_all_data())