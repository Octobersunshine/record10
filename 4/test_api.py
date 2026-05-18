import requests
import json

BASE_URL = "http://localhost:5000"

def test_health_check():
    print("=== Testing Health Check ===")
    response = requests.get(f"{BASE_URL}/health")
    print(f"Status: {response.status_code}")
    print(f"Response: {json.dumps(response.json(), indent=2)}\n")

def test_json_iqr_remove():
    print("=== Testing JSON - IQR Method - Remove Outliers ===")
    data = {
        "data": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 100, -50],
        "method": "iqr",
        "handle": "remove",
        "k": 1.5
    }
    response = requests.post(f"{BASE_URL}/detect", json=data)
    print(f"Status: {response.status_code}")
    print(f"Response: {json.dumps(response.json(), indent=2)}\n")

def test_json_zscore_mean():
    print("=== Testing JSON - Z-score Method - Replace with Mean ===")
    data = {
        "data": [10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 100, -50],
        "method": "zscore",
        "handle": "mean",
        "threshold": 2.0
    }
    response = requests.post(f"{BASE_URL}/detect", json=data)
    print(f"Status: {response.status_code}")
    print(f"Response: {json.dumps(response.json(), indent=2)}\n")

def test_json_median():
    print("=== Testing JSON - IQR Method - Replace with Median ===")
    data = {
        "data": [5, 7, 8, 9, 10, 11, 12, 13, 15, 200],
        "method": "iqr",
        "handle": "median"
    }
    response = requests.post(f"{BASE_URL}/detect", json=data)
    print(f"Status: {response.status_code}")
    print(f"Response: {json.dumps(response.json(), indent=2)}\n")

def test_csv_upload():
    print("=== Testing CSV File Upload ===")
    files = {'file': open('test_data.csv', 'rb')}
    data = {
        'method': 'iqr',
        'handle': 'remove',
        'k': 1.5
    }
    response = requests.post(f"{BASE_URL}/detect", files=files, data=data)
    print(f"Status: {response.status_code}")
    print(f"Response: {json.dumps(response.json(), indent=2)}\n")

def test_sliding_window_json():
    print("=== Testing Sliding Window Method - JSON ===")
    data = {
        "data": [1, 2, 3, 4, 5, 6, 100, 7, 8, 9],
        "method": "sliding_window",
        "handle": "mean",
        "window_size": 5,
        "k": 2.0,
        "center": False
    }
    response = requests.post(f"{BASE_URL}/detect", json=data)
    print(f"Status: {response.status_code}")
    result = response.json()
    print(f"Outliers detected: {result['detection_report']['outlier_values']}")
    print(f"Cleaned data: {result['handling_report']['cleaned_data']}\n")

def test_sliding_window_centered():
    print("=== Testing Sliding Window - Centered Mode ===")
    data = {
        "data": [10, 10, 10, 50, 10, 10, 10],
        "method": "sliding_window",
        "handle": "median",
        "window_size": 3,
        "k": 1.5,
        "center": True
    }
    response = requests.post(f"{BASE_URL}/detect", json=data)
    print(f"Status: {response.status_code}")
    result = response.json()
    print(f"Outliers detected: {result['detection_report']['outlier_values']}")
    print(f"Cleaned data: {result['handling_report']['cleaned_data']}\n")

def test_zscore_zero_std():
    print("=== Testing Z-score - Zero Standard Deviation (Constant Data) ===")
    data = {
        "data": [5, 5, 5, 5, 5, 5, 5],
        "method": "zscore",
        "handle": "remove",
        "threshold": 3.0
    }
    response = requests.post(f"{BASE_URL}/detect", json=data)
    print(f"Status: {response.status_code}")
    print(f"Response: {json.dumps(response.json(), indent=2)}\n")

def test_zscore_zero_std_mean_handle():
    print("=== Testing Z-score - Zero Std with Mean Handling ===")
    data = {
        "data": [10, 10, 10, 10],
        "method": "zscore",
        "handle": "mean",
        "threshold": 2.0
    }
    response = requests.post(f"{BASE_URL}/detect", json=data)
    print(f"Status: {response.status_code}")
    print(f"Response: {json.dumps(response.json(), indent=2)}\n")

def test_iqr_constant_data():
    print("=== Testing IQR - Constant Data ===")
    data = {
        "data": [7, 7, 7, 7, 7],
        "method": "iqr",
        "handle": "remove",
        "k": 1.5
    }
    response = requests.post(f"{BASE_URL}/detect", json=data)
    print(f"Status: {response.status_code}")
    print(f"Response: {json.dumps(response.json(), indent=2)}\n")

if __name__ == "__main__":
    try:
        test_health_check()
        test_json_iqr_remove()
        test_json_zscore_mean()
        test_json_median()
        test_csv_upload()
        test_sliding_window_json()
        test_sliding_window_centered()
        test_zscore_zero_std()
        test_zscore_zero_std_mean_handle()
        test_iqr_constant_data()
        print("All tests completed!")
    except requests.exceptions.ConnectionError:
        print("Error: Could not connect to the server.")
        print("Please start the Flask server first: python app.py")
    except Exception as e:
        print(f"Error: {e}")
