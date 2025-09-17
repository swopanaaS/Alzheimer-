import requests
import json

# URL of your Flask API
url = "http://localhost:5000/api/predict"

# Updated JSON data for the API request
data = {
    "BMI": 39.46303422,
    "AlcoholConsumption": 9.811292129,
    "PhysicalActivity": 8.819950351,
    "DietQuality": 0.434020278,
    "SleepQuality": 7.6440973,
    "CholesterolTotal": 224.5264374,
    "CholesterolLDL": 160.7564751,
    "CholesterolHDL": 68.32831269,
    "CholesterolTriglycerides": 172.7852201,
    "MMSE": 19.69610671,
    "FunctionalAssessment": 8.246968117,
    "MemoryComplaints": 0,
    "BehavioralProblems": 0,
    "ADL": 2.018940065
}

# Send a POST request to the API
try:
    # Make sure the data is being sent as JSON
    response = requests.post(url, json=data)

    # Check the response status code and print the prediction
    if response.status_code == 200:
        prediction = response.json()  # Parse the JSON response
        print("Prediction:", prediction)
    else:
        print(f"Failed to get prediction. Status code: {response.status_code}")
        print("Response:", response.text)

except Exception as e:
    print("An error occurred:", str(e))