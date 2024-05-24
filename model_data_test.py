import requests

# URL of your Flask API
url = 'http://127.0.0.1:5000/predict'

# Path to the CSV file
files = {'file': open('articles\\new_articles.csv', 'rb')}

# Send POST request
response = requests.post(url, files=files)

# Print the response
print(response.json())
