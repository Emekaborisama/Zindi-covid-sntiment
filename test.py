
import requests
def enpoint1(text2):
    url = "https://ilistener2-emeka101.cloud.okteto.net/named_entity"
    payload={'text': text2}
    files=[
    ]
    headers = {}
    response = requests.request("POST", url, headers=headers, data=payload, files=files)
    return (response.text)