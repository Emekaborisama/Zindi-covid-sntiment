

def enpoint1(text):
    url = "https://ilistener2-emekaborisama.cloud.okteto.net/named_entity"
    payload={'text': 'Hi, Usman, i hope you are doing great. Usman: yes, i am fine and you. i went to see Mr Kunle and i am looking forward to his investment, althought he said we should have a meeting tomorrow by 2 pm and we agreed to the oil deal with Gen Dangote. I hope you will be availble to attend the meeting? Yes. Meanwhile, pls come prepared incase Gen Dangote has question or is curious about how your application works. alright boss, see you at the meeting Usman.'}
    files=[

    ]
    headers = {}
    response = requests.request("POST", url, headers=headers, data=payload, files=files)
    return (response.text)