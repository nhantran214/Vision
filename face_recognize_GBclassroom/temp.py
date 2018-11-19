import requests
import json
import cv2

#server address
server_ip = '192.168.1.182'
addr = 'http://%s:8080' % (server_ip)
test_url = addr + '/api/test'

# prepare headers for http request
content_type = 'image/jpeg'
headers = {'content-type': content_type}

img = cv2.imread('FaceDb/KimNgan/KimNgan (4).jpeg')
print (img.shape)
# encode image as jpeg
_, img_encoded = cv2.imencode('.jpg', img)
# send http request with image and receive response
response = requests.post(test_url, data=img_encoded.tostring(), headers=headers)
# decode response
# print(response)
print(json.loads(response.text))