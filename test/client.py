import http.client

conn = http.client.HTTPSConnection("sandbox-api.dexcom.com")

payload = "client_secret=Vl3nfxSw6KXeA4T9&client_id=TeRAaBuNKZyQduAHq0lZYCwpYdl6odds&code=533d33c28705a6c8f06c2a3fde87da30&grant_type=authorization_code&redirect_uri=https://insulininfusioncontroller.com"

headers = {
    'content-type': "application/x-www-form-urlencoded",
    'cache-control': "no-cache"
    }

conn.request("POST", "/v2/oauth2/token", payload, headers)

res = conn.getresponse()
data = res.read()

print(data.decode("utf-8"))
