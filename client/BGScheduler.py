import http.client
import json
import schedule
from datetime import datetime, timedelta
from requests_oauthlib import OAuth2Session
import os

os.environ['OAUTHLIB_INSECURE_TRANSPORT'] = '1'

client_id = r'3hXhPcWaZ7wOuD8EjH7BPjZVJYkLlwJI'
client_secret = r'9QJ0892tQ3VHad5C'
redirect_uri = 'https://BGController.com'
scope = ['offline_access']
response_type = 'code'
code = r'533d33c28705a6c8f06c2a3fde87da30'
oauth = OAuth2Session(client_id=client_id, redirect_uri=redirect_uri,
                      scope=scope)
token_resp = oauth.fetch_token(
    token_url='https://sandbox-api.dexcom.com/v2/oauth2/token',
    client_id=client_id, client_secret=client_secret, code=code)


def fetch_bg_values():
    date_time_pattern = '%Y-%m-%dT%H:%M:%S'
    now = datetime.now()
    five_minute = timedelta(days=4)
    end_time = now.strftime(date_time_pattern)
    start_time = now - five_minute
    start_time = start_time.strftime(date_time_pattern)

    res = oauth.get(
        f'https://sandbox-api.dexcom.com/v2/users/self/egvs?startDate={start_time}&endDate={end_time}')

    resp_dict = json.loads(res.content.decode('utf-8'))
    return resp_dict


def get_current_bg():
    bgvs = fetch_bg_values()
    return bgvs['egvs'][0]['value']

