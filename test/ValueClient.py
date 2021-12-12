import http.client
import json
from controller import PPOController

def getInsulinvalues(startDate, endDate,startTime, endTime):
    conn = http.client.HTTPSConnection("sandbox-api.dexcom.com")
    headers = {
    'authorization': "Bearer eyJ0eXAiOiJKV1QiLCJhbGciOiJSUzI1NiJ9.eyJzdWIiOiIwMDc4ZjM5Mi1jMmM4LTQ5NjktYWNiNy0zOWVhOGQ0Yzk4OGMiLCJhdWQiOiJodHRwczovL3NhbmRib3gtYXBpLmRleGNvbS5jb20iLCJzY29wZSI6WyJlZ3YiLCJjYWxpYnJhdGlvbiIsImRldmljZSIsImV2ZW50Iiwic3RhdGlzdGljcyIsIm9mZmxpbmVfYWNjZXNzIl0sImlzcyI6Imh0dHBzOi8vc2FuZGJveC1hcGkuZGV4Y29tLmNvbSIsImV4cCI6MTYzOTI0MDI3OCwiaWF0IjoxNjM5MjMzMDc4LCJjbGllbnRfaWQiOiJUZVJBYUJ1TktaeVFkdUFIcTBsWllDd3BZZGw2b2RkcyJ9.geewjo_EujE9Tp7OHR_hHwrT78TFpfuUgJfcz3m49qy5CDdo_TCpfQx5IDkrEIByeqKKw7v2SZzkj6l2xNGxIwND3ucG1iEXUbOVgoELE9huhghqJiwOdVq-cBtid5b4t2gjEqJqADvVozKKRC_-WF0nKSeAf7gmOWtDqq8cQTyVjtr5nAcwYnFmIa0SEV4jdJ0m_mJnfIblDLVssvEm4eJvfmUmTxNc2CXAOZnH66vT4s7vfTxefG2cgvYyBUKUP46FsC2emJ546Qm-vc-i7YKfXjhex-siOqhXRzfiW3pBI1fv3EEZEVXwnFmBk_EVmbTLEA7FZ40jBW4O53Uf7w"
    }
    request = "/v2/users/self/egvs?startDate="+startDate+"T"+startTime+"&endDate="+endDate+"T"+endTime
    conn.request("GET", request, headers=headers)
    res = conn.getresponse()
    data = res.read()
    resp_dict = json.loads(data)
    observation = resp_dict.get('egvs')[0].get('value')
    pPOController = PPOController()
    pPOController.policy(observation,'','','')
    print(observation)

