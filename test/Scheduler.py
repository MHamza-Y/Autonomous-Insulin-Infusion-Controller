import schedule
import time
from datetime import datetime, timedelta
from ValueClient import getInsulinvalues
def func():
    now = datetime.now()
    end_time = now.strftime("%H:%M:%S")
    five_minute = timedelta(minutes=5)
    start_time = now - five_minute
    start_time = start_time.strftime("%H:%M:%S")
    startDate = '2020-12-10'
    getInsulinvalues(startDate,startDate,start_time,end_time)
    print("this is python")

func()
schedule.every(5).minutes.do(func)

while True:
    schedule.run_pending()
    time.sleep(1)