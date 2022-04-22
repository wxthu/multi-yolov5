import random
import numpy as np
import pandas as pd

def gen_rand(img_num=250, detector_num=6):

    rates= {}
    for i in range(img_num):
        rate = [bool(random.getrandbits(2)) for _ in range(detector_num)]
        rates.update({str(i) : rate})

    df = pd.DataFrame(data=rates)
    df.to_csv('mock_request_rate.csv', index=False)   

if __name__ == '__main__':
    gen_rand()
    df = pd.read_csv('mock_request_rate.csv')
    for col in df:
        print(df[col].values)
        print(sum(df[col].values))
        break
    
