import os 

for records in os.listdir("./logs/"):
    if records.endswith('.txt'):
        t = "./logs/"+records
        os.unlink(t)


for log in os.listdir("./logs/test_log/"):
    if log.endswith('.log'):
        s = "./logs/test_log/"+log
        os.unlink(s)

for curve in os.listdir("./logs/output_curve/"):
    if curve.endswith('.png'):
        y = "./logs/output_curve/"+curve
        os.unlink(y)


for genfit in os.listdir("./logs/curve_log/"):
    if genfit.endswith('.csv'):
        j = "./logs/curve_log/"+genfit
        os.unlink(j)
