import os 
import sys

def records():
    for records in os.listdir("./logs/"):
        if records.endswith('.csv'):
            t = "./logs/"+records
            os.unlink(t)

def log():
    for log in os.listdir("./logs/test_log/"):
        if log.endswith('.log'):
            s = "./logs/test_log/"+log
            os.unlink(s)

def curve():
    for curve in os.listdir("./logs/output_curve/"):
        if curve.endswith('.png'):
            y = "./logs/output_curve/"+curve
            os.unlink(y)

def genfit():
    for genfit in os.listdir("./logs/curve_log/"):
        if genfit.endswith('.csv'):
            j = "./logs/curve_log/"+genfit
            os.unlink(j)


if(len(sys.argv) == 2 or len(sys.argv) == 6):
    if(sys.argv[1] == 'a'):
        print("Cleaned all logs")
        records()
        log()
        curve()
        genfit()

    elif(sys.argv[1] == 's' and len(sys.argv) == 6 ):
        if (sys.argv[2] == '1'):
            print("Cleaned run logs")
            log()
        if (sys.argv[3] == '1'):
            print("Cleaned generation fitness logs")
            genfit()
        if (sys.argv[4] == '1'):
            print("Deleted curve exports")
            curve()
        if (sys.argv[5] == '1'):
            print("Cleaned record logs")
            records()

else:
    print("Invalid argument")
    sys.exit()
    