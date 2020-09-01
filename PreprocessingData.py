import numpy as np
import pandas as pd
import re

path = 'data/DivinaCommedia.txt'

file = open(path, 'r')

cont = 0
riga = 0

df = {
    # "id": [],
    "0": [],
    "1": [],
    "2": [],
    "1_syllables": [],
    "2_syllables": [],
    "3_syllables": [],
}

for line in file:
    l = line.strip()

    # check useless lines
    if l is "" or \
            l[0] == "[" or \
            l[0] == "" or \
            l == "INFERNO" or \
            l == "PURGATORIO" or \
            l == "PARADISO" or \
            l.startswith("Canto"):
        continue

    # Replace exceptional characters
    l = l.replace("ä", "a")
    l = l.replace("é", "è")
    l = l.replace("ë", "è")
    l = l.replace("Ë", "E")
    l = l.replace("ï", "i")
    l = l.replace("Ï", "I")
    l = l.replace("ó", "ò")
    l = l.replace("ö", "o")
    l = l.replace("ü", "u")

    l = l.replace("(", "-")
    l = l.replace(")", "-")
    l = l.replace("[", "")
    l = l.replace("]", "")

    l = re.sub(r'[0-9]+', '', l)
    l = l.replace(" \n", "\n")
    # print(l)

    if cont == 0:
        df["0"].append(l)
        df["1_syllables"].append(11)
        cont += 1
    elif cont == 1:
        df["1"].append(l)
        df["2_syllables"].append(11)
        cont += 1
    elif cont == 2:
        df["2"].append(l)
        df["3_syllables"].append(11)
        cont = 0
        # df["id"].append(str(riga))
        # riga += 1
    else:
        print("HOUSTON WE HAVE PRoBLEM")

# the Inferno have one terzina more than other so I delete it
df["0"] = df["0"][:-1]
df["1_syllables"] = df["1_syllables"][:-1]


dataframe = pd.DataFrame(df)
print(dataframe)
dataframe.to_csv("data/DivinaCommedia.csv")
