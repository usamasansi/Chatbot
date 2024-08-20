# from pymongo import MongoClient
import pandas as pd

data_url = 'http://archive.ics.uci.edu/ml/machine-learning-databases/autos/imports-85.data'

df = pd.read_csv(data_url, header=None)

print(df)

# myclient = MongoClient("mongodb://localhost:27017/")

# mydb = myclient["pymongo"]
# cursor = mydb["customer"]

# df = pd.Dataframe(list(cursor))
# mydict = { "name":"usama", "address":"chishtian" }

# x = cursor.insert_one(mydict)
# print(x.inserted_id)
