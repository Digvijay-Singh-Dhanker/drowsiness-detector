import pymongo
from pymongo import MongoClient
cluster = MongoClient("mongodb+srv://admin:1234@cluster0.wlcs7yd.mongodb.net/?retryWrites=true&w=majority")
db = cluster["employee"]
collection = db["records"]

post = {"_id":0, "name":"tim", "employee_eff":100}
collection.insert_one(post)