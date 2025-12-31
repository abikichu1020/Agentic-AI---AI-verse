from pymongo import MongoClient
import traceback

try:
    MONGO_URI = ""

    mongo_client = MongoClient(MONGO_URI)   # correct client
    db = mongo_client["interview_system"]   # USE THIS CLIENT

    print("=== First Interview Record ===")
    interview = db.interviews.find_one()
    print(interview)

except Exception as e:
    print("❌ MongoDB connection failed:", e)
    traceback.print_exc()
