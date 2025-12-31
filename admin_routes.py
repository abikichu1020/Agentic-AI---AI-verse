from flask import Blueprint, jsonify, request, session
from pymongo import MongoClient

# -----------------------
# Correct Blueprint
# -----------------------
admin_bp = Blueprint("admin", __name__)

# Mongo Connection
MONGO_URI = "mongodb+srv://i103:aishu@cluster0.zxk94yv.mongodb.net/?appName=Cluster0"
client = MongoClient(MONGO_URI)
db = client["interview_system"]

students_col = db["students"]
interviews_col = db["interviews"]

# Default Admin Credentials
ADMIN_EMAIL = "admin123@edu"
ADMIN_PASSWORD = "admin"

# -----------------------
# Helper
# -----------------------
def require_admin():
    return session.get("admin_logged_in") is True


# -----------------------
# LOGIN
# -----------------------
@admin_bp.route("/login", methods=["POST"])
def admin_login():
    data = request.get_json()
    email = data.get("email", "").lower().strip()
    password = data.get("password", "").strip()

    if email == ADMIN_EMAIL and password == ADMIN_PASSWORD:
        session["admin_logged_in"] = True
        session["admin_email"] = email
        return jsonify({"success": True})

    return jsonify({"success": False, "error": "Invalid email or password"}), 401


@admin_bp.route("/logout", methods=["POST"])
def admin_logout():
    session.clear()
    return jsonify({"success": True})


# -----------------------
# DASHBOARD METRICS
# -----------------------
@admin_bp.route("/dashboard", methods=["GET"])
def dashboard():
    if not require_admin():
        return jsonify({"error": "Unauthorized"}), 401

    total_students = students_col.count_documents({})
    total_interviews = interviews_col.count_documents({})

    return jsonify({
        "success": True,
        "total_students": total_students,
        "total_interviews": total_interviews,
    })


# -----------------------
# GET ALL STUDENTS
# -----------------------
@admin_bp.route("/students", methods=["GET"])
def get_students():
    if not require_admin():
        return jsonify({"error": "Unauthorized"}), 401

    data = list(students_col.find({}, {"_id": 0}))
    return jsonify({"success": True, "students": data})


# -----------------------
# GET SINGLE STUDENT + INTERVIEWS
# -----------------------
@admin_bp.route("/student", methods=["GET"])
def get_student():
    if not require_admin():
        return jsonify({"error": "Unauthorized"}), 401

    email = request.args.get("email")
    student = students_col.find_one({"email": email}, {"_id": 0})

    if not student:
        return jsonify({"error": "Student not found"}), 404

    interviews = list(interviews_col.find({"student_email": email}, {"_id": 0}))

    return jsonify({
        "success": True,
        "student": student,
        "interviews": interviews
    })


# -----------------------
# GET SINGLE INTERVIEW REPORT
# -----------------------
@admin_bp.route("/interview", methods=["GET"])
def interview_report():
    if not require_admin():
        return jsonify({"error": "Unauthorized"}), 401

    iid = request.args.get("id")
    interview = interviews_col.find_one({"interview_id": iid}, {"_id": 0})

    if not interview:
        return jsonify({"error": "Not found"}), 404

    return jsonify({
        "success": True,
        "report": interview.get("final_report", {}),
        "conversation": interview.get("conversation_history", []),
        "evaluations": interview.get("response_analyses", [])
    })
@admin_bp.route("/stats", methods=["GET"])
def admin_stats():
    if not require_admin():
        return jsonify({"error": "Unauthorized"}), 401

    # Basic totals
    total_students = students_col.count_documents({})
    total_interviews = interviews_col.count_documents({})

    avg = list(students_col.aggregate([
        {"$group": {"_id": None, "avg_score": {"$avg": "$average_score"}}}
    ]))
    average_score = avg[0]["avg_score"] if avg else 0

    # Top 5 students
    top_students = list(
        students_col.find({}, {"_id": 0})
        .sort("average_score", -1)
        .limit(5)
    )

    # Weak 5 students
    weak_students = list(
        students_col.find({}, {"_id": 0})
        .sort("average_score", 1)
        .limit(5)
    )

    return jsonify({
        "success": True,
        "total_students": total_students,
        "total_interviews": total_interviews,
        "average_score": round(average_score, 2),
        "top_students": top_students,
        "weak_students": weak_students
    })
