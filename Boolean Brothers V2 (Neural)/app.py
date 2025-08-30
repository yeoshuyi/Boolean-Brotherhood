from flask import Flask, request, render_template
from main import selectLocation, checkReview

app = Flask(__name__)

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/submit", methods=["POST"])
def submit():
    location = request.form["location"]
    if location == "":
        return render_template("index.html")
    selected = selectLocation(location)

    if selected[1] == 0:
        return "No results found."
    elif selected[1] == 1:
        return render_template("index.html", review=checkReview(selected[0][0]['place_id']))
    else:
        return render_template("multi.html", value=selected[0])

@app.route("/submit2", methods=["POST"])
def submit2():
    place = request.form["a"]
    return render_template("index.html", review=checkReview(place))

if __name__ == "__main__":
    app.run(debug=True)
