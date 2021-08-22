from flask import abort, request, jsonify, make_response, send_file, render_template
import io

from kColor.webapp import app

from kColor.algs.kColor import kColorFromDict

@app.route("/hello")
def hello_world():
    return "Hello World"

@app.route("/")
def index():
    return render_template("main.html")

@app.route("/submit", methods = ['POST'])
def submit():
    print(request)
    print(request.form)
    #print(request.form["colors"])
    print(request.form.to_dict(flat=False))

    print("about to print files")
    print(request.files)
    print("printed files")

    # check if the post request has the file part
    if 'inFile' not in request.files:
        print('No file part')
        return abort(400)

    imgFile = request.files['inFile']

    # If the user does not select a file, the browser submits an
    # empty file without a filename.
    if imgFile.filename == '':
        print('No file name')
        return abort(400)

    fileStr = imgFile.stream.read()
    imgFile.stream.close()
    try:
        outBytes, filename = kColorFromDict(fileStr, request.form.to_dict(flat=False))
        return send_file(io.BytesIO(outBytes), as_attachment=True, attachment_filename=filename)
    except Exception as e:
        print(e)
        return abort(400)
