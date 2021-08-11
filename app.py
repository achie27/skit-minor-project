# -*- coding: utf-8 -*-

from scripts import tabledef
from scripts import forms
from scripts import helpers
from flask import Flask, redirect, url_for, render_template, request, session, jsonify
import json
import sys
import os
import json
import numpy as np
import tensorflow as tf
from pred_model import SketchRec

app = Flask(__name__)

categories = [
	'book', 'boomerang', 'bottlecap', 'bracelet', 'brain', 'bread', 
	'bridge', 'broccoli', 'broom', 'bucket', 'bulldozer', 'bus', 
	'bush', 'butterfly', 'cactus', 'cake', 'calculator', 'calendar', 
	'camera', 'camouflage', 'campfire', 'candle', 'cannon', 'canoe', 
	'car', 'carrot', 'castle', 'cat', 'ceiling fan', 'cell phone'
]

mx_len, total_cats = 250, 30

# ======== Routing =========================================================== #


# --------- Classification --------------------------------------------------- #


@app.route('/classify', methods = ['GET', 'POST'])
def classify():

    model = SketchRec(mx_len, total_cats)
    inks = tf.placeholder('float32', [None, mx_len, 3])
    labels = tf.placeholder('int32', None)
    shapes = tf.placeholder('int32', [None, 2])

    sketches = [np.zeros((mx_len, 3))]
    shpes = [(mx_len, 3)]

    inkarray = json.loads(request.args.get('image', type=str))

    stroke_lengths = [len(stroke[0]) for stroke in inkarray]
    total_points = sum(stroke_lengths)

    if total_points > mx_len:
        return "Drawing too detailed"
    
    np_ink = np.zeros((total_points, 3), dtype=np.float32)
    current_t = 0
    for stroke in inkarray:
        for i in [0, 1]:
            np_ink[current_t:(current_t + len(stroke[0])), i] = stroke[i]
        current_t += len(stroke[0])
        np_ink[current_t - 1, 2] = 1  # stroke_end

    lower = np.min(np_ink[:, 0:2], axis=0)
    upper = np.max(np_ink[:, 0:2], axis=0)
    scale = upper - lower
    scale[scale == 0] = 1
    np_ink[:, 0:2] = (np_ink[:, 0:2] - lower) / scale

    np_ink = np_ink[1: ] - np_ink[0:-1]
    np_ink[np_ink[:, 2] == -1] = 0


    sketches.append(np_ink)
    shpes.append(np_ink.shape)
    
    sh = shpes[1][0]
    x = np.zeros((mx_len-sh, 3))
    if sketches[1][-1][2] == 0 :
        x[0][2] = 1

    sketches[1] = np.vstack((sketches[1], x))

    tf_obj = SketchRec(mx_len, total_cats)
    return jsonify({
        "data" : categories[tf_obj.predict(
            inks, shapes, labels, sketches, [0], shpes, model = "mk3_30cat_1000examples_150epoch_1000sketches"
        )[1]]
    })

# -------- Login ------------------------------------------------------------- #
@app.route('/', methods=['GET', 'POST'])
def login():
    if not session.get('logged_in'):
        form = forms.LoginForm(request.form)
        if request.method == 'POST':
            username = request.form['username'].lower()
            password = request.form['password']
            if form.validate():
                if helpers.credentials_valid(username, password):
                    session['logged_in'] = True
                    session['username'] = username
                    return json.dumps({'status': 'Login successful'})
                return json.dumps({'status': 'Invalid user/pass'})
            return json.dumps({'status': 'Both fields required'})
        return render_template('login.html', form=form)
    user = helpers.get_user()
    return render_template('home.html', user=user)


@app.route("/logout")
def logout():
    session['logged_in'] = False
    return redirect(url_for('login'))


# -------- Signup ---------------------------------------------------------- #
@app.route('/signup', methods=['GET', 'POST'])
def signup():
    if not session.get('logged_in'):
        form = forms.LoginForm(request.form)
        if request.method == 'POST':
            username = request.form['username'].lower()
            password = helpers.hash_password(request.form['password'])
            email = request.form['email']
            if form.validate():
                if not helpers.username_taken(username):
                    helpers.add_user(username, password, email)
                    session['logged_in'] = True
                    session['username'] = username
                    return json.dumps({'status': 'Signup successful'})
                return json.dumps({'status': 'Username taken'})
            return json.dumps({'status': 'User/Pass required'})
        return render_template('login.html', form=form)
    return redirect(url_for('login'))


# -------- Settings ---------------------------------------------------------- #
@app.route('/settings', methods=['GET', 'POST'])
def settings():
    if session.get('logged_in'):
        if request.method == 'POST':
            password = request.form['password']
            if password != "":
                password = helpers.hash_password(password)
            email = request.form['email']
            helpers.change_user(password=password, email=email)
            return json.dumps({'status': 'Saved'})
        user = helpers.get_user()
        return render_template('settings.html', user=user)
    return redirect(url_for('login'))


# ======== Main ============================================================== #
if __name__ == "__main__":
    app.secret_key = os.urandom(12)  # Generic key for dev purposes only
    app.run(debug=True, use_reloader=True)

