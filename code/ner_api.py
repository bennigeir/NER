# -*- coding: utf-8 -*-
"""
Created on Tue Nov  3 08:12:00 2020

@author: Benedikt
"""

import json

from flask import Flask, request, jsonify
from test_ner import run_model


app = Flask(__name__)


@app.route('/', methods=['GET'])
def ner():
    if request.method == 'GET':
        if set(['query']).issubset(set(request.args)):
            query = request.args.get('query')
            try:
                out = run_model(query)
                response = jsonify({'results': out})
                response.status_code = 201
            except Exception as e:
                response = jsonify({'error': e})
                response.status_code = 400
        else:
            response = json.dumps({'error': 'Parameters missing'})
            response.status_code = 400
    else:
        response = json.dumps({'message': 'Method Not Allowed'})
        response.status_code = 405
    return response


if __name__ == '__main__':
    app.run(debug=True, host='localhost',use_reloader=False)