from flask import Flask, render_template, request,url_for
import predictions

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('form.html')

@app.route('/process', methods=['POST'])
def process():
    if request.method == 'POST':
        user_input = request.form['user_input']
        val = predictions.prediction_tweet(user_input)
        return render_template('form.html', val=val)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8000, debug=True)
    

