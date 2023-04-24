import pickle
from flask import Flask, request, render_template
import pandas as pd

# Load your pickled data

app = Flask(__name__)

# Define your Flask route
@app.route('/')
def home():
    return render_template('index.html')

@app.route('/', methods=['GET', 'POST'])
def predict():
    if request.method == 'POST':
        # Get the form data
        input1 = request.form['field1']
        input2 = request.form['field2']
        input3 = request.form['field3']
        input4 = request.form['field4']
        input5 = request.form['field5']

        dictt = {'iucr': input1, 'primary_type': input2, 'description': input3, 'fbi_code': input4, 'primary_type_grouped': input5}
        
        # Run your prediction using the loaded data and the form inputs
        # prediction = predict_model_param(model, dictt)
        model = pickle.load(open('model.pkl', 'rb'))
        iucr = pickle.load(open('ip1.pkl', 'rb'))
        ptype = pickle.load(open('ip2.pkl', 'rb'))
        desc = pickle.load(open('ip3.pkl', 'rb'))
        fbi = pickle.load(open('ip4.pkl', 'rb'))
        grp = pickle.load(open('ip5.pkl', 'rb'))
        list = [iucr, ptype, desc, fbi, grp]

        for i, key in enumerate(dictt.keys()):
            ip_dict = list[i]
            dictt[key] = ip_dict[dictt[key]]
        
        # dictt = {'iucr': li[0], 'primary_type': li[1], 'description': li[2], 'fbi_code': li[3], 'primary_type_grouped': li[4]}
        f1 = pd.DataFrame(dictt, index=[0])
        pred = model.predict(f1)
        # print(prediction)


        if(pred[0] == 1):
            prediction = "ARREST - POSITIVE. There might be a need for possible arrest because of a crime."
        else:
            prediction = "ARREST - NEGATIVE. There might be no crime and arrest is not imminent."

        # Return the prediction as a webpage
        return render_template('result.html', prediction=prediction)
    
    # If it's a GET request, just return the form
    return render_template('index.html')


if __name__ == '__main__':
    app.run()
