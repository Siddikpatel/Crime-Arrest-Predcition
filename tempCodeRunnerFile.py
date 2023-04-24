@app.route('/', methods=['GET', 'POST'])
# def home():
#     if request.method == 'POST':
#         # Get the form data
#         input1 = request.form['field1']
#         input2 = request.form['field2']
#         input3 = request.form['field3']
#         input4 = request.form['field4']
#         input5 = request.form['field5']

#         dictt = {'iucr': input1, 'primary_type': input2, 'description': input3, 'fbi_code': input4, 'primary_type_grouped': input5}
        
#         # Run your prediction using the loaded data and the form inputs
#         prediction = predict_model_param(model, dictt)
        
#         # Return the prediction as a webpage
#         return render_template('result.html', prediction=prediction)
    
#     # If it's a GET request, just return the form
#     return render_template('index.html')