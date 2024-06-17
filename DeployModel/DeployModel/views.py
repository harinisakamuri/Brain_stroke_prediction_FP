# views.py

from django.shortcuts import render
import pandas as pd
import joblib

# Load the model once at the start
model = joblib.load('brain_stroke_model.pkl')

def home(request):
    return render(request, 'home.html')

def result(request):
    try:
        # Get parameters from the request and map to expected column names
        params = {
            'gender': str(request.GET.get('Gender')),
            'age': float(request.GET.get('Age')),
            'hypertension': int(request.GET.get('Hypertension')),
            'heart_disease': int(request.GET.get('Heart disease')),
            'ever_married': str(request.GET.get('Ever married')),
            'work_type': str(request.GET.get('Work type')),
            'Residence_type': str(request.GET.get('Residence type')),
            'avg_glucose_level': float(request.GET.get('Glucose level')),
            'bmi': float(request.GET.get('BMI')),
            'smoking_status': str(request.GET.get('Smoking status')),
        }

        # Convert parameters to a DataFrame
        df = pd.DataFrame([params])

        # Check for any NaN values and handle them
        if df.isnull().values.any():
            print("NaN values found in input data")
            df.fillna(0, inplace=True)
        else:
            print("No NaN values in input data")

        print("DataFrame contents before prediction:")
        print(df)
        print("DataFrame dtypes:")
        print(df.dtypes)

        # Now use this DataFrame to make predictions
        prediction = model.predict(df)
        print("Prediction:", prediction)  # Add this line for debugging

        # Return the result (assuming you have a template to render)
        return render(request, 'result.html', {'params': params, 'prediction': prediction[0]})
    except Exception as e:
        print("Error:", e)  # Add this line for debugging
        return render(request, 'result.html', {'error': str(e)})
