
import sys
import joblib
import os
import warnings
import pandas as pd

# Suppress warnings
warnings.filterwarnings("ignore")

def predict(cgpa):
    try:
        model_path = os.path.join(os.path.dirname(__file__), "model.joblib")
        if not os.path.exists(model_path):
            print("Error: model.joblib not found. Please run the notebook first.")
            sys.exit(1)
            
        model = joblib.load(model_path)
        
        # The notebook model (logreg) was trained on 14 features:
        # ['gender', 'ssc_p', 'hsc_p', 'hsc_p', 'workex', 'etest_p', 'specialisation', 'mba_p', 
        #  'dummy_Arts', 'dummy_Commerce', 'dummy_Science', 'dummy_Comm&Mgmt', 'dummy_Others', 'dummy_Sci&Tech']
        
        # We create a single row DataFrame with average values from the dataset for most features
        # and use the user-provided CGPA for 'degree_p'
        # Since 'degree_p' wasn't in the list above (Wait, I need to check the list again)
        
        # Re-checking feature list from notebook:
        # feature_cols=['gender','ssc_p','hsc_p','hsc_p','workex','etest_p','specialisation','mba_p',
        #              'dummy_Arts','dummy_Commerce','dummy_Science','dummy_Comm&Mgmt','dummy_Others','dummy_Sci&Tech']
        # Actually, 'degree_p' is NOT in the feature_cols list of the notebook's logreg!
        # It seems the notebook dropped degree_p or I misread.
        
        # Let's check index.ipynb line 1888:
        # X=placement_coded.drop(['status'],axis=1)
        # placement_coded had degree_p. 
        # So X has degree_p.
        
        # Let's use the actual feature names from the model itself to be safe
        features = model.feature_names_in_
        
        # User input is on 0-10 scale, but model was trained on 0-100 percentage scale
        percentage = float(cgpa) * 10.0
        
        # Default values (Means from the dataset)
        data = {
            'gender': [1], # Male
            'ssc_p': [67.3],
            'hsc_p': [66.3],
            'degree_p': [percentage], # Converted from CGPA
            'workex': [0], # No
            'etest_p': [72.1],
            'specialisation': [1], # Mkt&Fin
            'mba_p': [62.2],
            'dummy_Arts': [0],
            'dummy_Commerce': [0],
            'dummy_Science': [0],
            'dummy_Comm&Mgmt': [0],
            'dummy_Others': [0],
            'dummy_Sci&Tech': [0]
        }
        
        # Create DataFrame with only the features the model expects
        X_input = pd.DataFrame(data)[features]
        
        # Prediction
        prediction = model.predict(X_input)[0]
        # Probability
        probability = model.predict_proba(X_input)[0][1]
        
        return prediction, probability
    except Exception as e:
        print(f"Error: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python predict.py <cgpa>")
        sys.exit(1)
    
    cgpa_input = sys.argv[1]
    res, prob = predict(cgpa_input)
    print(f"{res},{prob}")
