import os
import subprocess
from pathlib import Path
import pandas as pd

from sklearn.ensemble import RandomForestRegressor

import mlflow

TARGET_COL = "cost"

NUMERIC_COLS = [
    "distance",
    "dropoff_latitude",
    "dropoff_longitude",
    "passengers",
    "pickup_latitude",
    "pickup_longitude",
    "pickup_weekday",
    "pickup_month",
    "pickup_monthday",
    "pickup_hour",
    "pickup_minute",
    "pickup_second",
    "dropoff_weekday",
    "dropoff_month",
    "dropoff_monthday",
    "dropoff_hour",
    "dropoff_minute",
    "dropoff_second",
]

CAT_NOM_COLS = [
    "store_forward",
    "vendor",
]

CAT_ORD_COLS = [
]

def test_evaluate_model():
    
    test_data = "/tmp/test"
    model_input = "/tmp/model"
    evaluation_output = "/tmp/evaluate"
    model_name = "taxi-model"
    runner = "LocalRunner"

    os.makedirs(test_data, exist_ok = True)
    os.makedirs(model_input, exist_ok = True)
    os.makedirs(evaluation_output, exist_ok = True)


    data = {
        'cost': [4.5, 6.0, 9.5, 4.0, 6.0, 11.5, 25.0, 3.5, 5.0, 11.0, 7.5, 24.5, 9.5,
                7.5, 6.0, 5.0, 9.0, 25.5, 17.5, 52.0],
        'distance': [0.83, 1.27, 1.8, 0.5, 0.9, 2.72, 6.83, 0.45, 0.77, 2.2, 1.5, 6.27,
                    2.0, 1.54, 1.24, 0.75, 2.2, 7.0, 5.1, 18.51],
        'dropoff_hour': [21, 21, 9, 17, 10, 13, 17, 10, 2, 1, 16, 18, 20, 20, 1, 17,
                        21, 16, 4, 10],
        'dropoff_latitude': [40.69454574584961, 40.81214904785156, 40.67874145507813,
                            40.75471496582031, 40.66966247558594, 40.77496337890625,
                            40.75603103637695, 40.67219161987305, 40.66605758666992,
                            40.69973754882813, 40.61215972900391, 40.74581146240234,
                            40.78779602050781, 40.76130676269531, 40.72980117797852,
                            40.71107864379883, 40.747501373291016, 40.752384185791016,
                            40.66606140136719, 40.64547729492188],
        'dropoff_longitude': [-73.97611236572266, -73.95975494384766,
                            -73.98030853271484, -73.92549896240234,
                            -73.91104125976562, -73.89237213134766,
                            -73.94535064697266, -74.01203918457031,
                            -73.97817993164062, -73.99366760253906,
                            -73.94902801513672, -73.98792266845703,
                            -73.95561218261719, -73.8807601928711, -73.9117202758789,
                            -73.96553039550781, -73.9442138671875,
                            -73.97544860839844, -73.87281036376953,
                            -73.77632141113281],
        'dropoff_minute': [5, 54, 57, 52, 34, 20, 5, 8, 37, 27, 21, 5, 26, 46, 25, 1,
                        5, 20, 41, 46],
        'dropoff_month': [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
        'dropoff_monthday': [3, 19, 5, 8, 29, 30, 8, 4, 9, 14, 12, 9, 14, 17, 10, 9, 8,
                            2, 15, 21],
        'dropoff_second': [52, 37, 28, 20, 59, 20, 38, 52, 43, 24, 59, 29, 58, 11, 3,
                        4, 34, 21, 6, 36],
        'dropoff_weekday': [6, 1, 1, 4, 4, 5, 4, 0, 5, 3, 1, 5, 3, 6, 6, 5, 4, 5, 4,
                            3],
        'passengers': [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 1, 1],
        'pickup_hour': [21, 21, 9, 17, 10, 13, 16, 10, 2, 1, 16, 17, 20, 20, 1, 16, 20,
                        15, 4, 10],
        'pickup_latitude': [40.6938362121582, 40.80146789550781, 40.6797981262207,
                            40.76081848144531, 40.66493988037109, 40.74625396728516,
                            40.80010223388672, 40.67601776123047, 40.67120361328125,
                            40.68327331542969, 40.6324462890625, 40.71521377563477,
                            40.80733871459961, 40.750484466552734, 40.7398796081543,
                            40.71691131591797, 40.773414611816406, 40.79001235961914,
                            40.660118103027344, 40.78546905517578],
        'pickup_longitude': [-73.98726654052734, -73.94845581054688, -73.9554443359375,
                            -73.92293548583984, -73.92304229736328, -73.8973159790039,
                            -73.9500503540039, -74.0144271850586, -73.98458099365234,
                            -73.96582794189453, -73.94767761230469,
                            -73.96052551269531, -73.96453094482422,
                            -73.88248443603516, -73.92410278320312,
                            -73.95661163330078, -73.92512512207031,
                            -73.94800567626953, -73.95987701416016,
                            -73.94915771484375],
        'pickup_minute': [2, 49, 46, 49, 28, 8, 32, 6, 34, 14, 14, 35, 17, 38, 20, 56,
                        56, 49, 23, 18],
        'pickup_month': [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
        'pickup_monthday': [3, 19, 5, 8, 29, 30, 8, 4, 9, 14, 12, 9, 14, 17, 10, 9, 8,
                            2, 15, 21],
        'pickup_second': [35, 17, 18, 12, 21, 46, 18, 22, 5, 45, 12, 52, 20, 8, 28, 54,
                        41, 53, 43, 2],
        'pickup_weekday': [6, 1, 1, 4, 4, 5, 4, 0, 5, 3, 1, 5, 3, 6, 6, 5, 4, 5, 4, 3],
        'store_forward': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        'vendor': [2, 2, 2, 1, 1, 2, 2, 2, 2, 2, 1, 2, 1, 2, 2, 2, 1, 1, 2, 2]
    }


    # Save the data
    df = pd.DataFrame(data)
    df.to_parquet(os.path.join(test_data, "test.parquet"))

    # Split the data into inputs and outputs
    y_test = df[TARGET_COL]
    X_test = df[NUMERIC_COLS + CAT_NOM_COLS + CAT_ORD_COLS]

    # Train a Random Forest Regression Model with the training set
    model = RandomForestRegressor(random_state=0)
    model.fit(X_test, y_test)

    # Save the model
    mlflow.sklearn.save_model(sk_model=model, path=model_input)


    cmd = f"python data-science/src/evaluate/evaluate.py --model_name={model_name} --model_input={model_input} --test_data={test_data} --evaluation_output={evaluation_output} --runner={runner}"
    p = subprocess.Popen(cmd, stdout=subprocess.PIPE, shell=True)
    out, err = p.communicate() 
    result = str(out).split('\\n')
    for lin in result:
        if not lin.startswith('#'):
            print(lin)
    
    assert os.path.exists(os.path.join(evaluation_output, "predictions.csv"))
    assert os.path.exists(os.path.join(evaluation_output, "score.txt"))
    
    print("Train Model Unit Test Completed")

if __name__ == "__main__":
    test_evaluate_model()
