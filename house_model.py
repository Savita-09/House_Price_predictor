import pandas as pd
import pickle
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_absolute_error, r2_score,mean_squared_error

from sklearn.ensemble import RandomForestRegressor


class HousePriceModel:

    def __init__(self, data_path):

        self.data_path = data_path

        self.df = None

        self.model = None

        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None

    def load_data(self):

        self.df = pd.read_csv(self.data_path)

        print("Dataset Loaded Successfully")
        print(self.df.head())

    # Prepare Data
    def prepare_data(self):

        if 'Unnamed: 0' in self.df.columns:
            self.df.drop(columns=['Unnamed: 0'], inplace=True)

        # Features
        X = self.df[['number of bedrooms', 'number of bathrooms', 'living area',
       'lot area', 'number of floors', 'waterfront present', 'number of views',
       'condition of the house', 'grade of the house',
       'Area of the house(excluding basement)', 'Area of the basement', 'Lattitude',
       'Longitude', 'living_area_renov', 'lot_area_renov']]

        # Target
        y = self.df['Price']
        y_log = np.log(y)
        
        numeric_col = X.select_dtypes(include = ['int64','float64']).columns

        # Preprocessor
        preprocessor = ColumnTransformer(
            transformers=[
                ('scaler', StandardScaler(),numeric_col)

            ],
            remainder='passthrough'
        )

        # Pipeline
        self.model = Pipeline(steps=[

            ('preprocessor', preprocessor),

            ('regressor',RandomForestRegressor(n_estimators=100, bootstrap = True

             ))
        ])

        # Train Test Split
        (
            self.X_train,
            self.X_test,
            self.y_train,
            self.y_test

        ) = train_test_split(
            X,
            y_log,
            test_size=0.2,
            random_state=42
        )

        print("Data Prepared Successfully")

    def train_model(self):

        self.model.fit(self.X_train, self.y_train)

        print("Model Trained Successfully")

    def evaluate_model(self):

        predictions_log = self.model.predict(self.X_test)


        mae = mean_absolute_error(self.y_test, predictions_log)

        mse = mean_squared_error(self.y_test, predictions_log)
        r2 = r2_score(self.y_test, predictions_log)

        print(f"Mean Absolute Error: {mae}")

        print(f"Mean Squared Error : {mse}")

        print(f"R2 Score: {r2}")

    def save_model(self, model_name="house_price_model.pkl"):

        with open(model_name, "wb") as file:

            pickle.dump(self.model, file)

        print(f"Model Saved as {model_name}")

    def predict_price(
        self,
        bedrooms,
        bathrooms,
        living_area,
        lot_area,
        floors,
        waterfront,
        views,
        condition,
        grade,
        area_excl_basement,
        basement_area,
        latitude,
        longitude,
        living_area_renov,
        lot_area_renov):

        input_data = pd.DataFrame({
        'number of bedrooms': [bedrooms],
        'number of bathrooms': [bathrooms],
        'living area': [living_area],
        'lot area': [lot_area],
        'number of floors': [floors],
        'waterfront present': [waterfront],
        'number of views': [views],
        'condition of the house': [condition],
        'grade of the house': [grade],
        'Area of the house(excluding basement)': [area_excl_basement],
        'Area of the basement': [basement_area],
        'Lattitude': [latitude],
        'Longitude': [longitude],
        'living_area_renov': [living_area_renov],
        'lot_area_renov': [lot_area_renov],
    })

        prediction_log = self.model.predict(input_data)

        prediction = np.exp(prediction_log)

        return prediction[0]

if __name__ == "__main__":

    house_model = HousePriceModel(
        "C:\\Users\\savita\\Downloads\\House Price India.csv"
    )

    house_model.load_data()

    house_model.prepare_data()

    house_model.train_model()

    house_model.evaluate_model()

    house_model.save_model()

    predicted_price = house_model.predict_price(
    bedrooms=5,
    bathrooms=4,
    living_area=3000,
    lot_area=5000,
    floors=2,
    waterfront=0,
    views=3,
    condition=4,
    grade=8,
    area_excl_basement=2500,
    basement_area=500,
    latitude=47.5,
    longitude=-122.2,
    living_area_renov=3200,
    lot_area_renov=5200
)
    print(f"Predicted House Price: ₹{predicted_price:,.2f}")