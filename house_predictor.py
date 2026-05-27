import streamlit as st
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

        if 'id' in self.df.columns:
            self.df.drop(columns=['id'], inplace=True)

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
        "House Price India.csv"
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
    
with open("house_price_model.pkl", "rb") as file:
    model = pickle.load(file)


st.set_page_config(
    page_title="House Price Predictor",
    page_icon="🏠",
    layout="centered"
)

st.markdown("""
    <style>
    /* Custom header styling */
    .header-main {
        background: linear-gradient(135deg, #2A63AB 30%, #A15A88 100%);
        color: white;
        padding: 0.5rem 7rem;
        border-radius: 15px;
        margin-bottom: 2rem;
        box-shadow: 0 8px 16px rgba(46, 134, 171, 0.2);
        animation: slideDown 0.6s ease-out;
    }
    
    .header-main h1 {
        margin: 0;
        font-size: 2.5rem;
        font-weight: 800;
        text-shadow: 2px 2px 4px rgba(0, 0, 0, 0.2);
        letter-spacing: -1px;
    }
    
    /* Global Styling */
    body {
        background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
    }
    
    /* Input Section Styling */
    .stSelectbox, .stNumberInput, .stSlider, .stTextInput {
        margin-bottom: 1.5rem;
    }
    
    .stSelectbox > div > div,
    .stNumberInput > div > div,
    .stSlider > div > div,
    .stTextInput > div > div {
        border-radius: 10px;
        background: linear-gradient(135deg, rgba(42, 99, 171, 0.05) 0%, rgba(161, 90, 136, 0.05) 100%);
        border: 2px solid rgba(42, 99, 171, 0.1);
        transition: all 0.3s ease;
    }
    
    .stSelectbox > div > div:hover,
    .stNumberInput > div > div:hover,
    .stSlider > div > div:hover,
    .stTextInput > div > div:hover {
        border: 2px solid rgba(42, 99, 171, 0.3);
        box-shadow: 0 4px 12px rgba(42, 99, 171, 0.1);
    }
    
    /* Button Styling */
    .stButton > button {
        background: linear-gradient(135deg, #2A63AB 0%, #A15A88 100%);
        color: white;
        border: none;
        padding: 0.75rem 2.5rem;
        border-radius: 10px;
        font-weight: 600;
        font-size: 1rem;
        cursor: pointer;
        transition: all 0.3s ease;
        box-shadow: 0 4px 15px rgba(42, 99, 171, 0.25);
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    
    .stButton > button:hover {
        transform: translateY(-3px);
        box-shadow: 0 8px 25px rgba(42, 99, 171, 0.35);
    }
    
    .stButton > button:active {
        transform: translateY(-1px);
    }
    
    /* Success Message Styling */
    .stSuccess {
        background: linear-gradient(135deg, #10b981 0%, #059669 100%);
        border-radius: 10px;
        padding: 1.5rem;
        border-left: 5px solid #059669;
        box-shadow: 0 4px 12px rgba(16, 185, 129, 0.2);
        animation: slideUp 0.5s ease-out;
    }
    
    .stSuccess > div {
        color: white;
        font-weight: 600;
        font-size: 1.1rem;
    }
    
    /* Error Message Styling */
    .stError {
        background: linear-gradient(135deg, #ef4444 0%, #dc2626 100%);
        border-radius: 10px;
        padding: 1.5rem;
        border-left: 5px solid #dc2626;
        box-shadow: 0 4px 12px rgba(239, 68, 68, 0.2);
        animation: slideUp 0.5s ease-out;
    }
    
    .stError > div {
        color: white;
        font-weight: 600;
        font-size: 1.1rem;
    }
    
    /* Subheader Styling */
    .stMarkdown h2 {
        color: #2A63AB;
        font-weight: 700;
        margin-top: 1.5rem;
        margin-bottom: 1rem;
        font-size: 1.8rem;
    }

    
    /* Label Styling */
    .stSelectbox > label,
    .stNumberInput > label,
    .stSlider > label,
    .stTextInput > label {
        font-weight: 600;
        color: #2A63AB;
        font-size: 1rem;
        margin-bottom: 0.5rem;
    }
    
    /* Container Padding */
    .main {
        padding: 2rem 1rem;
    }
    
    /* Animations */
    @keyframes slideDown {
        from {
            opacity: 0;
            transform: translateY(-30px);
        }
        to {
            opacity: 1;
            transform: translateY(0);
        }
    }
    
    @keyframes slideUp {
        from {
            opacity: 0;
            transform: translateY(30px);
        }
        to {
            opacity: 1;
            transform: translateY(0);
        }
    }
    
    /* Responsive Adjustments */
    @media (max-width: 768px) {
        .header-main {
            padding: 1.5rem 1rem;
        }
        
        .header-main h1 {
            font-size: 2rem;
        }
        
        .stButton > button {
            padding: 0.6rem 2rem;
            font-size: 0.95rem;
        }
    }
    
    /* Slider Track Styling */
    .stSlider [data-testid="stTickBar"] {
        background: linear-gradient(to right, #2A63AB, #A15A88);
        opacity: 0.3;
    }
    
    /* Text Input Focus */
    .stTextInput input:focus {
        border-color: #A15A88 !important;
        box-shadow: 0 0 0 3px rgba(161, 90, 136, 0.1) !important;
    }
    
    /* Number Input Focus */
    .stNumberInput input:focus {
        border-color: #A15A88 !important;
        box-shadow: 0 0 0 3px rgba(161, 90, 136, 0.1) !important;
    }
    
    /* Divider Line */
    .stMarkdown hr {
        border: none;
        height: 2px;
        background: linear-gradient(to right, #2A63AB, #A15A88);
        margin: 2rem 0;
    }
    
    /* Smooth Transitions */
    * {
        transition: all 0.3s ease;
    }
    
    /* Price Display Emphasis */
    .stSuccess > div > p {
        font-size: 1.3rem;
        letter-spacing: 1px;
    }
    
    </style>
""", unsafe_allow_html=True)

st.markdown("""
        <div class="header-main">
            <h1>🏠 House Price Predictor</h1>
        </div>
    """, unsafe_allow_html=True)

st.image("OIP (1).webp", width=700)
st.write("")
st.subheader("Predict house prices using Machine Learning")


bedrooms = st.slider(
    "Bedrooms",
    min_value=1,
    max_value=20,
    value=3
)

bathrooms = st.slider(
    "Bathrooms",
    min_value=1,
    max_value=10,
    value=3
)

living_area =  st.number_input(
    "Living Area",
    value=100.0
)

lot_area =  st.number_input(
    "Lot Area",
    value=100.0
)

floors = st.slider(
    "Floors",
    min_value=0,
    max_value=20,
    value=5
)

waterfront = st.slider(
    "Waterfront",
    min_value=0,
    max_value=20,
    value=3
)

views = st.slider(
    "Views",
    min_value=0,
    max_value=20,
    value=4
)

condition = st.slider(
    "Condition",
    min_value=0,
    max_value=20,
    value=3
)

grade = st.slider(
    "Grade",
    min_value=0,
    max_value=20,
    value=5
)

area_excl_basement =  st.number_input(
    "Area of House",
    value=1000.0
)

basement_area =  st.number_input(
    "Area of Basement",
    value=100.0
)

latitude =  st.number_input(
    "Latitude",
    value=100.0
)

longitude =  st.number_input(
    "Longitude",
    value=100.0
)


living_area_renov	=  st.number_input(
    "Living Area Renov",
    value=100.0
)


lot_area_renov	 =  st.number_input(
    "Lot Area Renov	",
    value=100.0
)


if st.button("Predict Price"):

    try:

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
        'lot_area_renov': [lot_area_renov]
    })

        prediction_log = model.predict(input_data)

        prediction = np.exp(prediction_log)

        st.success(
            f"Estimated House Price: {prediction[0]:,.0f}"
        )

    except Exception as e:

        st.error(f"Error: {e}")
