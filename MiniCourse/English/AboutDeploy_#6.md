# Mini Course on AI & Machine Learning

## Lesson 6: Make Your Model Come Alive! (Save, Load & Use It in the Real World)

Welcome to Lesson 6 — you’re doing amazing!

You can now train solid models, add features, compare Linear Regression vs Random Forest, and pick the winner. But if your model only lives inside a Jupyter notebook that you run once and close… it’s like baking the world’s best cake and then eating it alone in the dark.

Today we’re going to make your model truly alive so anyone (or any app) can use it — on your laptop, on a phone, on a website, or even on a server somewhere in the cloud. We’ll learn how to save the trained model, load it back instantly, and make predictions in milliseconds without retraining. Let’s go!

---

### Why Save a Model?

Imagine you just spent 3 hours training a big Random Forest on thousands of rows. Tomorrow your friend wants to use the same model… do you really want to train it again from scratch? Nope!

Saving a model = saving the “brain” that already learned everything. Train once → use forever (or at least until you build a better one).

---

### The Two Most Popular Tools in Python

| Tool      | Pros                                      | Best for                              |
|-----------|-------------------------------------------|---------------------------------------|
| `joblib`  | Faster & more efficient with big models   | Official recommendation from scikit-learn |
| `pickle`  | Built-in, super simple                    | Small-to-medium models                |

We’ll use joblib because it’s blazing fast with numpy arrays (which scikit-learn loves).

---

### Hands-on: Save & Load the House Price Model

We continue with the same (slightly bigger) dataset from Lesson 5.

```python
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score
import joblib   # ← new friend today!

# Slightly bigger dataset (same as before + a few rows)
data = {
    'Size': [1000, 1500, 1200, 1800, 900, 1400, 1600, 1100, 2000, 1300],
    'Bedrooms': [2, 3, 3, 4, 2, 3, 4, 2, 5, 3],
    'City': [1, 0, 1, 0, 1, 0, 0, 1, 1, 0],        # 1 = downtown, 0 = suburbs
    'Price': [200000, 250000, 220000, 300000, 180000, 240000,
              280000, 210000, 350000, 230000]
}
df = pd.DataFrame(data)

X = df[['Size', 'Bedrooms', 'City']]
y = df['Price']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Train the model (final) model
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Quick sanity check
pred = model.predict(X_test)
print(f"MAE: ${mean_absolute_error(y_test, pred):,.0f}")
print(f"R²: {r2_score(y_test, pred):.2f}")
```

#### 1. Save the Model to Disk

```python
# Save it — you only do this once!
joblib.dump(model, 'best_house_price_model.joblib')

print("Model saved successfully!")
```

You’ll now see a file called `best_house_price_model.joblib` in your folder.

#### 2. Load It Back (Even Tomorrow, Even on Another Computer)

```python
# Pretend we restart everything — delete the model from memory
del model

# Load it again in a split second
model = joblib.load('best_house_price_model.joblib')

print("Model loaded — ready to predict!")
```

#### 3. Make Predictions on Brand-New Houses (No Training Needed!)

```python
# New houses that the model has never seen
new_houses = pd.DataFrame({
    'Size': [1350, 2000, 950],
    'Bedrooms': [3, 5, 2],
    'City': [1, 0, 1]        # 1 = city center
})

predictions = model.predict(new_houses)

for i, price in enumerate(predictions):
    print(f"House {i+1} → Estimated price: ${price:,.0f}")
```

Magic! Instant predictions, zero waiting.

---

### Bonus: Clean Prediction Function (Ready to Share with Anyone)

```python
def predict_house_price(size, bedrooms, city_center=True):
    model = joblib.load('best_house_price_model.joblib')
    city = 1 if city_center else 0
    
    df_new = pd.DataFrame({
        'Size': [size],
        'Bedrooms': [bedrooms],
        'City': [city]
    })
    
    price = model.predict(df_new)[0]
    return f"Estimated price: ${price:,.0f}"

# Try it!
print(predict_house_price(1700, 4, city_center=True))
print(predict_house_price(1100, 2, city_center=False))
```

Now even your non-tech friends can use it!

---

### Want It Even Cooler? Deploy It!

A few super-popular next steps (we’ll go deep in the bonus lesson):

- Streamlit → 10-minute web app  
- FastAPI / Flask → REST API that anyone can call  
- TensorFlow Lite or ONNX → run on Android/iOS  
- Hugging Face, Vercel, Render → free hosting in seconds

---

### Common Mistakes & Quick Fixes

| Problem                            | Fix                                                                 |
|------------------------------------|---------------------------------------------------------------------|
| Saved file is huge (>500 MB)       | Use joblib + lower n_estimators, or try pickle with compression   |
| “ModuleNotFoundError” on another PC| Make sure scikit-learn version is the same (save it in requirements.txt) |
| Forgot to save the scaler/encoder  | Save every preprocessing object with joblib too!                    |

---

### Key Takeaways – Lesson 6

- A trained model can (and should!) be saved with joblib or pickle.  
- Once saved, you can load and predict instantly — no retraining ever again.  
- You can wrap it in a clean function or even turn it into a web/mobile app.  
- Your model is now production-ready!

---

### Challenges – Try It Yourself!

1. Save both a LinearRegression and a RandomForest model — compare file sizes.  
2. Write a function that asks for input using input() in the terminal.  
3. (Advanced) Build a tiny Streamlit app (starter code below):

```python
# app.py
import streamlit as st
import joblib
import pandas as pd

model = joblib.load("best_house_price_model.joblib")

st.title("House Price Predictor")
size = st.slider("Size (sqft)", 500, 3500, 1500)
beds = st.slider("Bedrooms", 1, 7, 3)
city = st.selectbox("Location", ["City Center", "Suburbs"])

city_val = 1 if city == "City Center" else 0
price = model.predict([[size, beds, city_val]])[0]

st.success(f"Estimated Price: **${price:,.0f}**")
```

Run with: `streamlit run app.py`

---

### What’s Next?

Lesson 7 : “Mini Project – Build a Complete House Price App from A to Z”  
+ Bonus: deploy it for free so the whole world can try your model with just one link!

Keep coding and see you in Lesson 7!