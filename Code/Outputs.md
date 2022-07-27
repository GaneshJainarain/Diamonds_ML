# Step 1: Getting Familiar with our Data
- We import pandas and read in the dataset
```python
df = pd.read_csv("/Users/richeyjay/Desktop/Diamonds_ML/venv/Code/diamonds.csv", index_col=0)
print(df.head())
```
![First Link to Familiar data](GettingStartedWithOurData.png)
- Notice how we have columns with string values like 'clarity' and 'cut'
- Machine learning uses math so these columns must be converted into numbers 

## We will be using linear regression, so its ideal that our string classifications are linear, meaning they have a meaningful order.
- *NOTE ON LINEAR REGRESSION* : Linear Regression is the supervised Machine Learning model in which the model finds the best fit linear line between the independent and dependent variable i.e it finds the linear relationship between the dependent and independent variable.
- A Linear Regression model’s main aim is to find the best fit linear line and the optimal values of intercept and coefficients such that the error is minimized.

```python
df['cut'].unique()
#array(['Ideal', 'Premium', 'Good', 'Very Good', 'Fair'], dtype=object)
```
```python
cut_class_dict = {"Fair": 1, "Good": 2, "Very Good": 3, "Premium": 4, "Ideal": 5}
```