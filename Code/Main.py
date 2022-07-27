import pandas as pd

#reading our dataset into the df variable
#outputing the first 5 entries 
df = pd.read_csv("/Users/richeyjay/Desktop/Diamonds_ML/venv/Code/diamonds.csv", index_col=0)
print(df.head())



print(df['cut'].unique())
#Yields ->
#array(['Ideal', 'Premium', 'Good', 'Very Good', 'Fair'], dtype=object)


