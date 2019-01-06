import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from PIL import Image, ImageDraw, ImageFont


# Create your df here:
dataframe = pd.read_csv("profiles.csv")
# Columns of interest.
columns = ['age', 'body_type', 'diet', 'job', 'education', 'sex']
# Create a copy of the dataframe with just the columns of interest.
profiles = dataframe[columns].copy()
# Drop the NaNs.
profiles = profiles.dropna()
profiles = profiles.query('age < 80')
profiles = profiles.query('body_type != ["rather not say"]')

# Convert the 'sex' column to a numerical value: {Female: 0, Male: 1}.
profiles['sex_code'] = profiles['sex'].astype("category").cat.codes
y = profiles.sex_code
y = y.ravel() # change it from a 1D array to a 2D array.
print()
print(profiles[['sex', 'sex_code']][2:8])
print()
ste

# Create a PNG file of the dataframe describe() results.
##txt = dataframe.describe().to_latex()
img = Image.new('RGB', (500, 500), color=(255, 255, 255))
d = ImageDraw.Draw(img)
d.text((5, 5), dataframe.describe().to_string(), fill=(0,0,0))
img.save('df_describe.png')

# Create a PNG file of 'body_type' value counts.
series = profiles.body_type.value_counts()
output = ''
for i, s in enumerate(series):
    #print("{0:18}{1:4}".format(series.index[i], s))
    output += "{0:18}{1:4}\n".format(series.index[i], s)
output += "Name: {}, dtype: {}".format(series.name, series.dtype)
img = Image.new('RGB', (250, 250), color=(255, 255, 255))
d = ImageDraw.Draw(img)
d.text((10, 10), output, fill=(0,0,0))
img.save('profiles_body_type_value_counts.png')

# Create a PNG file of the profiles is not counts.
img = Image.new('RGB', (250, 250), color=(255, 255, 255))
d = ImageDraw.Draw(img)
output = ''
for col in profiles.columns:
    txt = "{0:10} {1:4}\n".format(col, profiles[col].isnull().values.sum())
    output += txt
d.text((5, 5), output, fill=(0,0,0))
img.save('profiles_columns_null_counts.png')

# Create a PNG file of the profiles head() results.
img = Image.new('RGB', (1000, 250), color=(255, 255, 255))
d = ImageDraw.Draw(img)
d.text((5, 5), profiles.head().to_string(), fill=(0,0,0))
img.save('profiles_head.png')

# Create a PNG file of the body_type and diet scatter plot.
fig = plt.figure()
plt.title("Body Type vs. Diet")
plt.scatter(profiles.body_type, profiles.diet, alpha=0.2)
fig.tight_layout()
plt.xticks(rotation=45)
plt.savefig("body_type_vs_diet.png")
plt.show()

# Create a PNG file of the age and body type bar plot
plt.title("Age vs. Body Type")
fig.tight_layout()
plt.bar(profiles.age, profiles.body_type, alpha=0.2)
plt.savefig("age_vs_body_type.png")
plt.show()


