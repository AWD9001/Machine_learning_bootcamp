# Import bibliotek
import numpy as np
import pandas as pd
import cv2
from google.colab.patches import cv2_imshow

#Pobranie obrazów
!wget
https: // storage.googleapis.com / esmartdata - courses - files / ml - course / green.jpg
!wget
https: // storage.googleapis.com / esmartdata - courses - files / ml - course / ski.jpg
!wget
https: // storage.googleapis.com / esmartdata - courses - files / ml - course / view.jpg

# Eksploracja
img = cv2.imread('ski.jpg')
print(img.shape)

print(img)

cv2_imshow(img)

# przygotowanie obrazu do modelu
img_data = img.reshape((-1, 3))
img_data = np.float32(img_data)
print(img_data.shape)

df = pd.DataFrame(data=img_data, columns=['dim1', 'dim2', 'dim3'])
df.head(3)
