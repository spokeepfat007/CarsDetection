'''
    Подготовка датасета, параметры относящиеся к одной картинке трансформируем в отдельный файл и сохраняем
'''
import os

import pandas as pd

from PIL import Image
import  pandas as pd
df_empty = pd.DataFrame({'image_path' : []})
annotations = pd.read_csv("data/train_solution_bounding_boxes (1).csv")
for i in annotations.iloc[:,0]:
    data = annotations.loc[annotations['image'] == i]

    label_path = "data/labels/"+i[:-4]+".txt"
    if not len(df_empty.loc[df_empty['image_path'] == label_path]):
        df_empty = df_empty.append({'image_path' : label_path}, ignore_index=True)
    my_file = open(label_path, "w+")
    s = f"{i}"

    for i in range(len(data)):
        params = data.iloc[i]
        s+=f" {(params[3]+params[1])/2} {(params[4]+params[2])/2} {(params[3]-params[1])} {(params[4]-params[2])}"
    my_file.write(s)
    my_file.close()

df_empty.to_csv("data/train.csv", index=False)