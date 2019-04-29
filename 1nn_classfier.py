from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
import os
import cv2
import numpy as np

gallery = "/media/data2/laixc/AI_DATA/multi_pie_id_crop_test_gallery"
gallery_gray = "/media/data2/laixc/AI_DATA/multi_pie_id_crop_test_gallery_gray"
query = "/media/data2/laixc/AI_DATA/multi_pie_id_crop_test_query"
query_normalized = "/media/data2/laixc/AI_DATA/multi_pie_id_crop_test_query_normalized"
query_normalized_L1 = "/media/data2/laixc/AI_DATA/multi_pie_id_crop_test_query_normalized_L1"
query_normalized_ID = "/media/data2/laixc/AI_DATA/multi_pie_id_crop_test_query_normalized_ID"
query_normalized_ITI = "/media/data2/laixc/AI_DATA/ITI_output"
query_normalized_QI = "/media/data2/laixc/AI_DATA/NPL-QI_output"

def get_xy(path):
    subdirs = os.listdir(path)

    x = []
    y = []

    for dir in subdirs:
        subpath = os.path.join(path, dir)
        for f in os.listdir(subpath):
            y.append(int(dir))
            image = cv2.imread(os.path.join(subpath, f))
            image = cv2.resize(image, (256, 256))
            image_flatten = image.flatten()
            x.append(image_flatten)

    x = np.array(x)
    y = np.array(y)

    print(np.shape(x))
    print(np.shape(y))
    return x, y

# #####################  For rgb  #################################
# x, y = get_xy(gallery)
# one_nn = KNeighborsClassifier(n_neighbors=1)
# one_nn.fit(x,y)
#
# x, y = get_xy(query)
# y_predict = one_nn.predict(x)
# print("raw acc", accuracy_score(y, y_predict))
#
# x, y = get_xy(query_normalized)
# y_predict = one_nn.predict(x)
# print("starGan-normalized acc", accuracy_score(y, y_predict))
#
# x, y = get_xy(query_normalized_L1)
# y_predict = one_nn.predict(x)
# print("starGan-L1-normalized acc", accuracy_score(y, y_predict))
#
# x, y = get_xy(query_normalized_ID)
# y_predict = one_nn.predict(x)
# print("starGan-L1-ID-normalized acc", accuracy_score(y, y_predict))


# ########################  For gray  #############################
x, y = get_xy(gallery_gray)
one_nn = KNeighborsClassifier(n_neighbors=1)
one_nn.fit(x,y)

x, y = get_xy(query_normalized_ITI)
y_predict = one_nn.predict(x)
print("ITI acc", accuracy_score(y, y_predict))

x, y = get_xy(query_normalized_QI)
y_predict = one_nn.predict(x)
print("QI acc", accuracy_score(y, y_predict))
