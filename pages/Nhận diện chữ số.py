import streamlit as st
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.datasets import mnist
import numpy as np
import random

def create_model():
    model = Sequential([
        Conv2D(20, (5, 5), activation='relu', input_shape=(28, 28, 1)),
        MaxPooling2D((2, 2)),
        Conv2D(50, (5, 5), activation='relu'),
        MaxPooling2D((2, 2)),
        Flatten(),
        Dense(500, activation='relu'),
        Dense(10, activation='softmax')
    ])
    return model

def tao_anh_ngau_nhien(X_test):
    image = np.zeros((10*28, 10*28), np.uint8)
    data = np.zeros((100,28,28,1), np.uint8)

    for i in range(0, 100):
        n = random.randint(0, 9999)
        sample = X_test[n]
        data[i] = X_test[n]
        x = i // 10
        y = i % 10
        image[x*28:(x+1)*28,y*28:(y+1)*28] = sample[:,:,0]    
    return image, data

def main():
    if 'is_load' not in st.session_state:
        # load model
        model = create_model()
        model.compile(loss="categorical_crossentropy", optimizer='adam', metrics=["accuracy"])
        model.load_weights("mnist/digit_weight.h5")
        st.session_state.model = model

        # load data
        (_, _), (X_test, _) = mnist.load_data()
        X_test = X_test.reshape((10000, 28, 28, 1))
        st.session_state.X_test = X_test

        st.session_state.is_load = True
        print('Lần đầu load model và data')
    else:
        print('Đã load model và data rồi')

    if st.button('Tạo ảnh'):
        image, data = tao_anh_ngau_nhien(st.session_state.X_test)
        st.session_state.image = image
        st.session_state.data = data

    if 'image' in st.session_state:
        image = st.session_state.image
        st.image(image)

        if st.button('Nhận dạng'):
            data = st.session_state.data
            data = data / 255.0
            data = data.astype('float32')
            ket_qua = st.session_state.model.predict(data)
            dem = 0
            s = ''
            for x in ket_qua:
                s = s + '%d ' % (np.argmax(x))
                dem = dem + 1
                if (dem % 10 == 0) and (dem < 100):
                    s = s + '\n'
            st.text(s)

if __name__ == '__main__':
    main()
