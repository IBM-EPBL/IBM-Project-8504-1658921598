{
  "cells": [
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "ynUOT9Yc6fQk",
        "outputId": "df1a945a-f917-4156-8b99-18c07da26956"
      },
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "Mounted at /content/drive\n"
          ]
        }
      ],
      "source": [
        "from google.colab import drive\n",
        "drive.mount('/content/drive')"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "NhP7V93wMVho",
        "outputId": "35b72ad9-1b52-42f9-e2d5-7912c02d6bd0"
      },
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "\u001b[0m\u001b[01;34mdrive\u001b[0m/  \u001b[01;34msample_data\u001b[0m/\n"
          ]
        }
      ],
      "source": [
        "ls"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "_VuPfwOuQYAm",
        "outputId": "1c35e7ef-399f-4dcc-c090-946534fa802f"
      },
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "/content/drive/.shortcut-targets-by-id/1LL5lvl6AsdVwW9LWVu_GXEUCoV7jYm-c/Dataset\n"
          ]
        }
      ],
      "source": [
        "cd//content/drive/MyDrive/Colab Notebooks/Dataset\n"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {
        "id": "kdGnCU-FQz8C",
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "outputId": "0444b204-4ffd-4b90-e1a0-9508410b2e55"
      },
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "IBM_review.pptx  photo-1589820296156-2454bb8a6ad1.jpg  \u001b[0m\u001b[01;34mTRAIN_SET\u001b[0m/\n",
            "nutrition.h5     \u001b[01;34mTEST_SET\u001b[0m/\n"
          ]
        }
      ],
      "source": [
        "ls"
      ]
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "whbZ5Uw35pHw"
      },
      "source": [
        "### Importing Neccessary Libraries"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {
        "id": "60eg6zmo5pHx"
      },
      "outputs": [],
      "source": [
        "import numpy as np#used for numerical analysis\n",
        "import tensorflow #open source used for both ML and DL for computation\n",
        "from tensorflow.keras.models import Sequential #it is a plain stack of layers\n",
        "from tensorflow.keras import layers #A layer consists of a tensor-in tensor-out computation function\n",
        "#Dense layer is the regular deeply connected neural network layer\n",
        "from tensorflow.keras.layers import Dense,Flatten\n",
        "#Faltten-used fot flattening the input or change the dimension\n",
        "from tensorflow.keras.layers import Conv2D,MaxPooling2D,Dropout #Convolutional layer\n",
        "#MaxPooling2D-for downsampling the image\n",
        "from keras.preprocessing.image import ImageDataGenerator\n",
        "\n",
        "\n"
      ]
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "vnVt93M05pH0"
      },
      "source": [
        "### Image Data Agumentation"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {
        "id": "-VLZKCTd5pH1"
      },
      "outputs": [],
      "source": [
        "#setting parameter for Image Data agumentation to the training data\n",
        "train_datagen = ImageDataGenerator(rescale=1./255,shear_range=0.2,zoom_range=0.2,horizontal_flip=True)\n",
        "#Image Data agumentation to the testing data\n",
        "test_datagen=ImageDataGenerator(rescale=1./255)"
      ]
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "kpsHveuq5pH4"
      },
      "source": [
        "### Loading our data and performing data agumentation"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "8Hkc9ffd5pH5",
        "outputId": "e33f9587-33e3-44a0-8841-843d9f55a254"
      },
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "Found 4138 images belonging to 5 classes.\n",
            "Found 929 images belonging to 3 classes.\n"
          ]
        }
      ],
      "source": [
        "#performing data agumentation to train data\n",
        "x_train = train_datagen.flow_from_directory(\n",
        "    r'/content/drive/MyDrive/Colab Notebooks/Dataset/TRAIN_SET',\n",
        "    target_size=(64, 64),batch_size=5,color_mode='rgb',class_mode='sparse')\n",
        "#performing data agumentation to test data\n",
        "x_test = test_datagen.flow_from_directory(\n",
        "    r'/content/drive/MyDrive/Colab Notebooks/Dataset/TEST_SET',\n",
        "    target_size=(64, 64),batch_size=5,color_mode='rgb',class_mode='sparse') "
      ]
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "szwYFmls5pH8",
        "outputId": "b2301196-ebac-4c81-f97f-a4d10d78dec2"
      },
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "{'APPLES': 0, 'BANANA': 1, 'ORANGE': 2, 'PINEAPPLE': 3, 'WATERMELON': 4}\n"
          ]
        }
      ],
      "source": [
        "print(x_train.class_indices)#checking the number of classes"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "8SliKn605pH-",
        "outputId": "61c57cec-a36f-4aa8-f56f-fe426dfc2944"
      },
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "{'APPLES': 0, 'BANANA': 1, 'ORANGE': 2}\n"
          ]
        }
      ],
      "source": [
        "print(x_test.class_indices)#checking the number of classes"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "yWWDoRDw5pIA",
        "outputId": "298cbdd8-3d86-4682-9aa1-c8502f57c3da"
      },
      "outputs": [
        {
          "output_type": "execute_result",
          "data": {
            "text/plain": [
              "Counter({0: 995, 1: 1374, 2: 1019, 3: 275, 4: 475})"
            ]
          },
          "metadata": {},
          "execution_count": 12
        }
      ],
      "source": [
        "from collections import Counter as c\n",
        "c(x_train .labels)"
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "# Initializing the CNN\n",
        "classifier = Sequential()\n",
        "\n",
        "# First convolution layer and pooling\n",
        "classifier.add(Conv2D(32, (3, 3), input_shape=(64, 64, 3), activation='relu'))\n",
        "classifier.add(MaxPooling2D(pool_size=(2, 2)))\n",
        "\n",
        "# Second convolution layer and pooling\n",
        "classifier.add(Conv2D(32, (3, 3), activation='relu'))\n",
        "\n",
        "# input_shape is going to be the pooled feature maps from the previous convolution layer\n",
        "classifier.add(MaxPooling2D(pool_size=(2, 2)))\n",
        "\n",
        "# Flattening the layers\n",
        "classifier.add(Flatten())\n",
        "\n",
        "# Adding a fully connected layer\n",
        "classifier.add(Dense(units=128, activation='relu'))\n",
        "classifier.add(Dense(units=5, activation='softmax')) # softmax for more than 2\n",
        "\n"
      ],
      "metadata": {
        "id": "7U7d0yvp1HEO"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "classifier.summary()#summary of our model"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "WvrA--WP1MFZ",
        "outputId": "0a3474f4-6517-493d-c5d0-ea31c45b4d8e"
      },
      "execution_count": null,
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "Model: \"sequential\"\n",
            "_________________________________________________________________\n",
            " Layer (type)                Output Shape              Param #   \n",
            "=================================================================\n",
            " conv2d (Conv2D)             (None, 62, 62, 32)        896       \n",
            "                                                                 \n",
            " max_pooling2d (MaxPooling2D  (None, 31, 31, 32)       0         \n",
            " )                                                               \n",
            "                                                                 \n",
            " conv2d_1 (Conv2D)           (None, 29, 29, 32)        9248      \n",
            "                                                                 \n",
            " max_pooling2d_1 (MaxPooling  (None, 14, 14, 32)       0         \n",
            " 2D)                                                             \n",
            "                                                                 \n",
            " flatten (Flatten)           (None, 6272)              0         \n",
            "                                                                 \n",
            " dense (Dense)               (None, 128)               802944    \n",
            "                                                                 \n",
            " dense_1 (Dense)             (None, 5)                 645       \n",
            "                                                                 \n",
            "=================================================================\n",
            "Total params: 813,733\n",
            "Trainable params: 813,733\n",
            "Non-trainable params: 0\n",
            "_________________________________________________________________\n"
          ]
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "# Compiling the CNN\n",
        "# categorical_crossentropy for more than 2\n",
        "classifier.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy']) "
      ],
      "metadata": {
        "id": "j6ekSlkp1Nr-"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "classifier.fit_generator(\n",
        "        generator=x_train,steps_per_epoch = len(x_train),\n",
        "        epochs=10, validation_data=x_test,validation_steps = len(x_test))# No of images in test set"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "8YkGfW7e1V4q",
        "outputId": "ca0f5698-b658-4e53-ff8b-6214a6abef2d"
      },
      "execution_count": null,
      "outputs": [
        {
          "output_type": "stream",
          "name": "stderr",
          "text": [
            "/usr/local/lib/python3.7/dist-packages/ipykernel_launcher.py:3: UserWarning: `Model.fit_generator` is deprecated and will be removed in a future version. Please use `Model.fit`, which supports generators.\n",
            "  This is separate from the ipykernel package so we can avoid doing imports until\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "Epoch 1/10\n",
            "828/828 [==============================] - 1189s 1s/step - loss: 0.5894 - accuracy: 0.7748 - val_loss: 0.5930 - val_accuracy: 0.7427\n",
            "Epoch 2/10\n",
            "828/828 [==============================] - 29s 35ms/step - loss: 0.4094 - accuracy: 0.8371 - val_loss: 0.5117 - val_accuracy: 0.8159\n",
            "Epoch 3/10\n",
            "828/828 [==============================] - 27s 33ms/step - loss: 0.3728 - accuracy: 0.8586 - val_loss: 0.3814 - val_accuracy: 0.8558\n",
            "Epoch 4/10\n",
            "828/828 [==============================] - 29s 35ms/step - loss: 0.3465 - accuracy: 0.8664 - val_loss: 0.4036 - val_accuracy: 0.8525\n",
            "Epoch 5/10\n",
            "828/828 [==============================] - 28s 33ms/step - loss: 0.3175 - accuracy: 0.8797 - val_loss: 0.4061 - val_accuracy: 0.8428\n",
            "Epoch 6/10\n",
            "828/828 [==============================] - 29s 35ms/step - loss: 0.3020 - accuracy: 0.8896 - val_loss: 0.3806 - val_accuracy: 0.8558\n",
            "Epoch 7/10\n",
            "828/828 [==============================] - 30s 36ms/step - loss: 0.2848 - accuracy: 0.8888 - val_loss: 0.4778 - val_accuracy: 0.8041\n",
            "Epoch 8/10\n",
            "828/828 [==============================] - 29s 35ms/step - loss: 0.2673 - accuracy: 0.8980 - val_loss: 0.4117 - val_accuracy: 0.8385\n",
            "Epoch 9/10\n",
            "828/828 [==============================] - 31s 37ms/step - loss: 0.2485 - accuracy: 0.9065 - val_loss: 0.3935 - val_accuracy: 0.8611\n",
            "Epoch 10/10\n",
            "828/828 [==============================] - 28s 34ms/step - loss: 0.2399 - accuracy: 0.9120 - val_loss: 0.4292 - val_accuracy: 0.8525\n"
          ]
        },
        {
          "output_type": "execute_result",
          "data": {
            "text/plain": [
              "<keras.callbacks.History at 0x7f40e2afcf10>"
            ]
          },
          "metadata": {},
          "execution_count": 16
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "# Save the model\n",
        "classifier.save('nutrition.h5')"
      ],
      "metadata": {
        "id": "Qe2POBdU7akr"
      },
      "execution_count": 40,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "from tensorflow.keras.models import load_model\n",
        "from tensorflow.keras.preprocessing import image\n",
        "import numpy as np"
      ],
      "metadata": {
        "id": "XnTA5ZZ68Lv3"
      },
      "execution_count": 27,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "img = image.load_img(\"/content/drive/MyDrive/Colab Notebooks/Dataset/TRAIN_SET/APPLES/n07740461_10067.jpg\",target_size= (64,64))#loading of the image\n",
        "img"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/",
          "height": 81
        },
        "id": "rgFMwRuH8Pnp",
        "outputId": "d018abe7-7002-44a3-aa15-7101bd60721c"
      },
      "execution_count": 28,
      "outputs": [
        {
          "output_type": "execute_result",
          "data": {
            "text/plain": [
              "<PIL.Image.Image image mode=RGB size=64x64 at 0x7F40DD08AA50>"
            ],
            "image/png": "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\n"
          },
          "metadata": {},
          "execution_count": 28
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "x=image.img_to_array(img)#conversion image into array"
      ],
      "metadata": {
        "id": "vndVbu0q9O4P"
      },
      "execution_count": 31,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "x"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "1u8piYkF9b9p",
        "outputId": "83c2f78b-c54d-436f-eb3c-2712ae4799e8"
      },
      "execution_count": 32,
      "outputs": [
        {
          "output_type": "execute_result",
          "data": {
            "text/plain": [
              "array([[[255., 255., 255.],\n",
              "        [255., 255., 255.],\n",
              "        [255., 255., 255.],\n",
              "        ...,\n",
              "        [255., 255., 255.],\n",
              "        [255., 255., 255.],\n",
              "        [255., 255., 255.]],\n",
              "\n",
              "       [[255., 255., 255.],\n",
              "        [255., 255., 255.],\n",
              "        [255., 255., 255.],\n",
              "        ...,\n",
              "        [255., 255., 255.],\n",
              "        [255., 255., 255.],\n",
              "        [255., 255., 255.]],\n",
              "\n",
              "       [[255., 255., 255.],\n",
              "        [255., 255., 255.],\n",
              "        [255., 255., 255.],\n",
              "        ...,\n",
              "        [255., 255., 255.],\n",
              "        [255., 255., 255.],\n",
              "        [255., 255., 255.]],\n",
              "\n",
              "       ...,\n",
              "\n",
              "       [[255., 255., 255.],\n",
              "        [255., 255., 255.],\n",
              "        [255., 255., 255.],\n",
              "        ...,\n",
              "        [255., 255., 255.],\n",
              "        [255., 255., 255.],\n",
              "        [255., 255., 255.]],\n",
              "\n",
              "       [[255., 255., 255.],\n",
              "        [255., 255., 255.],\n",
              "        [255., 255., 255.],\n",
              "        ...,\n",
              "        [255., 255., 255.],\n",
              "        [255., 255., 255.],\n",
              "        [255., 255., 255.]],\n",
              "\n",
              "       [[255., 255., 255.],\n",
              "        [255., 255., 255.],\n",
              "        [255., 255., 255.],\n",
              "        ...,\n",
              "        [255., 255., 255.],\n",
              "        [255., 255., 255.],\n",
              "        [255., 255., 255.]]], dtype=float32)"
            ]
          },
          "metadata": {},
          "execution_count": 32
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "x.ndim"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "yaVVqp4F9WTB",
        "outputId": "3d2bdf0a-4754-4112-dae1-2d4b832eca36"
      },
      "execution_count": 30,
      "outputs": [
        {
          "output_type": "execute_result",
          "data": {
            "text/plain": [
              "3"
            ]
          },
          "metadata": {},
          "execution_count": 30
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "x=np.expand_dims(x,axis=0) #expand the dimension"
      ],
      "metadata": {
        "id": "_ksKd6z09WaM"
      },
      "execution_count": 33,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "x.ndim"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "7H5wVKTE9Wgy",
        "outputId": "1086f9fa-4f63-40b6-8d0a-dbae88a4bfc1"
      },
      "execution_count": 34,
      "outputs": [
        {
          "output_type": "execute_result",
          "data": {
            "text/plain": [
              "4"
            ]
          },
          "metadata": {},
          "execution_count": 34
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "pred = classifier.predict(x)"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "UPDDmAKg9Wmh",
        "outputId": "52c35432-7500-4965-a7b1-910301bfb14f"
      },
      "execution_count": 35,
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "1/1 [==============================] - 0s 79ms/step\n"
          ]
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "pred"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "_7ZC16gb9q02",
        "outputId": "599c97d3-5173-4a26-fc93-06c07c1672db"
      },
      "execution_count": 36,
      "outputs": [
        {
          "output_type": "execute_result",
          "data": {
            "text/plain": [
              "array([[1., 0., 0., 0., 0.]], dtype=float32)"
            ]
          },
          "metadata": {},
          "execution_count": 36
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "labels=['APPLES', 'BANANA', 'ORANGE','PINEAPPLE','WATERMELON']\n",
        "labels[np.argmax(pred)]"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/",
          "height": 36
        },
        "id": "F4SkklKo9q9n",
        "outputId": "3708e2cb-81cb-48de-d729-db60097ce171"
      },
      "execution_count": 37,
      "outputs": [
        {
          "output_type": "execute_result",
          "data": {
            "text/plain": [
              "'APPLES'"
            ],
            "application/vnd.google.colaboratory.intrinsic+json": {
              "type": "string"
            }
          },
          "metadata": {},
          "execution_count": 37
        }
      ]
    }
  ],
  "metadata": {
    "colab": {
      "collapsed_sections": [],
      "provenance": []
    },
    "kernelspec": {
      "display_name": "Python 3",
      "language": "python",
      "name": "python3"
    },
    "language_info": {
      "codemirror_mode": {
        "name": "ipython",
        "version": 3
      },
      "file_extension": ".py",
      "mimetype": "text/x-python",
      "name": "python",
      "nbconvert_exporter": "python",
      "pygments_lexer": "ipython3",
      "version": "3.8.3"
    }
  },
  "nbformat": 4,
  "nbformat_minor": 0
}
