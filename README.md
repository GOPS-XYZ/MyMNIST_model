# MyMNIST_model
MNIST Classifier using TensorFlow (5 layers)
# MNIST model with 99.32% Test accuracy
#
#     Input Image   (28,28,1)
#    ========================
#    convl     ↓↓  (5,5,1,32)
#    ========================
#    Resultant     (28,28,32)
#    ========================
#    maxpool   ↓↓  (2 X 2)
#    ========================
#    Resultant     (14,14,32)
#    ========================
#    convl     ↓↓  (5,5,1,64)
#    ========================
#    Resultant     (14,14,64)
#    ========================
#    maxpool   ↓↓  (2 X 2)
#    ========================
#    Resultant     (7,7,64)
#    ========================
#    Softmax   ↓↓ Weight(7*7*64,1024)
#    ========================
#    Resultant     (1024 neurons)
#    ========================
#    Softmax   ↓↓ Weight(1024,10)
#    ========================
#    Resultant     (10 neurons)
#    ========================
#
# Leaning Rate: 0.0001
# Batch Size:   50
# Total Training Steps: 20000
