from src.models.convolution_neural_network import ConvolutionNeuralNetwork
from src.othello.load_othello import load_o

othello_ccn = ConvolutionNeuralNetwork(load_nn_name="reversi2")
# x_shape = (100000, 2, 8, 8)
# othello_ccn.set_last_layer("ms")
# othello_ccn.add_cn(cn=2, filter_num=32, filter_size=3)
# othello_ccn.add_activation()
# othello_ccn.add_cn(filter_num=32, filter_size=2)
# othello_ccn.add_activation()
# othello_ccn.add_affine(x_shape, 64)
# othello_ccn.shape_summary(x_shape)

for i in range(10):
    for i in load_o():
        train_data, label_data = i
        othello_ccn.batch_train(train_data, label_data, epochs=1, save_param_name="reversi2")
