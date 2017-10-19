from keras import backend as K
from keras.models import Model
from keras.layers import (BatchNormalization, Conv1D, Dense, Input,
    TimeDistributed, Activation, Bidirectional, SimpleRNN, GRU, LSTM, Concatenate)

def simple_rnn_model(input_dim, output_dim=29):
    """ Build a recurrent network for speech
    """
    # Main acoustic input
    input_data = Input(name='the_input', shape=(None, input_dim))
    # Add recurrent layer
    simp_rnn = GRU(output_dim, return_sequences=True,
                 implementation=2, name='rnn')(input_data)
    # Add softmax activation layer
    y_pred = Activation('softmax', name='softmax')(simp_rnn)
    # Specify the model
    model = Model(inputs=input_data, outputs=y_pred)
    model.output_length = lambda x: x
    print(model.summary())
    return model

def rnn_model(input_dim, units, activation, output_dim=29):
    """ Build a recurrent network for speech
    """
    # Main acoustic input
    input_data = Input(name='the_input', shape=(None, input_dim))
    # Add recurrent layer
    simp_rnn = GRU(units, activation=activation,
        return_sequences=True, implementation=2, name='rnn')(input_data)
    # TODO: Add batch normalization
    bn_rnn = BatchNormalization()(simp_rnn)
    # TODO: Add a TimeDistributed(Dense(output_dim)) layer
    time_dense = TimeDistributed(Dense(output_dim))(bn_rnn)
    # Add softmax activation layer
    y_pred = Activation('softmax', name='softmax')(time_dense)
    # Specify the model
    model = Model(inputs=input_data, outputs=y_pred)
    model.output_length = lambda x: x
    print(model.summary())
    return model


def cnn_rnn_model(input_dim, filters, kernel_size, conv_stride,
    conv_border_mode, units, output_dim=29):
    """ Build a recurrent + convolutional network for speech
    """
    # Main acoustic input
    input_data = Input(name='the_input', shape=(None, input_dim))
    # Add convolutional layer
    conv_1d = Conv1D(filters, kernel_size,
                     strides=conv_stride,
                     padding=conv_border_mode,
                     activation='relu',
                     name='conv1d')(input_data)
    # Add batch normalization
    bn_cnn = BatchNormalization(name='bn_conv_1d')(conv_1d)
    # Add a recurrent layer
    simp_rnn = SimpleRNN(units, activation='relu',
        return_sequences=True, implementation=2, name='rnn')(bn_cnn)
    # TODO: Add batch normalization
    bn_rnn = BatchNormalization(name='bn_rnn')(simp_rnn)
    # TODO: Add a TimeDistributed(Dense(output_dim)) layer
    time_dense = TimeDistributed(Dense(output_dim), name='time_dense')(bn_rnn)
    # Add softmax activation layer
    y_pred = Activation('softmax', name='softmax')(time_dense)
    # Specify the model
    model = Model(inputs=input_data, outputs=y_pred)
    model.output_length = lambda x: cnn_output_length(
        x, kernel_size, conv_border_mode, conv_stride)
    print(model.summary())
    return model

def cnn_output_length(input_length, filter_size, border_mode, stride,
                       dilation=1):
    """ Compute the length of the output sequence after 1D convolution along
        time. Note that this function is in line with the function used in
        Convolution1D class from Keras.
    Params:
        input_length (int): Length of the input sequence.
        filter_size (int): Width of the convolution kernel.
        border_mode (str): Only support `same` or `valid`.
        stride (int): Stride size used in 1D convolution.
        dilation (int)
    """
    if input_length is None:
        return None
    assert border_mode in {'same', 'valid'}
    dilated_filter_size = filter_size + (filter_size - 1) * (dilation - 1)
    if border_mode == 'same':
        output_length = input_length
    elif border_mode == 'valid':
        output_length = input_length - dilated_filter_size + 1
    return (output_length + stride - 1) // stride

def deep_rnn_model(input_dim, units, recur_layers, output_dim=29):
    """ Build a deep recurrent network for speech
    """
    # Main acoustic input
    input_data = Input(name='the_input', shape=(None, input_dim))
    # TODO: Add recurrent layers, each with batch normalization
    rec_layers = list()
    batch_layers = list()

    rec_layers.append(GRU(units, activation='relu',
        return_sequences=True, implementation=2, name='rnn_0')(input_data))
    batch_layers.append(BatchNormalization(name='bn_0')(rec_layers[0]))

    for i in range(0,recur_layers-1):
        rec_layers.append(GRU(units, activation='relu',
            return_sequences=True, implementation=2, name=('rnn_'+str(i+1)))(batch_layers[i]))
        batch_layers.append(BatchNormalization(name=('bn_'+str(i+1)))(rec_layers[i+1]))

    # TODO: Add a TimeDistributed(Dense(output_dim)) layer
    time_dense = TimeDistributed(Dense(output_dim), name='time_dense')(batch_layers[recur_layers-1])
    # Add softmax activation layer
    y_pred = Activation('softmax', name='softmax')(time_dense)
    # Specify the model
    model = Model(inputs=input_data, outputs=y_pred)
    model.output_length = lambda x: x
    print(model.summary())
    return model

def bidirectional_rnn_model(input_dim, units, output_dim=29):
    """ Build a bidirectional recurrent network for speech
    """
    # Main acoustic input
    input_data = Input(name='the_input', shape=(None, input_dim))
    # TODO: Add bidirectional recurrent layer
    bidir_rnn = Bidirectional(GRU(units, activation='relu',
        return_sequences=True, implementation=2, name='bidir_rnn'), merge_mode='concat')(input_data)
    # TODO: Add a TimeDistributed(Dense(output_dim)) layer
    time_dense = TimeDistributed(Dense(output_dim),name='time_dense')(bidir_rnn)
    # Add softmax activation layer
    y_pred = Activation('softmax', name='softmax')(time_dense)
    # Specify the model
    model = Model(inputs=input_data, outputs=y_pred)
    model.output_length = lambda x: x
    print(model.summary())
    return model

def final_model(input_dim, units_s, units_d, output_dim=29):
    """ Build a deep network for speech
    """
    # Main acoustic input
    input_data = Input(name='the_input', shape=(None, input_dim))

    # 3x 1d convolution layers, each with batch normalization at the end
    # Inspired by Wavenet paper, they are dilated causal convolutions
    # 128 filter of width 4 and dilation rate 1, so simple convolutions
    # 64 filters of width 4 and dilation rate 4, so effectively spanning 16 input vectors
    # 64 filters of width 4 and dilation rate 16, so effectively spanning 64 input vectors
    # This way, conv1d_3 spans 64 input vectors, having much fewer parameters
    # compared to standard conv1d with 64 unit kernel
    conv_1d_1 = Conv1D(128, 4, dilation_rate=1, padding='causal',
        activation='relu', name='conv1d_1')(input_data)
    bn_cnn_1 = BatchNormalization(name='bn_cnn_1')(conv_1d_1)

    conv_1d_2 = Conv1D(64, 4, dilation_rate=4, padding='causal',
        activation='relu', name='conv1d_2')(bn_cnn_1)
    bn_cnn_2 = BatchNormalization(name='bn_cnn_2')(conv_1d_2)

    conv_1d_3 = Conv1D(64, 4, dilation_rate=16, padding='causal',
        activation='relu', name='conv1d_3')(bn_cnn_2)
    bn_cnn_3 = BatchNormalization(name='bn_cnn_3')(conv_1d_3)

    # Stacking all 3 convolution layers, so that the GRU cells can use
    # all filters (of different resolution)
    bn_concat = Concatenate()([bn_cnn_1, bn_cnn_2, bn_cnn_3])

    # 2x Unidirectional GRU cells, with units defined in call to function
    # and 25% dropout rate both on input and on hidden state passed between
    # cells; also, each layer ended with batch normalization
    rnn_gru_1 = GRU(units_s, activation='relu',
        return_sequences=True, implementation=2,
        dropout=0.25, recurrent_dropout=0.25, name='rnn_gru_1')(bn_concat)
    rnn_bn_1 = BatchNormalization(name='rnn_bn_1')(rnn_gru_1)

    rnn_gru_2 = GRU(units_d, activation='relu',
        return_sequences=True, implementation=2,
        dropout=0.25, recurrent_dropout=0.25, name='rnn_gru_2')(rnn_bn_1)
    rnn_bn_2 = BatchNormalization(name='rnn_bn_2')(rnn_gru_2)

    # Single time dense layer, mapping from 256 units from GRU above
    # onto the 29 unit vectors (alphabet defined earlier)
    time_dense = TimeDistributed(Dense(output_dim),name='time_dense')(rnn_bn_2)

    # Softmax activation as with the previous models
    y_pred = Activation('softmax', name='softmax')(time_dense)
    # Specify the model
    model = Model(inputs=input_data, outputs=y_pred)

    # Because we use conv1d with causal padding, the output length of the
    # convolution is the same as input, hence model output lenght will
    # be just x, no need to use helper function
    model.output_length = lambda x: x
    print(model.summary())
    return model
