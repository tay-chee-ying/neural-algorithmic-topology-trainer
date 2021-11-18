import tensorflow as tf
from tensorflow.keras.layers import Dense, Flatten, Conv1D, SeparableConv1D, LSTM, LSTMCell
from tensorflow.keras import Model

class NGPU_cell(Model):
    def __init__(self,h):
        super(NGPU_cell,self).__init__()
        self.h = h
        self.u_forget_conv = Conv1D(h,3,padding = "same")
        self.z_reset_conv = Conv1D(h,3,padding = "same")
        self.candidate = Conv1D(h,3,padding = "same")
        return
    def call(self,hidden_state):
        u_forget = tf.nn.sigmoid(self.u_forget_conv(hidden_state))
        z_reset = tf.nn.sigmoid(self.z_reset_conv(hidden_state))
        y = u_forget*hidden_state + (1 - u_forget)*tf.nn.tanh(self.candidate(z_reset*hidden_state))
        return y

class PW_NGPU_cell(Model):
    def __init__(self,h):
        super(PW_NGPU_cell,self).__init__()
        self.h = h
        self.u_forget_conv = SeparableConv1D(h,3,padding = "same")
        self.z_reset_conv = SeparableConv1D(h,3,padding = "same")
        self.candidate = SeparableConv1D(h,3,padding = "same")
        return
    def call(self,hidden_state):
        u_forget = tf.nn.sigmoid(self.u_forget_conv(hidden_state))
        z_reset = tf.nn.sigmoid(self.z_reset_conv(hidden_state))
        y = u_forget*hidden_state + (1 - u_forget)*tf.nn.tanh(self.candidate(z_reset*hidden_state))
        return y

class NGPU(Model):
    def __init__(self,h,o):
        super(NGPU,self).__init__()
        self.h = h
        self.o = o
        #self.cell = PW_NGPU_cell(self.h)
        self.cell = NGPU_cell(self.h)
        self.h_lin = Dense(self.h,activation = "tanh")
        self.o_lin = Dense(self.o,activation = None)
        return
    def call(self,input_sequence,dropout = 0.0):
        hidden = self.h_lin(input_sequence)
        for c in range(input_sequence.shape[1]):
            hidden = self.cell(hidden)
            hidden = tf.nn.dropout(hidden,dropout)*(1-dropout)
        o = self.o_lin(hidden)
        return o