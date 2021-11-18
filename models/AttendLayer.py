import tensorflow as tf
from tensorflow.keras.layers import Dense, Flatten, Conv1D, SeparableConv1D, LSTM, LSTMCell
from tensorflow.keras import Model

class _cell_attend(Model):
    def __init__(self,h):
        super(_cell_attend,self).__init__()
        self.h = h
        self.f = 7
        self.u_forget_conv = _apply_aconv(self.h,self.f)
        self.z_reset_conv = _apply_aconv(self.h,self.f)
        self.candidate = _apply_aconv(self.h,self.f)
        return
    def call(self,hidden_state,dconv):
        u_forget = tf.nn.sigmoid(self.u_forget_conv(hidden_state,dconv))
        z_reset = tf.nn.sigmoid(self.z_reset_conv(hidden_state,dconv))
        y = u_forget*hidden_state + (1 - u_forget)*tf.nn.tanh(self.candidate(z_reset*hidden_state,dconv))
        return y