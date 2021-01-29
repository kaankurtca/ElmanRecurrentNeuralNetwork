import numpy as np
import matplotlib.pyplot as plt

class RNN():

    def __init__(self,input_dim,hidden_dim,output_dim,act="tanh"):
        self.act = act # aktivasyon fonksiyonu girdi olarak verilebiliyor.
        self.input_dim = input_dim; self.hidden_dim = hidden_dim; self.output_dim = output_dim
        # giriş boyutu, gizli(içerik) katman boyutu, çıkış boyutu girdi olarak verilir.

        self.Wx = 0.2*np.random.rand(hidden_dim,hidden_dim)-0.1
        self.Wu = 0.2*np.random.rand(hidden_dim,input_dim)-0.1
        self.Wy = 0.2*np.random.rand(output_dim,hidden_dim)-0.1
        # Ağırlık matrisleri oluşturuldu.

    def train(self, X, yd, epochs, lr):
        self.errorVector = np.zeros(epochs)
        for i in range(epochs):
            sumSquaredError = 0  # her epoch'ta hata hesabı yapmak için

            #x = 0.1 * np.ones((X.shape[1]-1,1)).reshape(-1,1)
            for j, inp in enumerate(X):

                x = inp[:-1].reshape(-1,1)
                u = inp[-1].reshape(-1,1) # giriş u[k] için, son sütunu yani e[k] gürültüsünü aldık.

                v = np.dot(self.Wx,x) + np.dot(self.Wu,u)
                x = self.sigmoid(v)
                y = np.dot(self.Wy,x)
                # çıkış oluşturuldu.

                error = yd[j] - y

                self.Wx = self.Wx + lr * np.dot((np.dot(self.Wy.T,error) * self.sigmoid_derivative(x)), x.reshape(1,-1))
                self.Wu = self.Wu + lr * np.dot((np.dot(self.Wy.T,error) * self.sigmoid_derivative(x)), u.reshape(1,-1))
                self.Wy = self.Wy + lr * np.dot(error,x.reshape(1,-1))
                # ağırlıklar öğrenme kuralına göre güncellendi

                squaredError = self.squaredError(yd[j],y)
                sumSquaredError += squaredError

            meanSquaredError = sumSquaredError / X.shape[0]
            self.errorVector[i] = meanSquaredError
            if ((i + 1) % 10 == 0):
                print("Eğitim için Ortalama Kare Hata: {}, iterasyon sayısı: {}".format(meanSquaredError, i + 1))
                # her 10 iterasyonda bir hatamız yazdırılıyor.
                pass
            if (meanSquaredError < 0.002):
                self.last = i # durdurma koşulu
                break
            self.last=i

    def test(self,x_test):


        x = x_test[:-1].reshape(-1, 1)
        u = x_test[-1].reshape(-1, 1)

        v = np.dot(self.Wx, x) + np.dot(self.Wu, u)
        x = self.sigmoid(v)
        y = np.dot(self.Wy, x)
        # çıkış oluşturuldu.

        return y


    def sigmoid(self, x):
        if self.act == "sigmoid":
            y = 1 / (1 + np.exp(-x))
        elif self.act == "tanh":
            y = (np.exp(x) - np.exp(-x)) / (np.exp(x) + np.exp(-x))
        elif self.act == 'linear':
            y = x
        return y

    def sigmoid_derivative(self, x):
        if self.act == "sigmoid":
            y =  x * (1.0 - x)
        elif self.act == "tanh":
            y = 0.5 - x ** 2
        elif self.act == 'linear':
            y = 1

        return y

    def squaredError(self, hedef, cıkıs):
        return np.average(0.5 * (hedef - cıkıs) ** 2)

