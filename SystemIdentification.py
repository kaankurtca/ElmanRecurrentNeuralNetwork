import numpy as np
from RNN import RNN
import matplotlib.pyplot as plt

mu, sigma = 0, 0.01 # mean and standard deviation
e = np.random.normal(mu, sigma, 150)  # Billing Sistemine eklenecek olan gürültü vektörü
y=np.zeros(150)
y[:2]=0.1*np.random.rand(2)              # Billing Sistemiminin ilk iki değerleini (y[0],y[1]) başlangıçta sıfır olarak ayarlandı.

for k in range(2,150):
    y[k] = (0.8-(0.5*np.exp(-y[k-1]**2)))*y[k-1]-(0.3+(0.9*np.exp(-y[k-1]**2)))*y[k-2] + (0.1*np.sin(np.pi*y[k-1]))+e[k]
    # Sistemin formülüne göre çıkışlar oluşturuldu.

firstFeatureTrain = y[:120].reshape(-1, 1)          # y[k-2]
secondFeatureTrain = y[1:121].reshape(-1, 1)        # y[k-1]
thirdFeatureTrain = e[2:122].reshape(-1, 1)         # e[k]
# Eğitim kümesinin sütunları(feature) oluşturuldu.

temp1=np.concatenate([firstFeatureTrain, secondFeatureTrain], axis=1)
x_train=np.concatenate([temp1, thirdFeatureTrain], axis=1)   # Eğitim kümesi girdileri oluşturuldu.

y_train = y[2:122].reshape(-1, 1)      # Eğitim kümesi çıktıları oluşturuldu. #y[k]


elman = RNN(input_dim=1,hidden_dim=2,output_dim=1,act='sigmoid')

elman.train(x_train,y_train,1000,0.01)

firstFeatureTest = y[120:148].reshape(-1, 1)    # y[k-2]
secondFeatureTest = y[121:149].reshape(-1, 1)   # y[k-1]
thirdFeatureTest = e[122:150].reshape(-1, 1)    # e[k]

temp2=np.concatenate([firstFeatureTest, secondFeatureTest], axis=1)
x_test=np.concatenate([temp2, thirdFeatureTest], axis=1)     # Test kümesi girdileri oluşturuldu.

y_test = y[122:150].reshape(-1, 1)                            # Test kümesi çıktıları oluşturuldu. #y[k]


outputs=np.zeros(len(x_test))

for i in range(len(x_test)):

    output = elman.test(x_test[i])
    outputs[i] = output


plt.figure()
plt.scatter(range(len(outputs)),outputs,label="Tahminler",color='r')
plt.plot(range(len(y_test)),y_test,label="Gerçek Değerler",color='b')
plt.legend(loc="lower left")

plt.show()






