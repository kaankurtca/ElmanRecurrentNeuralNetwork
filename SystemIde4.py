import numpy as np
from RNN import RNN
import matplotlib.pyplot as plt

# Bu kodda gizli katmana y[k-1], y[k-2], y[k-3] ve y[k-4] eklenmiştir.


mu, sigma = 0, 0.2 # mean and standard deviation
e = np.random.normal(mu, sigma, 150)  # Billing Sistemine eklenecek olan gürültü vektörü
y=np.zeros(150)
y[:4]=0.1*np.random.rand(4)              # Billing Sistemiminin ilk iki değerleini (y[0],y[1]) başlangıçta sıfır olarak ayarlandı.

for k in range(4,150):
    y[k] = (0.8-(0.5*np.exp(-y[k-1]**2)))*y[k-1]-(0.3+(0.9*np.exp(-y[k-1]**2)))*y[k-2] + (0.1*np.sin(np.pi*y[k-1]))+e[k]
    # Sistemin formülüne göre çıkışlar oluşturuldu.

firstFeatureTrain = y[:120].reshape(-1, 1)          # y[k-4]
secondFeatureTrain = y[1:121].reshape(-1, 1)        # y[k-3]
thirdFeatureTrain = y[2:122].reshape(-1, 1)         # y[k-2]
forthFeatureTrain = y[3:123].reshape(-1, 1)         # y[k-1]
fifthFeatureTrain = e[4:124].reshape(-1, 1)         # e[k]
# Geçmiş değerler ve giriş e[k] oluşturuldu.


temp1=np.concatenate([firstFeatureTrain, secondFeatureTrain], axis=1)
temp2=np.concatenate([temp1, thirdFeatureTrain], axis=1)
temp3=np.concatenate([temp2, forthFeatureTrain], axis=1)
x_train=np.concatenate([temp3, fifthFeatureTrain], axis=1) # Eğitim kümesi girdileri oluşturuldu.

y_train = y[4:124].reshape(-1, 1)      # Eğitim kümesi çıktıları oluşturuldu. #y[k]


elman = RNN(input_dim=1,hidden_dim=4,output_dim=1,act='tanh')

elman.train(x_train,y_train,200,0.01)
plt.figure()
plt.plot(range(len(elman.errorVector))[:elman.last],elman.errorVector[:elman.last])
plt.xlabel('epoch number'); plt.ylabel('Mean Squared Error')

firstFeatureTest = y[120:146].reshape(-1, 1)    # y[k-4]
secondFeatureTest = y[121:147].reshape(-1, 1)   # y[k-3]
thirdFeatureTest = y[122:148].reshape(-1, 1)    # y[k-2]
forthFeatureTest = y[123:149].reshape(-1, 1)    # y[k-1]
fifthFeatureTest = e[124:150].reshape(-1, 1)    # e[k]

temp4=np.concatenate([firstFeatureTest, secondFeatureTest], axis=1)
temp5=np.concatenate([temp4,thirdFeatureTest],axis=1)
temp6=np.concatenate([temp5, forthFeatureTest], axis=1)
x_test=np.concatenate([temp6, fifthFeatureTest], axis=1) # Test kümesi girdileri oluşturuldu.

y_test = y[124:150].reshape(-1, 1)                            # Test kümesi çıktıları oluşturuldu. #y[k]


outputs=np.zeros(len(x_test))
for i in range(len(x_test)):
    output = elman.test(x_test[i])
    outputs[i] = output

# r2 score
y_test_mean = (np.mean(y_test)*np.ones(len(y_test))).reshape(-1,1)
SSres = np.sum((y_test-outputs.reshape((-1,1)))**2)
SStot = np.sum((y_test-y_test_mean)**2)
r2 = 1 - (SSres/SStot)
print(f"R2 Accuracy Score : {r2}")


plt.figure()
plt.scatter(range(len(outputs)),outputs,label="Tahminler",color='r')
plt.plot(range(len(y_test)),y_test,label="Gerçek Değerler",color='b')
plt.legend(loc="lower left")

plt.show()






