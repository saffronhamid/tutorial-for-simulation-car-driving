from utils import *
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
### Step 1
path = "DataSet"
data = importDataInfo(path)
### Step 2
balanceData(data)

### Step 3
imagesPath, steering = loadData(path,data)
# print(imagesPath)
# print(steering)
### step 4
x_train, x_val, y_train, y_val = train_test_split(imagesPath,steering,test_size = 0.2, random_state = 5)
print('Total Training images: ',len(x_train))
print('Total validation images: ', len(x_val)) 
### step 5
model = createModel()
model.summary()
### step 6
history = model.fit(batchGen(x_train, y_train, 10, 1),steps_per_epoch = 20, epochs = 2,
                        validation_data = batchGen(x_val,y_val,20,0),validation_steps = 20)
### step 6
model.save('model.h5')
print('model saved')

plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.legend(['Training', 'Validation'])
plt.ylim([0, 1])
plt.title('Loss')
plt.xlabel('Epoch')
plt.show()