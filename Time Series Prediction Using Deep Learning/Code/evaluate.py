import numpy as np
from tensorflow.keras.models import load_model
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt

X_test = np.load('X_test.npy')
Y_test = np.load('Y_test.npy')
scaler = np.load('scaler.npy', allow_pickle=True).item()

model = load_model('model/saved_model/model.h5')

predictions = model.predict(X_test)
predictions = scaler.inverse_transform(predictions)

Y_test = scaler.inverse_transform([Y_test])
test_score_rmse = np.sqrt(mean_squared_error(Y_test[0], predictions[:,0]))
test_score_mae = mean_absolute_error(Y_test[0], predictions[:,0])
test_score_r2 = r2_score(Y_test[0], predictions[:,0])

print(f'RMSE: {test_score_rmse}')
print(f'MAE: {test_score_mae}')
print(f'R2 Score: {test_score_r2}')

plt.figure(figsize=(12, 6))
plt.plot(Y_test[0], label='True Data')
plt.plot(predictions[:,0], label='Predictions')
plt.xlabel('Time')
plt.ylabel('Sales')
plt.legend()
plt.show()
