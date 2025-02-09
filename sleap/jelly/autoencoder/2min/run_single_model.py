from run_models import *

prediction_18 = run_model_7(train_slice_only=True, window_size=5, epochs=50)
print(prediction_18.shape)
np.save('../results/prediction_18.npy', prediction_18)