from run_models import *

with tf.device('/CPU:0'):
    model_save_path = '../models/model_19.keras'
    prediction_19 = run_model_7(train_slice_only=True, window_size=5, epochs=20, optimizer=Adam(1e-4), model_save_path=model_save_path)
    print(prediction_19.shape)
    np.save('../results/prediction_19.npy', prediction_19)