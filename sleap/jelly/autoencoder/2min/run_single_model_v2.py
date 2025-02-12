from run_models import *

with tf.device('/CPU:0'):
    model_save_path = '../models/m7t2.keras'
    prediction_7 = run_model_7(window_size=5, epochs=20, optimizer=Adam(1e-4), model_save_path=model_save_path)
    print(prediction_7.shape)
    np.save('../results/r2/m7t2.npy', prediction_7)