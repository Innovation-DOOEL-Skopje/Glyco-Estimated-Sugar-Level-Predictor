# %%
import tensorflow as tf
from tensorflow.keras.models import load_model
import numpy as np
from tensorflow.keras.callbacks import CSVLogger
import kerastuner
import autokeras as ak
import sys
csv_logger = CSVLogger("Log.csv", append=True, separator=';')

# %%
def rmse(y_true, y_pred):
    from keras import backend
    return backend.sqrt(backend.mean(backend.square(y_pred - y_true), axis=-1))

def mse(y_true, y_pred):
    from keras import backend
    return backend.mean(backend.square(y_pred - y_true), axis=-1)

def r_square(y_true, y_pred):
    from keras import backend as K
    SS_res =  K.sum(K.square(y_true - y_pred)) 
    SS_tot = K.sum(K.square(y_true - K.mean(y_true))) 
    return (1 - SS_res/(SS_tot + K.epsilon()))

def r_square_loss(y_true, y_pred):
    from keras import backend as K
    SS_res =  np.sum(np.square(y_true - y_pred)) 
    SS_tot = np.sum(np.square(y_true - np.mean(y_true))) 
    return 1 - ( 1 - SS_res/(SS_tot + K.epsilon()))

Early_Stopping = tf.keras.callbacks.EarlyStopping(
    monitor='val_loss', min_delta=0, patience=15, verbose=0,
    mode='min', baseline=None, restore_best_weights=False
)



# %%

train_x = np.load("Data\\train_x.npy", allow_pickle=True)
test_x = np.load("Data\\test_x.npy", allow_pickle=True)
train_y = np.load("Data\\train_y.npy", allow_pickle=True)
test_y = np.load("Data\\test_y.npy", allow_pickle=True)
# %%

reg = ak.StructuredDataRegressor(
    project_name = "Regressor",
    objective=kerastuner.Objective('mean_squared_error', direction='max'),
    metrics=["mean_squared_error"],
    overwrite=True,
    max_trials=11,
    )

# %%

reg.fit(
    train_x,
    train_y,
    epochs=11,
    verbose=2,
    callbacks=[Early_Stopping, csv_logger],
    )

# %%
print("clf export called")
reg_best_model = reg.export_model()
print("\n================================= Regressor Evaluattion ==============================\n")
print(reg.evaluate(test_x, test_y))
print("\n=======================================================================================\n")
# %%
try:
    try:
        reg_best_model.save("Modal", save_format="tf")
    except:
        reg_best_model.save("Modal\\Modal.h5")
except:
    pass

# %%

from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score



loaded_model = load_model("Modal")
y_pred = loaded_model.predict(test_x)

rmse = mean_squared_error(test_y, y_pred, squared=False)
mse = mean_squared_error(test_y, y_pred, squared=True)
r_squared_loss = r_square_loss(test_y, y_pred)
r_squared_score = r2_score(test_y, y_pred)
print("rmse -> " + str(mean_squared_error(test_y, y_pred, squared=False)))
print("mse -> " + str(mean_squared_error(test_y, y_pred, squared=True)))
print("r_squared_loss" + str(r_square_loss(test_y, y_pred)))
print("r_squared_score" + str(r2_score(test_y, y_pred)))

f = open("Metrics_Final.txt", "a")
f.write(str(rmse) + "|" + str(mse) + "|" + str(r_squared_loss) + "|" + str(r_squared_score) + "\n")
f.close()