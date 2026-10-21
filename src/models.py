'''
Models: two baselines, XGBoost with quantile output, and the kinematic LSTM.

All models return predictions for the quantiles in QUANTILES (km/h), shape (n, 5).
'''

import numpy as np

QUANTILES=[0.05, 0.25, 0.50, 0.75, 0.95]
WINDOW=48

# Columns that form the hourly sequence for the LSTM (past information only)
SEQ_COLS_BASE=['speed_t0', 'speed_diff1', 'vehicles_t0', 'speed_filled_t0']
SEQ_COLS_NB=[f'nb{i}_{s}' for i in range(1, 7) for s in ('speed_t0', 'diff1')]
SEQ_COLS_WEATHER=['temperature_t0', 'rain_t0', 'snowfall_t0', 'wind_speed_t0']


class Scaler:
    '''Median / IQR scaling, fitted on training rows. Binary columns are left as they are.'''

    def fit(self, X):
        self.cols=list(X.columns)
        binary=X.nunique()<=2
        med=X.median()
        iqr=(X.quantile(0.75)-X.quantile(0.25)).replace(0, 1.0)
        self.center=med.where(~binary, 0.0)
        self.scale=iqr.where(~binary, 1.0)
        return self

    def transform(self, X):
        return ((X[self.cols]-self.center)/self.scale).clip(-10, 10).astype('float32').values


def sort_quantiles(pred):
    return np.sort(pred, axis=1)


# ---------------------------------------------------------------- baselines

def persistence(X):
    '''Speed at the target hour = current speed. Quantiles from training errors are added later.'''
    return X['speed_t0'].values


def seasonal_naive(X):
    '''Speed at the target hour = speed at the same hour last week.'''
    return X['speed_lag168_target'].values


def add_error_quantiles(point_train, y_train, point_eval):
    '''Turn a point forecast into quantiles using the empirical training errors.'''
    err=y_train-point_train
    offsets=np.quantile(err, QUANTILES)
    return point_eval[:, None]+offsets[None, :]


# ---------------------------------------------------------------- XGBoost

def fit_predict_xgb(X_tr, y_tr, X_va, y_va, X_ev, seed):
    import xgboost as xgb
    model=xgb.XGBRegressor(
        objective='reg:quantileerror', quantile_alpha=np.array(QUANTILES),
        n_estimators=2000, learning_rate=0.03, max_depth=6, min_child_weight=5,
        subsample=0.8, colsample_bytree=0.8, reg_lambda=1.0,
        tree_method='hist', early_stopping_rounds=100, random_state=seed,
        device='cuda' if _has_gpu() else 'cpu',
    )
    model.fit(X_tr, y_tr, eval_set=[(X_va, y_va)], verbose=False)
    return sort_quantiles(model.predict(X_ev)), {'best_iteration':int(model.best_iteration)}


def _has_gpu():
    try:
        import subprocess
        return subprocess.run(['nvidia-smi'], capture_output=True).returncode==0
    except Exception:
        return False


# ---------------------------------------------------------------- LSTM

def pinball_loss(y_true, y_pred):
    import tensorflow as tf
    y_pred=tf.sort(y_pred, axis=-1)
    q=tf.constant(QUANTILES, dtype=y_pred.dtype)
    e=y_true-y_pred
    return tf.reduce_mean(tf.maximum(q*e, (q-1.0)*e))


def build_lstm(n_seq, n_ctx):
    '''Kinematic LSTM: output = current speed + learned change, for 5 quantiles.'''
    import tensorflow as tf
    from tensorflow.keras import layers, Model

    seq_in=layers.Input(shape=(WINDOW, n_seq), name='sequence')
    ctx_in=layers.Input(shape=(n_ctx,), name='context')
    how_in=layers.Input(shape=(1,), dtype='int32', name='hour_of_week')
    anchor_in=layers.Input(shape=(1,), name='current_speed')

    x=layers.LSTM(128, dropout=0.2)(seq_in)                  # no recurrent_dropout -> fast cuDNN kernel on GPU
    x=layers.LayerNormalization()(x)
    emb=layers.Flatten()(layers.Embedding(168, 8)(how_in))
    c=layers.Dense(64, activation='swish')(layers.Concatenate()([ctx_in, emb]))
    z=layers.Dense(128, activation='swish')(layers.Concatenate()([x, c]))
    z=layers.Dropout(0.3)(z)
    delta=layers.Dense(len(QUANTILES))(z)
    out=layers.Add()([anchor_in, delta])                     # V(t+h) = V(t) + delta

    model=Model([seq_in, ctx_in, how_in, anchor_in], out)
    model.compile(optimizer=tf.keras.optimizers.AdamW(1e-3, weight_decay=1e-4), loss=pinball_loss)
    return model


def make_windows(seq_table, origins):
    '''seq_table: scaled array for every hour of the year (rows aligned with hour_positions).'''
    idx=origins[:, None]+np.arange(-WINDOW+1, 1)[None, :]
    return seq_table[idx]


def fit_predict_lstm(parts, seed, max_epochs=150, patience=15, batch=64):
    '''parts: dict with keys tr/va/ev, each a dict of arrays (seq, ctx, how, anchor, y).'''
    import tensorflow as tf
    tf.keras.utils.set_random_seed(seed)
    try:
        tf.config.experimental.enable_op_determinism()
    except Exception:
        pass

    tr, va, ev=parts['tr'], parts['va'], parts['ev']
    model=build_lstm(tr['seq'].shape[2], tr['ctx'].shape[1])
    inputs=lambda p: [p['seq'], p['ctx'], p['how'], p['anchor']]
    cb=[
        tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=patience, restore_best_weights=True),
        tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.3, patience=6, min_lr=1e-5),
    ]
    hist=model.fit(inputs(tr), tr['y'], validation_data=(inputs(va), va['y']),
                   epochs=max_epochs, batch_size=batch, callbacks=cb, verbose=0, shuffle=True)
    pred=model.predict(inputs(ev), batch_size=1024, verbose=0)
    return sort_quantiles(pred), {'epochs':len(hist.history['loss']), 'best_val_loss':float(min(hist.history['val_loss']))}


# ---------------------------------------------------------------- metrics

def evaluate(y, q):
    '''y: (n,), q: (n, 5) quantile forecasts in km/h. Point forecast = median.'''
    med=q[:, 2]
    err=y-med
    e=y[:, None]-q
    qs=np.array(QUANTILES)
    return {
        'n':int(len(y)),
        'MAE':float(np.mean(np.abs(err))),
        'RMSE':float(np.sqrt(np.mean(err**2))),
        'MAPE':float(np.mean(np.abs(err)/y)*100),
        'R2':float(1-np.sum(err**2)/np.sum((y-y.mean())**2)),
        'pinball':float(np.mean(np.maximum(qs*e, (qs-1)*e))),
        'PICP90':float(np.mean((y>=q[:, 0])&(y<=q[:, 4]))*100),
        'MPIW90':float(np.mean(q[:, 4]-q[:, 0])),
        'PICP50':float(np.mean((y>=q[:, 1])&(y<=q[:, 3]))*100),
    }
