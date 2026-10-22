'''
Mahmutbey traffic forecast: interactive demo.

Replays the saved forecasts of the test period (November-December 2024).
Pick a moment in time; the app shows the speed measured before it and the
forecasts for the next 1, 2 and 3 hours with calibrated uncertainty intervals,
next to what really happened.

Run:  streamlit run app/app.py
'''

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

ROOT=Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))
from src.calibration import calibrate

PRED_DIR=ROOT/'results'/'calibration'
FIG_DIR=ROOT/'figures'
MODELS={'XGBoost':'xgboost', 'LSTM':'lstm'}
HORIZONS=[1, 2, 3]
BLUE, GREY, WHITE='#4C8DFF', '#8A8F98', '#E8E8E8'

st.set_page_config(page_title='Mahmutbey traffic forecast', page_icon='🚦', layout='wide')


@st.cache_data
def load_forecasts(model):
    '''Calibrated test-period forecasts for all horizons, indexed by forecast origin.'''
    frames=[]
    for h in HORIZONS:
        p=pd.read_parquet(PRED_DIR/f'{model}_h{h}_s1.parquet')
        cal, _=calibrate(p[p['split']=='val'], p[p['split']=='test'])
        cal['horizon']=h
        cal['origin']=pd.to_datetime(cal['target_time'])-pd.Timedelta(hours=h)
        frames.append(cal)
    return pd.concat(frames)


@st.cache_data
def load_measured():
    '''Measured speed for every hour of the validation and test periods.'''
    p=pd.read_parquet(PRED_DIR/'xgboost_h1_s1.parquet')
    return p.set_index(pd.to_datetime(p['target_time']))['y'].sort_index()


def level_of_service(speed):
    if speed>=60:
        return 'Free flow'
    if speed>=30:
        return 'Dense'
    return 'Congested'


def forecast_chart(measured, fc, origin):
    hist=measured.loc[origin-pd.Timedelta(hours=12):origin]
    future=measured.reindex(fc.index)
    fig=go.Figure()
    fig.add_hrect(y0=0, y1=30, fillcolor='#3A1F1F', opacity=0.35, line_width=0)
    fig.add_hrect(y0=60, y1=100, fillcolor='#1B2A1F', opacity=0.35, line_width=0)
    fig.add_trace(go.Scatter(x=hist.index, y=hist.values, mode='lines', line=dict(color=WHITE, width=2), name='Measured (past)'))
    fig.add_trace(go.Scatter(x=list(fc.index)+list(fc.index[::-1]), y=list(fc['q95'])+list(fc['q05'][::-1]),
                             mode='none', fill='toself', fillcolor='rgba(76,141,255,0.22)', name='90% interval', hoverinfo='skip'))
    fig.add_trace(go.Scatter(x=list(fc.index)+list(fc.index[::-1]), y=list(fc['q75'])+list(fc['q25'][::-1]),
                             mode='none', fill='toself', fillcolor='rgba(76,141,255,0.40)', name='50% interval', hoverinfo='skip'))
    fig.add_trace(go.Scatter(x=fc.index, y=fc['q50'], mode='lines+markers', line=dict(color=BLUE, width=2, dash='dash'),
                             marker=dict(size=9), name='Forecast'))
    fig.add_trace(go.Scatter(x=future.index, y=future.values, mode='markers', marker=dict(color=WHITE, size=10, symbol='circle-open', line=dict(width=2)),
                             name='What really happened'))
    fig.add_vline(x=origin, line=dict(color=GREY, dash='dot'))
    fig.update_layout(template='plotly_dark', paper_bgcolor='#000000', plot_bgcolor='#000000', height=430,
                      margin=dict(l=10, r=10, t=10, b=10), yaxis=dict(title='Average speed (km/h)', range=[0, 95]),
                      legend=dict(orientation='h', y=-0.15), font=dict(family='monospace'))
    return fig


st.title('Mahmutbey traffic forecast')
st.caption('1–3 hour speed forecasts with calibrated uncertainty for Mahmutbey Junction, Istanbul · IBB open data, 2024')

tab_demo, tab_results, tab_about=st.tabs(['Forecast replay', 'Model results', 'About'])

with tab_demo:
    measured=load_measured()
    col_a, col_b, col_c=st.columns([1, 1, 1])
    model_name=col_a.radio('Model', list(MODELS), horizontal=True)
    day=col_b.date_input('Day', value=pd.Timestamp('2024-11-20').date(),
                         min_value=pd.Timestamp('2024-11-01').date(), max_value=pd.Timestamp('2024-12-31').date())
    hour=col_c.slider('Forecast made at (hour)', 0, 20, 15)

    origin=pd.Timestamp(day)+pd.Timedelta(hours=hour)
    fc_all=load_forecasts(MODELS[model_name])
    fc=fc_all[fc_all['origin']==origin].set_index(pd.to_datetime(fc_all.loc[fc_all['origin']==origin, 'target_time'])).sort_index()

    if fc.empty:
        st.info('No forecast for this moment: the IBB system had no data at the target hours (outage). Try another hour or day.')
    else:
        st.plotly_chart(forecast_chart(measured, fc, origin), width='stretch', theme=None)
        cols=st.columns(len(fc))
        for col, (t, row) in zip(cols, fc.iterrows()):
            actual=measured.get(t, np.nan)
            col.metric(f"+{int(row['horizon'])} h · {t:%H:%M}", f"{row['q50']:.0f} km/h")
            measured_txt=f'measured {actual:.0f} km/h' if not np.isnan(actual) else 'not measured'
            col.caption(f"90%: {row['q05']:.0f}–{row['q95']:.0f} km/h · {level_of_service(row['q50'])} · {measured_txt}")
        st.caption('Forecasts use only information available at the chosen hour. Shaded areas: congested (< 30 km/h) and free flow (> 60 km/h).')
        with st.expander('How to read this chart'):
            st.markdown('''
- **The dotted line** is the moment the forecast is made. Everything to its left was already measured; the model
  has seen nothing to its right.
- **Blue points** are the forecasts for 1, 2 and 3 hours later (the median of the predicted distribution).
- **Blue bands** show the uncertainty: the true speed should fall in the light band 90% of the time and in the dark
  band 50% of the time. The bands were calibrated on September–October data, so these percentages hold in practice.
- **White circles** show what was really measured later, so you can judge each forecast yourself.

**How the forecast is made.** XGBoost (or the LSTM) uses the last hours of speed at Mahmutbey and at six neighbouring
road cells, the usual speed for that weekday and hour, the Turkish calendar (holidays, Ramadan, iftar, school days)
and recent weather. It predicts five quantiles of the future speed; the outer ones form the 90% band.
''')

with tab_results:
    st.subheader('Test period, November–December 2024 (measured hours only)')
    st.markdown('''
| | 1 h ahead | 2 h ahead | 3 h ahead |
|---|---|---|---|
| **XGBoost** MAE | **2.72** | **3.10** | **3.32** |
| **LSTM** MAE | **2.71** | 3.17 | 3.42 |
| Seasonal naive MAE | 4.62 | 4.62 | 4.62 |
| 90% interval coverage, after calibration | 89–90% | 90–91% | 89–91% |
''')
    for name, caption in [('fig17_accuracy', 'Accuracy by horizon'), ('fig19_ablation', 'Which information helps'),
                          ('fig20_calibration', 'Interval calibration')]:
        path=FIG_DIR/f'{name}.png'
        if path.exists():
            st.image(str(path), caption=caption, width='stretch')

with tab_about:
    st.markdown('''
**What is this?** A research project that forecasts the hourly average speed at Mahmutbey Junction, one of the busiest
nodes in Istanbul, 1 to 3 hours ahead. Each forecast comes with an uncertainty interval that is calibrated with
conformalized quantile regression, so a 90% interval really contains the true speed about 90% of the time.

**Data.** IBB hourly traffic density data (İBB Open Data License), Open-Meteo weather (CC BY 4.0), Turkish calendar.

**This demo** replays saved forecasts of the test period; it does not use live data.

Code and full analysis: [github.com/sfeyzayildiz/mahmutbey-traffic-forecast](https://github.com/sfeyzayildiz/mahmutbey-traffic-forecast)
''')
