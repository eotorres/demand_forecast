import streamlit as st
import pandas as pd
import numpy as np

from prophet import Prophet
from prophet.plot import add_changepoints_to_plot, plot_plotly

from prophet.diagnostics import cross_validation
from prophet.diagnostics import performance_metrics
from prophet.plot import plot_cross_validation_metric
import json
from prophet.serialize import model_to_json, model_from_json
import holidays

import altair as alt
import plotly as plt
import matplotlib.pyplot as pltt
import plotly.offline as pyoff
import plotly.graph_objs as go
import plotly.figure_factory as ff
import base64
import itertools
from datetime import datetime
import json
import warnings

class SessionState:
  def __init__(self):
    self.clear_cache = False

state = SessionState()

st.set_page_config(page_title="Forecast",
                   initial_sidebar_state="collapsed",
                   page_icon="🔮")

st.markdown("<h1 style='text-align: center;'>Forecast - Application 🔮</h1>"
	    """
       <style>
       [data-testid="stSidebar"][aria-expanded="true"]{
           min-width: 350px;
           max-width: 500px;
       }
       """,unsafe_allow_html=True)

tabs = ["Application","Analytics","About"]
page = st.sidebar.radio("Menus", tabs)

# @st.cache(persist=False,
#           allow_output_mutation=True,
#           suppress_st_warning=True,
#           show_spinner=True)
@st.cache_data(persist=False,
               show_spinner=True)

def load_csv():
  df_input = pd.DataFrame()
  df_input = pd.read_csv(input,
                         sep=';',
                         engine='python',
                         encoding='utf-8',
                         parse_dates=True,
                         infer_datetime_format=True)
  return df_input

def prep_data(df):
  df_input = df.rename({
    date_col: "ds",
    metric_col: "y"
  },
                       errors='raise',
                       axis=1)
  st.markdown(
    "A coluna de data selecionada agora é rotulada como **ds** e as colunas de valores como **y**"
  )
  df_input = df_input[['ds', 'y']]
  df_input = df_input.sort_values(by='ds', ascending=True)
  return df_input

code1 = """
st.dataframe(df)          
st.write(df.describe())

try:
    line_chart=alt.Chart(df).mark_line().encode(x='ds:T',y="y:Q").properties(title="Time series preview").interactive()
    st.altair_chart(line_chart,use_container_width=True)
except:
    st.line_chart(df['y'],use_container_width =True,height=300) 
"""

code2 = """
m = Prophet(
    seasonality_mode=seasonality,
    daily_seasonality=daily,
    weekly_seasonality=weekly,
    yearly_seasonality=yearly,
    growth=growth,
    changepoint_prior_scale=changepoint_scale,
    seasonality_prior_scale= seasonality_scale)

if holidays:
    m.add_country_holidays(country_name=selected_country)
                        
if monthly:
    m.add_seasonality(name='monthly', period=30.4375, fourier_order=5)

m = m.fit(df)
future = m.make_future_dataframe(periods=periods_input,freq='D')
future['cap']=cap
future['floor']=floor
"""

code3 = """
try:     
    df_cv=cross_validation(m,initial=initial,period=period,horizon = horizon,parallel="processes")
except:
    df_cv = cross_validation(m, initial=initial,period=period, horizon = horizon,parallel="threads")
except:
    st.write("Invalid configuration")    
    df_p = performance_metrics(df_cv)
    st.dataframe(df_p)

metrics = ['mse','rmse','mae','mape','mdape','coverage']

selected_metric = st.radio(label='Plot metric',options=metrics)
st.write(selected_metric)
fig4 = plot_cross_validation_metric(df_cv, metric=selected_metric)
st.write(fig4)
"""

code4 = """
param_grid = {'changepoint_prior_scale': [0.01, 0.1, 0.5],'seasonality_prior_scale': [0.1, 1.0, 10.0]}

# Generate all combinations of parameters
all_params = [dict(zip(param_grid.keys(),v)) for v in itertools.product(*param_grid.values())]
rmses = []  # Store the RMSEs for each params here

# Use cross validation to evaluate all parameters
for params in all_params:
m = Prophet(**params).fit(df)  # Fit model with given params
df_cv = cross_validation(m, initial=initial,period=period,horizon=horizon,parallel="processes")
df_p = performance_metrics(df_cv, rolling_window=1)
rmses.append(df_p['rmse'].values[0])

# Find the best parameters
tuning_results = pd.DataFrame(all_params)
tuning_results['rmse'] = rmses
st.write(tuning_results)

best_params = all_params[np.argmin(rmses)]
st.write('The best parameter combination is:')
st.write(f"Changepoint prior scale: ** {best_params[0]} ** ")
st.write(f"Seasonality prior scale: ** {best_params[1]} ** ")
"""

code_options = [
  "Dataframe information", "Model fitting", "Cross validation",
  "Hyperparameter tuning"
]

if page == "Application":
  with st.sidebar:
    if st.button(label='Limpar cache'): state.clear_cache = True

    with st.expander("Code snippets"):
      snippet = st.radio('Code snippets', options=code_options)
      if snippet == code_options[0]: st.code(code1)
      if snippet == code_options[1]: st.code(code2)
      if snippet == code_options[2]: st.code(code3)
      if snippet == code_options[3]: st.code(code4)

  st.title('Forecast Away 🧙🏻')
  st.write(
    'Este aplicativo permite gerar previsões de séries temporais sem quaisquer dependências.'
  )
  st.markdown(
    """A biblioteca de previsão usada é **[Prophet](https://facebook.github.io/prophet/)**."""
  )
  state.clear_cache = True
  df = pd.DataFrame()

  st.subheader('1. Data loading 🏋️')
  st.write("Importe um arquivo .csv de série temporal. 'Usar separador ;")
  with st.expander("Data format"):
    st.write(
      "O conjunto de dados pode conter várias colunas, mas você precisará selecionar uma coluna para ser usada como datas e uma segunda coluna contendo a métrica que deseja prever. As colunas serão renomeadas como **ds** e **y** para serem compatíveis com o Profeta. Mesmo que estejamos usando o analisador de data padrão do Pandas, a coluna ds (datestamp) deve ter um formato esperado pelo Pandas, idealmente AAAA-MM-DD para uma data ou AAAA-MM-DD HH:MM:SS para um registro de data e hora. A coluna y deve ser numérica."
    )

  input = st.file_uploader('')

  if input is None:
    st.write("Ou use um conjunto de dados de amostra para testar o aplicativo")
    sample = st.checkbox("Baixe dados de amostra do GitHub")

  try:
    if sample:
      st.markdown(
        """[download_link](https://raw.githubusercontent.com/rbb-99/forecast-away/main/sample_forecast.csv)"""
      )
  except:
    if input:
      with st.spinner('Loading data...'):
        df = load_csv()

        st.write("Columns:")
        st.write(list(df.columns))
        columns = list(df.columns)

        col1, col2 = st.columns(2)
        with col1:
          date_col = st.selectbox("Select date column",
                                  index=0,
                                  options=columns,
                                  key="date")
        with col2:
          metric_col = st.selectbox("Select values column",
                                    index=1,
                                    options=columns,
                                    key="values")

        df = prep_data(df)
        output = 0

    if st.checkbox('Chart data', key='show'):
      with st.spinner('Plotting data..'):
        col1, col2 = st.columns(2)
        with col1:
          st.dataframe(df)

        with col2:
          st.write("Dataframe description:")
          st.write(df.describe())

      try:
        line_chart = alt.Chart(df).mark_line().encode(
          x='ds:T', y="y:Q",
          tooltip=['ds:T',
                   'y']).properties(title="Time series preview").interactive()
        st.altair_chart(line_chart, use_container_width=True)

      except:
        st.line_chart(df['y'], use_container_width=True, height=300)

  st.subheader("2. Configuração de parâmetros 🛠️")
  with st.container():
    st.write('Nesta seção, você pode modificar as configurações do algoritmo.')

    with st.expander("Horizon"):
      periods_input = st.number_input(
        'Selecione quantos períodos futuros (dias) serão previstos.',
        min_value=1,
        max_value=366,
        value=90)

    with st.expander("Seasonality"):
      st.markdown(
        """A sazonalidade padrão usada é aditiva, mas a melhor escolha depende do caso específico, portanto, é necessário conhecimento específico do domínio. Para mais informações visite o [documentation](https://facebook.github.io/prophet/docs/multiplicative_seasonality.html)"""
      )
      seasonality = st.radio(label='Seasonality',
                             options=['additive', 'multiplicative'])

    with st.expander("Componentes de tendência"):
      st.write("Adicionar ou remover componentes:")
      daily = st.checkbox("Daily")
      weekly = st.checkbox("Weekly")
      monthly = st.checkbox("Monthly")
      yearly = st.checkbox("Yearly")

    with st.expander("Modelo de crescimento"):
      st.write('Prophet uses by default a linear growth model.')
      st.markdown(
        """Para mais informações verifique o [documentation](https://facebook.github.io/prophet/docs/saturating_forecasts.html#forecasting-growth)"""
      )

      growth = st.radio(label='Growth model', options=['linear', "logistic"])

      if growth == 'linear':
        growth_settings = {'cap': 1, 'floor': 0}
        cap = 1
        floor = 1
        df['cap'] = 1
        df['floor'] = 0

      if growth == 'logistic':
        st.info('Configure saturation')

        cap = st.slider('Cap', min_value=0.0, max_value=1.0, step=0.05)
        floor = st.slider('Floor', min_value=0.0, max_value=1.0, step=0.05)
        if floor > cap:
          st.error('Configurações inválidas. Cap deve ser maior então floor.')
          growth_settings = {}

        if floor == cap:
          st.warning('Cap deve ser superior a floor')
        else:
          growth_settings = {'cap': cap, 'floor': floor}
          df['cap'] = cap
          df['floor'] = floor

    with st.expander('Holidays'):
      countries = [
        'Country name', 'Italy', 'Spain', 'United States', 'France', 'Germany',
        'Ukraine','Brazil'
      ]
      with st.container():
        years = [2021]
        selected_country = st.selectbox(label="Select country",
                                        options=countries)
        if selected_country == 'Italy':
          for date, name in sorted(holidays.IT(years=years).items()):
            st.write(date, name)
        if selected_country == 'Spain':
          for date, name in sorted(holidays.ES(years=years).items()):
            st.write(date, name)
        if selected_country == 'United States':
          for date, name in sorted(holidays.US(years=years).items()):
            st.write(date, name)
        if selected_country == 'France':
          for date, name in sorted(holidays.FR(years=years).items()):
            st.write(date, name)
        if selected_country == 'Germany':
          for date, name in sorted(holidays.DE(years=years).items()):
            st.write(date, name)
        if selected_country == 'Ukraine':
          for date, name in sorted(holidays.UKR(years=years).items()):
            st.write(date, name)
        if selected_country == 'Brazil':
          for date, name in sorted(holidays.UKR(years=years).items()):
            st.write(date, name)
        else:
          holidays = False
        holidays = st.checkbox('Adicionar feriados do país ao modelo')

    with st.expander('Hiperparâmetros'):
      st.write(
        'Nesta seção é possível ajustar os coeficientes de escala.')

      seasonality_scale_values = [0.1, 1.0, 5.0, 10.0]
      changepoint_scale_values = [0.01, 0.1, 0.5, 1.0]

      st.write(
        "A escala anterior do ponto de mudança determina a flexibilidade da tendência e, em particular, quanto a tendência muda nos pontos de mudança da tendência."
      )
      changepoint_scale = st.select_slider(label='Changepoint prior scale',
                                           options=changepoint_scale_values)

      st.write(
        "O ponto de mudança de sazonalidade controla a flexibilidade da sazonalidade."
      )
      seasonality_scale = st.select_slider(label='Seasonality prior scale',
                                           options=seasonality_scale_values)

      st.markdown(
        """Para mais informações leia o [documentation](https://facebook.github.io/prophet/docs/diagnostics.html#parallelizing-cross-validation)"""
      )

  with st.container():
    st.subheader("3. Previsão 🔮")
    st.write("Ajuste o modelo nos dados e gere previsões futuras.")
    st.write("Carregue uma série temporal para ativar.")

    if input:
      if st.checkbox("Initialize model (Fit)", key="fit"):
        if len(growth_settings) == 2:
          m = Prophet(seasonality_mode=seasonality,
                      daily_seasonality=daily,
                      weekly_seasonality=weekly,
                      yearly_seasonality=yearly,
                      growth=growth,
                      changepoint_prior_scale=changepoint_scale,
                      seasonality_prior_scale=seasonality_scale)
          if holidays:
            m.add_country_holidays(country_name=selected_country)
          if monthly:
            m.add_seasonality(name='monthly', period=30.4375, fourier_order=5)
          with st.spinner('Fitting the model..'):
            m = m.fit(df)
            future = m.make_future_dataframe(periods=periods_input, freq='D')
            future['cap'] = cap
            future['floor'] = floor
            st.write("O modelo produzirá previsões de até ",
                     future['ds'].max())
            st.success('Modelo instalado com sucesso')
        else:
          st.warning('Configuração inválida')

      if st.checkbox("Generate forecast (Predict)", key="predict"):
        try:
          with st.spinner("Forecasting.."):
            forecast = m.predict(future)
            st.success('Previsão gerada com sucesso')
            st.dataframe(forecast)
            fig1 = m.plot(forecast)
            st.write(fig1)
            output = 1

            if growth == 'linear':
              fig2 = m.plot(forecast)
              a = add_changepoints_to_plot(fig2.gca(), m, forecast)
              st.write(fig2)
              output = 1
        except:
          st.warning("Você precisa treinar o modelo primeiro.")

      if st.checkbox('Show components'):
        try:
          with st.spinner("Loading.."):
            fig3 = m.plot_components(forecast)
            st.write(fig3)
        except:
          st.warning("Requer geração de previsão.")

    st.subheader('4. Validação do modelo 🧪')
    st.write(
      "Nesta seção é possível fazer a validação cruzada do modelo.")
    with st.expander("Explicação"):
      st.markdown(
        """A biblioteca do Profeta torna possível dividir nossos dados históricos em dados de treinamento e dados de teste para validação cruzada. Os principais conceitos para validação cruzada com o Prophet são:"""
      )
      st.write(
        "Dados de treinamento (inicial): a quantidade de dados reservada para treinamento. O parâmetro está na API chamada inicial."
      )
      st.write("Horizon: Os dados separados para validação.")
      st.write(
        "Cutoff (period): uma previsão é feita para cada ponto observado entre cutoff and cutoff + horizon."
        "")

    with st.expander("Cross validation"):
      initial = st.number_input(value=365,
                                label="initial",
                                min_value=30,
                                max_value=1096)
      initial = str(initial) + " days"

      period = st.number_input(value=90,
                               label="period",
                               min_value=1,
                               max_value=365)
      period = str(period) + " days"

      horizon = st.number_input(value=90,
                                label="horizon",
                                min_value=30,
                                max_value=366)
      horizon = str(horizon) + " days"

      st.write(
        f"Aqui fazemos validação cruzada para avaliar o desempenho da previsão em um horizon de **{horizon}** dias, começando com **{initial}** dias de dados de treinamento no primeiro cutoff e, em seguida, fazendo previsões a cada **{period}**."
      )
      st.markdown(
        """Para mais informações leia o [documentation](https://facebook.github.io/prophet/docs/diagnostics.html#parallelizing-cross-validation)"""
      )

    with st.expander("Metrics"):
      if input:
        if output == 1:
          metrics = 0
          if st.checkbox('Calculate metrics'):
            with st.spinner("Cross validating.."):
              try:
                df_cv = cross_validation(m,
                                         initial=initial,
                                         period=period,
                                         horizon=horizon,
                                         parallel="processes")
                df_p = performance_metrics(df_cv)
                st.write(df_p)
                metrics = 1
              except:
                st.write("Configuração inválida. Tente outros parâmetros.")
                metrics = 0

              st.markdown('**Definição de métricas**')
              st.write("Mse: mean absolute error")
              st.write("Mae: Mean average error")
              st.write("Mape: Mean average percentage error")
              st.write("Mse: mean absolute error")
              st.write("Mdape: Median average percentage error")

              if metrics == 1:
                metrics = [
                  'Choose a metric', 'mse', 'rmse', 'mae', 'mape', 'mdape',
                  'coverage'
                ]
                selected_metric = st.selectbox("Select metric to plot",
                                               options=metrics)
                if selected_metric != metrics[0]:
                  fig4 = plot_cross_validation_metric(df_cv,
                                                      metric=selected_metric)
                  st.write(fig4)
      else:
        st.write("Crie uma previsão para ver as métricas")

    st.subheader('5. Ajuste de hiperparâmetros 🧲')
    st.write(
      "Nesta seção é possível encontrar a melhor combinação de hiperparâmetros."
    )
    st.markdown(
      """Para mais informações visite o [documentation](https://facebook.github.io/prophet/docs/diagnostics.html#hyperparameter-tuning)"""
    )

    param_grid = {
      'changepoint_prior_scale': [0.01, 0.1, 0.5, 1.0],
      'seasonality_prior_scale': [0.1, 1.0, 5.0, 10.0],
    }

    # Generate all combinations of parameters
    all_params = [
      dict(zip(param_grid.keys(), v))
      for v in itertools.product(*param_grid.values())
    ]
    rmses = []  # Store the RMSEs for each params here

    if input:
      if output == 1:
        if st.button("Otimizar hiperparâmetros"):
          with st.spinner("Encontrando a melhor combinação. Por favor, aguarde.."):
            try:
              # Use cross validation to evaluate all parameters
              for params in all_params:
                m = Prophet(**params).fit(df)  # Fit model with given params
                df_cv = cross_validation(m,
                                         initial=initial,
                                         period=period,
                                         horizon=horizon,
                                         parallel="processes")
                df_p = performance_metrics(df_cv, rolling_window=1)
                rmses.append(df_p['rmse'].values[0])
            except:
              for params in all_params:
                m = Prophet(**params).fit(df)  # Fit model with given params
                df_cv = cross_validation(m,
                                         initial=initial,
                                         period=period,
                                         horizon=horizon,
                                         parallel="threads")
                df_p = performance_metrics(df_cv, rolling_window=1)
                rmses.append(df_p['rmse'].values[0])

          # Find the best parameters
          tuning_results = pd.DataFrame(all_params)
          tuning_results['rmse'] = rmses
          st.write(tuning_results)

          best_params = all_params[np.argmin(rmses)]

          st.write('The best parameter combination is:')
          st.write(best_params)
          #st.write(f"Changepoint prior scale:  {best_params[0]} ")
          #st.write(f"Seasonality prior scale: {best_params[1]}  ")
          st.write(
            " You may repeat the process using these parameters in the configuration section 2"
          )
      else:
        st.write("Create a model to optimize")

    st.subheader('6. Exportar resultados ✨')
    st.write(
      "Por fim, você pode exportar sua previsão de resultados, configuração do modelo e métricas de avaliação."
    )

    if input:
      if output == 1:
        col1, col2, col3, col4 = st.columns(4)
        with col1:
          if st.button('Previsão de exportação (.csv)'):
            with st.spinner("Exporting.."):
              #export_forecast = pd.DataFrame(forecast[['ds','yhat_lower','yhat','yhat_upper']]).to_csv()
              export_forecast = pd.DataFrame(
                forecast[['ds', 'yhat_lower', 'yhat', 'yhat_upper']])
              st.write(export_forecast.head())
              export_forecast = export_forecast.to_csv(decimal=',')
              b64 = base64.b64encode(export_forecast.encode()).decode()
              href = f'<a href="data:file/csv;base64,{b64}">Download CSV File</a> (click > guardar como **forecast.csv**)'
              st.markdown(href, unsafe_allow_html=True)
        with col2:
          if st.button("Exportar métricas de modelo (.csv)"):
            try:
              df_p = df_p.to_csv(decimal=',')
              b64 = base64.b64encode(df_p.encode()).decode()
              href = f'<a href="data:file/csv;base64,{b64}">Download CSV File</a> (click > guardar como **metrics.csv**)'
              st.markdown(href, unsafe_allow_html=True)
            except:
              st.write("Nenhuma métrica para exportar")
        with col3:
          if st.button('Salve a configuração do modelo (.json) na memória'):
            with open('serialized_model.json', 'w') as fout:
              json.dump(model_to_json(m), fout)
        with col4:
          if st.button('Limpe a memória cache por favor'):
            state.clear_cache = True
      else:
        st.write("Gere uma previsão para download.")

if page == "Analytics":
        
  st.header("Analytics")
  uploaded_file = st.file_uploader("Upload CSV", type="csv")
  if uploaded_file is not None:
    df = pd.read_csv(uploaded_file,sep=';')


    st.write(df.head())
    time_column = st.selectbox(
    'Escolha o nome da coluna de tempo: ',
    tuple(['<select>'] + list(df.columns)))
    

    forecasting_column = st.selectbox(
    'Escolha o nome da coluna de previsão: ',
    tuple(['<select>'] + list(df.columns)))
    if time_column!='<select>' and forecasting_column!='<select>':
        df = df[[time_column, forecasting_column]].copy()
        df.rename(columns={time_column:'ds',forecasting_column:'y'},inplace=True)

        weeks = st.selectbox(
        'Quantas semanas no futuro você deseja fazer previsão: ',
        tuple(['<select>'] + [i for i in range(1,105)]))

        interval_width = st.selectbox('Escolha a largura do intervalo: ',
                                 tuple(['<select>'] + [i/100 for i in range(0,100,5)]))

        if weeks != '<select>' and interval_width!='<select>':

            model = Prophet(interval_width=float(interval_width))
            model.fit(df)

            future = model.make_future_dataframe(periods=int(weeks)*7)
            forecast = model.predict(future)

            tail_df = forecast.tail(int(weeks)*7)

            # Plot the forecast
            fig = plot_plotly(model,forecast)
            st.plotly_chart(fig,use_container_width=True)

            @st.cache_data
            def convert_df(df):
                return df.to_csv(index=False).encode('utf-8')
            csv = convert_df(tail_df)
            st.download_button("Press to Download", csv, f"forecast_{forecasting_column}.csv", "text/csv", key='download-csv')

if page == "About":
  st.header("About")
  st.markdown(
    "Documentação oficial de **[Facebook Prophet](https://facebook.github.io/prophet/)**"
  )
  st.markdown(
    "Documentação oficial de **[Streamlit](https://docs.streamlit.io/en/stable/getting_started.html)**"
  )
  st.write("")
  st.write("Autor:")
  st.markdown(""" **Emmanuel**""")
  st.write("Created on 29/05/2023")
