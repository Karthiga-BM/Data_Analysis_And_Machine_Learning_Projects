#!/usr/bin/env python
# md5: 99f886eaeece6e32f0944b98c07dca14
#!/usr/bin/env python
# coding: utf-8


#!pip install --upgrade plotly




import plotly.graph_objs as go
import statsmodels.api as sm
import numpy as np
from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot
init_notebook_mode(connected=True)

def plot_data(data, **kwargs):
  title = kwargs.get('title')
  xlabel = kwargs.get('xlabel')
  ylabel = kwargs.get('ylabel')
  font = kwargs.get('font')
  barmode = kwargs.get('barmode')
  layout = go.Layout()
  if title is not None:
    layout.title = go.layout.Title(text = title)
  if xlabel is not None:
    layout.xaxis = go.layout.XAxis(title = xlabel)
  if ylabel is not None:
    layout.yaxis = go.layout.YAxis(title = ylabel)
  if font is not None:
    layout.font = font
  if barmode is not None:
    layout.barmode = barmode
  fig = go.Figure(data=data, layout=layout)
  iplot(fig)

def plot_histogram(x_values, **kwargs):
  data = [go.Histogram(x=x_values)]
  plot_data(data, **kwargs)

def plot_histogram_percent(x_values, **kwargs):
  histogram_options = {'x': x_values, 'histnorm': 'percent'}
  if 'nbinsx' in kwargs:
    histogram_options['nbinsx'] = kwargs['nbinsx']
  data = [go.Histogram(**histogram_options)]
  plot_data(data, **kwargs)

def plot_histogram_cdf(x_values, **kwargs):
  histogram_options = {'x': x_values, 'histnorm': 'probability', 'cumulative': {'enabled': True}}
  if 'nbinsx' in kwargs:
    histogram_options['nbinsx'] = kwargs['nbinsx']
  data = [go.Histogram(**histogram_options)]
  plot_data(data, **kwargs)

def plot_points(points, **kwargs):
  data = [go.Scatter(x=[x for x,y in points], y=[y for x,y in points])]
  plot_data(data, **kwargs)

def plot_points_scatter(points, **kwargs):
  data = [go.Scatter(x=[x for x,y in points], y=[y for x,y in points], mode='markers')]
  plot_data(data, **kwargs)

def plot_points_scatter_with_regression(points, **kwargs):
  Y = [y for x,y in points]
  X = [x for x,y in points]
  X = sm.add_constant(X)
  model = sm.OLS(Y, X).fit()
  xvals = sm.add_constant(sorted(list(set([x for x,y in points]))))
  yvals_pred = model.predict(xvals)
  data = [
    go.Scatter(x=[x for x,y in points], y=[y for x,y in points], mode='markers'),
    go.Scatter(x=[y for x,y in xvals], y=yvals_pred, mode='lines'),
  ]
  plot_data(data, **kwargs)

def plot_several_points(points_list, **kwargs):
  data = []
  for label,points in points_list:
    data.append(go.Scatter(x=[x for x,y in points], y=[y for x,y in points], name=label))
  plot_data(data, **kwargs)

def plot_bar(label_with_value_list, **kwargs):
  labels = [x for x,y in label_with_value_list]
  remap_labels = kwargs.get('remap_labels')
  orientation = kwargs.get('orientation', 'v')
  if remap_labels is not None:
    labels = [remap_labels.get(x, x) for x in labels]
  if orientation == 'h':
    data = [go.Bar(y=labels, x=[y for x,y in label_with_value_list], orientation=orientation)]
  else:
    data = [go.Bar(x=labels, y=[y for x,y in label_with_value_list], orientation=orientation)]
  plot_data(data, **kwargs)

def plot_bar_as_percent(label_with_value_list, **kwargs):
  labels = [x for x,y in label_with_value_list]
  remap_labels = kwargs.get('remap_labels')
  if remap_labels is not None:
    labels = [remap_labels.get(x, x) for x in labels]
  val_sum = sum([y for x,y in label_with_value_list])
  normalized_y = [100*y/val_sum for x,y in label_with_value_list]
  data = [go.Bar(x=labels, y=normalized_y)]
  plot_data(data, **kwargs)

def plot_dict_as_bar(d, **kwargs):
  data = []
  items = list(d.items())
  items.sort(key=lambda x: x[1], reverse=True)
  plot_bar(items, **kwargs)

def plot_dict_of_lists_as_bar_of_medians(d, **kwargs):
  data = []
  d_medians = {}
  for key,val_list in d.items():
    d_medians[key] = np.median(val_list)
  items = list(d_medians.items())
  items.sort(key=lambda x: x[1], reverse=True)
  plot_bar(items, **kwargs)

def plot_dict_as_bar_percent(d, **kwargs):
  data = []
  nd = {}
  val_sum = sum(d.values())
  for k,v in d.items():
    nd[k] = 100 * v / val_sum
  items = list(nd.items())
  items.sort(key=lambda x: x[1], reverse=True)
  plot_bar(items, **kwargs)

def plot_dict_as_bar_fraction(d, **kwargs):
  data = []
  nd = {}
  val_sum = sum(d.values())
  for k,v in d.items():
    nd[k] = v / val_sum
  items = list(nd.items())
  items.sort(key=lambda x: x[1], reverse=True)
  plot_bar(items, **kwargs)

def plot_heatmap(heatmap_data, **kwargs):
  ticktext = kwargs.get('ticktext')
  colorscale = kwargs.get('colorscale')
  
  heatmap = go.Heatmap(z = heatmap_data)
  heatmap.colorscale = 'Viridis' # Greys
  if colorscale is not None:
    heatmap.colorscale = colorscale
  heatmap.showscale = True
  if ticktext is not None:
    heatmap.colorbar = dict(
      tickmode = 'array',
      tickvals = list(range(len(ticktext))),
      ticktext = ticktext,
      #tick0 = 0,
      #dtick = 1,
      #autotick = False,
    )
  data = [ heatmap ]
  plot_data(data, **kwargs)

def plot_dictdict_as_bar(d_dict, **kwargs):
  data = []
  remap_labels = kwargs.get('remap_labels', {})
  for label,d in d_dict.items():
    label = remap_labels.get(label, label)
    items = list(d.items())
    items.sort(key=lambda x: x[1], reverse=True)
    data.append(go.Bar(x=[remap_labels.get(x, x) for x,y in items], y=[y for x,y in items], name=label))
  plot_data(data, **kwargs)

def plot_stacked_bar_chart(target_name_to_type_to_time_spent_all, states=None, **kwargs):
  target_name_to_time_spent = {}
  for target_name,type_to_time_spent in target_name_to_type_to_time_spent_all.items():
    target_name_to_time_spent[target_name] = sum(type_to_time_spent.values())
  target_name_list = list(target_name_to_time_spent.keys())
  target_name_list.sort(key=lambda x: target_name_to_time_spent[x], reverse=True)
  data_list = []
  if states is None:
    states = sorted(set(itertools.chain.from_iterable([d.keys() for d in target_name_to_type_to_time_spent_all.values()])))
  for state in states:
    data_list.append(go.Bar(name=state, x=target_name_list, y=[target_name_to_type_to_time_spent_all[x].get(state, 0) for x in target_name_list]))
  plot_data(data_list, barmode='stack', **kwargs)



