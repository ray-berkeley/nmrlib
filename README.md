# nmrlib
A collection of classes to facilitate working with NMR data, based mostly on nmrglue. 

## Plotting Data
Data in Bruker or SPARKY format can be plotted with matplotlib/HoloViews to yield contour plots with one 'object' per contour line. This makes it easy to prepare spectra for publication in Illustrator or InkScape.

```python
import holoviews as hv
from holoviews import opts
hv.extension('matplotlib')
import nmrlib as nl

pdir = "/path/to/data/pdata/1/"
rdir = "/path/to/data/"

data = nl.BrukerData(pdir, rdir)
plot = hv.Path(data.get_contours(), vdims = 'level', label = 'My Spectrum')

plot.opts(invert_xaxis = True, invert_yaxis = True,
          data_aspect = 1, # responsive = True, can be used with hv's bokeh backend 
          xlabel='¹³C (ppm)', ylabel='¹³C (ppm)',
          xlim = (0, 200), ylim = (0, 200),
          color = 'Black', show_legend = True
          )

hv.save(plot, 'my_spectrum.svg')
```