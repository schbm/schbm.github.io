---
layout: single
title:  "Including Plotly into Jekyll Post"
date:   2026-01-19 12:00:00 +0100
show_date: true
categories: ds
tags: ds guide
toc: true
---

In this short post i want to quickly show you how
to include interactive data plots in Jekyll posts using the [Plotly Python package](https://plotly.com/python/).

The use case is to include these plots from a Python codebase or Jupyter Notebooks.

First we need to include these imports:
{% highlight python %}
    import plotly.graph_objects as go
    import plotly.io as pio
{% endhighlight %}

Now include or create your plots:
{% highlight python %}
fig = go.Figure(
    (...)
)

# Notice here we also fix the height,
# but not the width to make it fit automatically
fig.update_layout(
    width=None,
    height=600,
    (...)
)

fig.show()
{% endhighlight %}

For the export we can make use of an existing write_html function.

Set the include_plotlyjs to 'cdn', a script tag that references the plotly.js CDN is included
in the output. HTML files generated with this option are about 3MB
smaller than those generated with include_plotlyjs=True, but they
require an active internet connection in order to load the plotly.js
library. Set full_html to False.
If False, produce a string containing a single <div> element.
Set auto_play to false. It controls whether to automatically 
start the animation sequence on page load
if the figure contains frames. Has no effect if the figure does not
contain frames.

{% highlight python %}
    pio.write_html(
        fig, file="plot.html", include_plotlyjs="cdn",
        auto_play=False, full_html=False,
        config={"responsive": True},
    )
{% endhighlight %}

Then finally move this file to the _includes directory.
Check if the contents of the file is of the correct form:

{% highlight html %}

<div>
    <script type="text/javascript">window.PlotlyConfig = {MathJaxConfig: 'local'};</script>
    (...)
</div>

{% endhighlight %}

And then include it in the post:

{% highlight html %}
    \{\% include plot.html \%\}
{% endhighlight %}

Without the backslashes!

{% include heart.html %}