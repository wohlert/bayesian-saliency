# Bayesian Saliency

This project aims to recreate the results of [Zhang et al 2008](http://jov.arvojournals.org/article.aspx?articleid=2297284#133855129)
by implementing a Bayesian framework to compute visual saliency
based on natural image statistics.

## How to use

The `main.py` file will compute the ICA components and the parameters of
the distributions. Make sure that `eizaburo-doi-kyoto_natim-c2015ff` is
located in the same directory along with `CAT2000_train`.

Then run.

```
python main.py
```
