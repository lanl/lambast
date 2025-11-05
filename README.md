# LAMBAST - Los Alamos Model Bias Assessment and Statistical Toolkit

## Installation

You can install LAMBAST with

    pip install .

## Use

Simply `import lambast.<module>` in your python script.

## Docs

TODO

### Overview

LAMBAST consists currently of 3 modules:

- detection_methods
- generate_data
- utils

Each of these modules can be explored through the `html` files in the `docs`
directory or using `help` from the python REPL, for example:

```python-repl
>>> from lambast import generate_data
>>> help(generate_data)
>>> help(generate_data.copulas)
```

### Examples

See the `lambast/examples` directory for examples on how to use lambast.
