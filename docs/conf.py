# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = 'UniGeneX'
copyright = '2026, UniGeneX authors'
author = 'UniGeneX authors'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

templates_path = ['_templates']
exclude_patterns = ['_build', '**/.ipynb_checkpoints']

extensions = [
    'sphinx_rtd_theme',
    'nbsphinx',
]

# Documentation notebooks contain saved outputs from large analyses. Render
# those outputs on Read the Docs without rerunning the notebooks.
nbsphinx_execute = 'never'

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = 'sphinx_rtd_theme'
html_static_path = ['_static']
