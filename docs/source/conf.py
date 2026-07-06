# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = "transport_flow_model"
copyright = "2026, Tom Russell, Raghav Pant, Yue Li, Alberto Fernández-Pérez"
author = "Tom Russell, Raghav Pant, Yue Li, Alberto Fernández-Pérez"

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.doctest",
    "sphinx.ext.intersphinx",
    "sphinx.ext.napoleon",
]

autodoc_member_order = "bysource"
autodoc_typehints = "description"

intersphinx_mapping = {
    "pandas": ("https://pandas.pydata.org/docs", None),
    "pyarrow": ("https://arrow.apache.org/docs", None),
    "python": ("https://docs.python.org/3", None),
}

templates_path = ["_templates"]
exclude_patterns = []


# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = "pydata_sphinx_theme"
html_static_path = ["_static"]
html_sidebars = {
    "**": ["globaltoc.html"],
}
