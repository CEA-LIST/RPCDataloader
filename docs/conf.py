# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

import inspect
import os
from importlib.metadata import version
import importlib
import importlib.util
import sys

# -- Path setup --------------------------------------------------------------

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
#
# import os
# import sys
# sys.path.insert(0, os.path.abspath('.'))


# -- Project information -----------------------------------------------------

project = 'RPCDdataloader'
copyright = '2022, CEA LIST'
author = 'Nicolas Granger'

# The full version, including alpha/beta/rc tags.
pkg_version = version("rpcdataloader")

if len(pkg_version.split("+")) > 1:
    release = pkg_version.split("+")[0]
    commit = pkg_version.split("+")[1].split('.')[0][1:]
    version = f"latest ({release})"
else:
    release = pkg_version.split("+")[0]
    commit = f"v{release}"
    version = f"stable ({release})"


# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.intersphinx',
    'sphinx.ext.linkcode',
    'sphinxext.opengraph',
    'sphinx_copybutton',
    'sphinx_sitemap'
]

typehints_defaults = 'braces-after'
intersphinx_mapping = {
    'python': ('https://docs.python.org/3', None),
    'torch': ('https://pytorch.org/docs/stable', None)
}
ogp_site_url = "https://cea-list.github.io/RPCDataloader/"
sitemap_url_scheme = "{link}"

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']

# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
html_theme = 'sphinx_rtd_theme'

html_baseurl = 'https://cea-list.github.io/RPCDataloader/'


# -- Options for Linkcode extension -------------------------------------------

linkcode_url = "https://github.com/CEA-LIST/RPCDataloader/blob/" \
               + commit + "/{filepath}#L{linestart}-L{linestop}"


def linkcode_resolve(domain, info):
    if domain != 'py' or not info['module']:
        return None

    spec = importlib.util.find_spec(info['module'])
    if spec is None or not spec.has_location:
        return None

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    obj = module
    for part in info['fullname'].split('.'):
        try:
            obj = getattr(obj, part)
        except AttributeError:
            return None

    filepath = inspect.getfile(obj)
    for p in sys.path:
        if filepath.startswith(os.path.abspath(p)):
            filepath = os.path.relpath(filepath, os.path.abspath(p))
            break

    try:
        source, lineno = inspect.getsourcelines(obj)
    except OSError:
        return None
    else:
        linestart, linestop = lineno, lineno + len(source) - 1

    return linkcode_url.format(
        filepath=filepath,
        linestart=linestart,
        linestop=linestop)
