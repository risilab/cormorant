# -*- coding: utf-8 -*-
from __future__ import unicode_literals

import os


extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.coverage',
    'sphinx.ext.extlinks',
    'sphinx.ext.ifconfig',
    'sphinx.ext.mathjax',
    'sphinx.ext.napoleon',
    'sphinx.ext.viewcode',
    'sphinx.ext.graphviz',
    'sphinx.ext.intersphinx'
    # 'sphinxcontrib.plantuml'
]

intersphinx_mapping = {
            'python': ('https://docs.python.org/', None),
            'numpy': ('https://docs.scipy.org/doc/numpy/', None),
            'PyTorch': ('https://pytorch.org/docs/master/', None)
                }
#    'sphinx.ext.autosummary',
#    'sphinx.ext.todo',
#    'sphinx.ext.doctest',
if os.getenv('SPELLCHECK'):
    extensions += 'sphinxcontrib.spelling',
    spelling_show_suggestions = True
    spelling_lang = 'en_US'

show_authors = False

source_suffix = '.rst'
master_doc = 'index'
project = u'cormorant'
year = '2019'
author = u'Brandon Anderson, Erik Thiede'
copyright = '{0}, {1}'.format(year, author)
version = release = u'0.1.0'

pygments_style = 'trac'
templates_path = ['.']
on_rtd = os.environ.get('READTHEDOCS', None) == 'True'

if not on_rtd:  # only set the theme if we're building docs locally
    html_theme = 'sphinx_rtd_theme'
    # html_theme = 'pytorch_sphinx_theme'

html_use_smartypants = True
html_last_updated_fmt = '%b %d, %Y'
html_split_index = False
html_sidebars = {
    '**': ['searchbox.html', 'globaltoc.html', 'sourcelink.html'],
}
html_short_title = '%s-%s' % (project, version)

napoleon_use_ivar = True
napoleon_use_rtype = False
napoleon_use_param = False
