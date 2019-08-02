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
    'sphinx.ext.graphviz'
    # 'sphinxcontrib.plantuml'
]

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
author = u'Brandon Anderson'
copyright = '{0}, {1}'.format(year, author)
version = release = u'0.1.0'

pygments_style = 'trac'
templates_path = ['.']
# extlinks = {
#     'issue': ('https://github.com/DiffusionMapsAcademics/pydiffmap/issues/%s', '#')
#     'pr': ('https://github.com/DiffusionMapsAcademics/pydiffmap/pull/%s', 'PR #'),
# }
# on_rtd is whether we are on readthedocs.org
on_rtd = os.environ.get('READTHEDOCS', None) == 'True'

if not on_rtd:  # only set the theme if we're building docs locally
    html_theme = 'sphinx_rtd_theme'

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
