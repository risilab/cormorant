Ladot 1.2
---------

2005-12-04
Brighten Godfrey
godfreyb@bigw.org


License
-------

This program is free software; you can redistribute it and/or
modify it under the terms of the GNU General Public License
as published by the Free Software Foundation; either version 2
of the License, or (at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program; if not, write to the Free Software
Foundation, Inc., 59 Temple Place - Suite 330, Boston, MA  02111-1307, USA.

Description
-----------

For a general description of Ladot, please see

    http://brighten.bigw.org/projects/ladot/

One-time setup
--------------

Install psfrag, dot (graphviz), and latex.  If you are using this
program, you probably already have dot and latex.  psfrag is available
here:

    http://www.ctan.org/tex-archive/help/Catalogue/entries/psfrag.html?action=/tex-archive/macros/latex/contrib/supported/psfrag/


Usage
-----

1.  Specify your graph in a file ending in `.ladot'.  The format is dot's,
    except that you may replace text with LaTeX math expressions, thusly:

        digraph mygraph {
            $v_1$(2) $v_2$
            $v_1$ -> $v_2$
            }

    You may optionally append a size hint in parenthesis as shown above.
    This tells Ladot rougly how many characters wide your rendered LaTeX
    will be (not an exact science).  If you leave out the size hint, as
    in node $v_2$ in the above example, Ladot will take a guess.  You only
    have to use the size hint once for each unique LaTeX snippet.

2.  Run ladot on the file:

        ladot mygraph.ladot

    This creates `mygraph.ps' (the raw PostScript output of dot) and
    `mygraph.tex' (a snippet of LaTeX code which tells psfrag to
    substitute tags in the PostScript for the LaTeX figures).

3.  In the preamble of your LaTeX file, put

        \usepackage{psfrag}

    LyX users: Set your preamble in Layout -> Document -> Preamble.

4.  Where you would like to include the graph, write
    
        \input{mydot.tex}
        \includegraphics{mydot.ps}

    LyX users: use Insert -> Include File and Insert -> Graphics.

5.  Process your LaTeX file with `latex' as normal.  Note that psfrag
    is sadly not compatible with pdflatex.

An example of all this is in the `example' directory.  Just type `make'
to produce the final output, `test.ps'.

Limitations/Bugs/To Do
----------------------

    * Is limited by psfrag: doesn't work with pdflatex.
    * Cannot exploit full power of psfrag, especially regarding alignment
      in formatting.  Ladot centers everything.  Should add option for this.
    * Quoting is nonexistent.  Don't try to put a dollar sign in any of your
      TeX or Dot! :-)  Several problems would be fixed if we used a real dot
      parser instead of just some perl hack, but I haven't gotten around to
      doing this.
    * Each LaTeX snippet must appear on a single line in your Ladot file.
      That is, you cannot have something like
      
          $foo
             bar$
    
    * Since I only ever need to use LaTeX in Dot for math, Ladot only handles
      math expressions.  This is a needless limitation, but it makes the
      format more convenient.  You can still use normal text, however, by
      turning on the appropriate font within the equation.
    * It can't take input on stdin.
    * Your files must have the suffix ".ladot".
    * Ladot picks short random strings for the stubs inserted into the
      PostScript (which you can see in the mydot.ps output, but which are
      replaced with LaTeX in the output of your LaTeX file).  There is a
      chance these could conflict with each other or with other labels in the
      Dot file, causing incorrect, confusing, and frustrating results.
      (Note that we could remember which stubs we picked and pick new ones, but
      still there might be conflicts with other text if stubs are short.)
      The best solution would be to make the stubs much longer, thus
      making the probability of any conflict negligible.  However, currently
      we sometimes need to pick short stubs since the stub length tells Dot
      how much space to leave for the label.  This would be fixed by fixing
      the next limitation...
    * Should have better control on how much space is left for rendered LaTeX
      (e.g., inches/pixels/mm instead of fuzzy "number of characters" measure).

Change Log
----------

Version 1.2 (2005-12-04)

    * Fixed problem that arises with some versions of `sed'.

Version 1.1 (2005-12-03)

    * Workaround for compatibility with Graphviz 2.2 and higher.  Those versions
      of Graphviz output postscript which confuses psfrag, so we have to do
      a little postprocessing of Graphviz's postscript.  The fix is courtesy of
      Riccardo de Maria and Kjell Magne Fauske.
    * You may now include $math$ at any point within a label, rather than being
      forced to make the entire label one math expression.  For example:
      
          mynode [label = "$math on line one$(10)\n$...and line two$(7)"]
          
      Thanks to Alet Roux for suggesting this.

Version 1.0 (2003-03-31)

    * Initial release.
