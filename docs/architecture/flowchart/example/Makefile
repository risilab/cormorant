LADOT=../ladot

test.ps: test.dvi
	dvips test.dvi -o test.ps

test.dvi: test.tex testfig.ps
	latex test.tex

testfig.ps: testfig.ladot
	$(LADOT) testfig.ladot

clean:
	rm -f *.ps *.dvi testfig.tex *.aux *.log
