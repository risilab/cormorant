#!/bin/bash

dot ${1}.dot -Tpdf -o ${1}.pdf; open ${1}.pdf
