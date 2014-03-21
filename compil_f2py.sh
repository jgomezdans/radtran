#!/usr/bin/bash

f2py -c --fcompiler='gfortran' -m prospect_py dataSpec_P5B.f90 main.f90 prospect_5B.f90 tav_abs.f90
