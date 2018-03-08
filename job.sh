#!/bin/bash
rm cfp.config
cp cfp.config.0 cfp.config
py2 main.py
rm cfp.config
cp cfp.config.1 cfp.config
py2 main.py
rm cfp.config
cp cfp.config.2 cfp.config
py2 main.py
