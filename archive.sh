#!/bin/bash -ev

git archive --format=tar.gz -o dynamic.tar.gz --prefix=/dynamic/ HEAD .

