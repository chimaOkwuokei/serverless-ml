#!/bin/bash

set -e

cd src/credit-card-fraud

jupyter nbconvert --to notebook --execute 2_cc_feature_pipeline.ipynb

