#!/usr/bin/env bash
python3 -m venv venv
source venv/bin/activate
pip install discord.py
pip install transformers
pip install deepspeed