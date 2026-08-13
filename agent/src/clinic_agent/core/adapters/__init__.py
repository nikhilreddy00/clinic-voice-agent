"""Adapters: the only components in the engine that touch I/O.

Each one turns actions into vendor calls and vendor callbacks into events. They hold no
dialogue state — the reducer owns all of it — which is what lets several sessions share a
process and lets a recorded call replay with every adapter absent.
"""
