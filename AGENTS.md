# Agent Instructions

This is a research project. Optimize for fast iterations. No need to test everything or run test. We run, we check, we evaluate. Compute is cheap.
Prefer simpler solutions, do not over-engineer or introduce unnecessary abstractions, frameworks, dependencies, or infrastructure.
Keep code simple and easy to change. 

Do not spend tokens inspecting images or browsing/viewing websites just to check results, except if ask. The user checks. Provide him with the graphs.
Do not take shortcut or hidden assumptions.  State every material assumption explicitly in your response and add them to the documentation.
Do not keep stuff around for fear of failure. A rewrote module -> the old one is discarded. even if that creates some problems, everything (calibrations, runs) will anyway be fully rerun with the latest version of the code. In writing and code, do keep track of the history behind a decision. Just describe the things. If something is important add it to the documentation's log.

When running on longleaf, we have two choices:
- regular GPUs partitions (you have one example)
- our patron nodes in a hidden partition named jlessler. We will run mostly on this, parallelizing runs in a single GPUs. we have
  - g1803jles01.ll.unc.edu:  512GB ram 56 physical CPU cores Quantity 4 of Nvidia L40, 48GB
  - g1803jles02.ll.unc.edu: 64 physical CPU cores 2Tb RAM 2x Nvidia H100, 96 GB
