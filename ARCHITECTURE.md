# Architecture notes

## Core design
Kind focuses on a local, single-pass decision engine for routing and scoring.
This keeps latency low while maintaining typed outputs for downstream systems.

## Use cases
- request routing
- guardrails
- incident triage
- decision calibration
