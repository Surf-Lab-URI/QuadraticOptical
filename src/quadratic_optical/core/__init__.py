"""Image-only quadratic registration and conservative field evaluation.

All coordinates are zero-based image pixels: x right and y down. Core inputs
contain images, masks, geometry and calibration, never supplied PIV velocities.
Each stage receives its pair directory explicitly; no working-directory state
or platform-specific process executor is required.
"""
